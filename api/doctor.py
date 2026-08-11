"""Diagnose a broken demo stack in one command.

The failure modes this catches all look the same from the browser - a thumbnail
that will not load, a verification that will not run - and none of them say why.

    python -m api.doctor

Each check prints OK, WARN or FAIL, and every FAIL carries the exact command
that fixes it. Exits non-zero when something is actually broken, so it can gate
a start-up script.

The three faults worth knowing about, because they are the ones that bite after
moving the project to another machine or retraining a model:

1. **The image encryption key changed.** Stored images are encrypted at rest.
   Historically an unset ``SV_IMAGE_ENCRYPTION_KEY`` meant a fresh key per
   process, so the seeding run and the serving run disagreed and every image
   request returned 500. The key is now cached on disk, but a database carried
   over from another machine without its key file still hits this.
2. **Stored embeddings are stale.** Retraining changes what an embedding means.
   Nothing errors - scores keep arriving in the normal range and are quietly
   wrong. Only the recorded model version reveals it.
3. **The artifacts do not match.** A checkpoint served with another run's
   cohort statistics or calibrator produces plausible, wrong numbers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from api.db import get_sessionmaker, init_db, schema_drift
from api.models.tables import Customer, CustomerEnrolment, Employee, ReferenceSignature
from api.settings import get_settings

OK, WARN, FAIL = "OK", "WARN", "FAIL"

_MARK = {OK: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: list[str] = field(default_factory=list)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", fix: list[str] | None = None) -> Check:
        check = Check(name, status, detail, fix or [])
        self.checks.append(check)
        return check

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    def render(self) -> str:
        lines = []
        width = max((len(c.name) for c in self.checks), default=0)
        for check in self.checks:
            lines.append(f"[{_MARK[check.status]}] {check.name.ljust(width)}  {check.detail}")
        remedies = [c for c in self.checks if c.fix]
        if remedies:
            lines.append("")
            lines.append("To fix:")
            for check in remedies:
                lines.append(f"\n  {check.name}:")
                for step in check.fix:
                    lines.append(f"    {step}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_secrets(report: Report) -> None:
    settings = get_settings()

    if settings.image_encryption_key:
        report.add("image encryption key", OK, "supplied via SV_IMAGE_ENCRYPTION_KEY")
    elif settings.dev_key_path.exists():
        report.add(
            "image encryption key",
            WARN,
            f"generated and cached at {settings.dev_key_path} (fine for a demo)",
            [
                "For anything beyond a demo, put a stable key in .env:",
                '  python -c "from cryptography.fernet import Fernet; '
                'print(\'SV_IMAGE_ENCRYPTION_KEY=\' + Fernet.generate_key().decode())" >> .env',
                "Do NOT delete the cached key file - existing images become unreadable.",
            ],
        )
    else:
        report.add("image encryption key", OK, "none yet; one will be generated on first write")

    if settings.jwt_secret == "dev-only-insecure-secret-change-me":
        report.add("jwt secret", WARN, "built-in development value")
    elif len(settings.jwt_secret.encode()) < 32:
        report.add(
            "jwt secret",
            FAIL,
            f"{len(settings.jwt_secret.encode())} bytes; HS256 needs 32",
            [
                'python -c "import secrets; print(\'SV_JWT_SECRET=\' + '
                'secrets.token_urlsafe(48))" >> .env'
            ],
        )
    else:
        report.add("jwt secret", OK, f"{len(settings.jwt_secret.encode())} bytes")


def check_artifacts(report: Report) -> None:
    settings = get_settings()

    if not settings.checkpoint_path.exists():
        report.add(
            "checkpoint",
            FAIL,
            f"missing: {settings.checkpoint_path}",
            [
                "Train a model, or point at an existing one:",
                "  SV_CHECKPOINT_PATH=artifacts/<your model>.pt  (in .env)",
                "  python -m ml.embed.train --help",
            ],
        )
        return
    report.add("checkpoint", OK, str(settings.checkpoint_path))

    missing = [
        (name, path)
        for name, path in (
            ("cohort", settings.cohort_path),
            ("calibrator", settings.calibrator_path),
        )
        if not path.exists()
    ]
    for name, path in missing:
        report.add(
            name,
            WARN,
            f"missing: {path} - scores will be raw similarity, not calibrated",
            [
                "Both are written by the benchmark run:",
                f"  python -m ml.eval.benchmark --checkpoint {settings.checkpoint_path} "
                "--split test",
            ],
        )
    # Do the artifacts actually belong to this checkpoint? Until this check
    # existed, `cohort.npz` and `calibrator.json` produced by one training run
    # were served alongside another run's weights, and the only symptom was
    # that two thirds of skilled forgeries scored 99.5 out of 100.
    from ml.embed.provenance import read_weights_id
    from ml.scoring.calibrate import ScoreCalibrator
    from ml.scoring.znorm import CohortNormalizer

    try:
        model_id = read_weights_id(settings.checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a check, not a crash
        report.add("artifact identity", FAIL, f"could not read the checkpoint: {exc}")
        return

    stamps: list[tuple[str, Path, str]] = []
    if settings.cohort_path.exists():
        stamps.append(
            ("cohort", settings.cohort_path, getattr(CohortNormalizer.load(settings.cohort_path), "weights_id", ""))
        )
    if settings.calibrator_path.exists():
        stamps.append(
            ("calibrator", settings.calibrator_path, ScoreCalibrator.load(settings.calibrator_path).weights_id)
        )

    for name, path, stamp in stamps:
        if stamp == model_id:
            report.add(name, OK, f"{path} (weights {model_id})")
        else:
            report.add(
                name,
                FAIL,
                (
                    f"produced by weights {stamp}, but the checkpoint is {model_id}"
                    if stamp
                    else f"carries no weights stamp and cannot be matched to {model_id}"
                ),
                [
                    "A cohort or calibrator from another run gives scores in a normal",
                    "range that mean nothing. Regenerate both against this checkpoint:",
                    f"  python -m ml.eval.benchmark --checkpoint {settings.checkpoint_path} "
                    "--split test",
                    "  python -m api.reenrol --apply",
                ],
            )


def check_model(report: Report) -> str:
    from api.services.inference import get_service

    service = get_service()
    if not service.is_ready:
        report.add(
            "model load",
            FAIL,
            service.load_error or "not loaded",
            ["Fix the checkpoint path above, then re-run this command."],
        )
        return ""

    status = service.status()
    report.add("model load", OK, f"{status['model_version']}")
    if not status["calibrated"]:
        report.add("calibration", WARN, "placeholder calibrator; scores are uncalibrated")
    else:
        report.add("calibration", OK, "isotonic calibrator loaded")
    # Not a warning either way. Cohort normalisation is off by default because
    # it measured worse; what matters is that scores are normalised somehow.
    report.add(
        "score normalisation",
        OK if status.get("writer_normalisation") or status["cohort_normalisation"] else WARN,
        ", ".join(
            filter(
                None,
                [
                    "writer-internal" if status.get("writer_normalisation") else "",
                    "cohort S-norm" if status["cohort_normalisation"] else "",
                ],
            )
        )
        or "none; raw similarity is not comparable across customers",
    )
    return status["model_version"] or ""


def check_schema(report: Report) -> bool:
    """Verify the database matches the models. Returns False if it does not.

    This runs before any other database check because a stale schema makes
    every one of them raise, and the resulting traceback names a column instead
    of the actual problem.
    """
    drift = schema_drift()

    if drift["missing_columns"]:
        detail = f"missing {', '.join(drift['missing_columns'])}"
        if drift["unexpected_columns"]:
            detail += f"; database still has {', '.join(drift['unexpected_columns'])}"
        report.add(
            "database schema",
            FAIL,
            f"{detail}. `create_all` cannot add a column to an existing table, so "
            "this database predates the current code. Every authenticated request "
            "fails with a 500.",
            [
                "The demo data is regenerated from the manifest, so rebuilding costs",
                "nothing but the seeding time:",
                "  python -m api.seed --reset --manifest data/manifest_real.json \\",
                "      --customers 10 --references 3",
                "",
                "If the rows are not disposable, migrate instead of rebuilding - the",
                "columns above tell you what to ALTER.",
            ],
        )
        return False

    if drift["unexpected_columns"]:
        report.add(
            "database schema",
            WARN,
            f"unused columns present: {', '.join(drift['unexpected_columns'])}",
        )
        return True

    report.add("database schema", OK, "matches the models")
    return True


def check_database(report: Report, session) -> tuple[int, int]:
    employees = session.execute(select(Employee)).scalars().all()
    customers = session.execute(select(Customer)).scalars().all()
    references = session.execute(select(ReferenceSignature)).scalars().all()

    if not employees:
        report.add(
            "operators",
            FAIL,
            "no operator account - nothing can sign in",
            ["python -m api.seed --customers 10 --references 3"],
        )
    else:
        report.add("operators", OK, f"{len(employees)} account(s)")

    if not customers:
        report.add(
            "customers",
            FAIL,
            "no customers enrolled",
            [
                "python -m api.seed --manifest data/manifest_real.json "
                "--customers 10 --references 3"
            ],
        )
    else:
        without = [c for c in customers if not c.references]
        detail = f"{len(customers)} customers, {len(references)} specimens"
        if without:
            report.add(
                "customers",
                FAIL,
                f"{detail}; {len(without)} have no specimen and cannot be verified",
                ["python -m api.seed --reset --manifest data/manifest_real.json --customers 10"],
            )
        else:
            report.add("customers", OK, detail)

    return len(customers), len(references)


def check_storage(report: Report, session) -> None:
    """Try to actually decrypt every stored image. This is the 500 hunt."""
    from api.services.storage import get_store

    references = session.execute(select(ReferenceSignature)).scalars().all()
    if not references:
        return

    store = get_store()
    ok = missing = undecryptable = 0
    for reference in references:
        for key in (reference.image_key, reference.canvas_key):
            try:
                store.get_image(key)
                ok += 1
            except FileNotFoundError:
                missing += 1
            except ValueError:
                undecryptable += 1

    if undecryptable:
        report.add(
            "stored images",
            FAIL,
            f"{undecryptable} cannot be decrypted - the encryption key does not match "
            "the one they were written with",
            [
                "The images are unrecoverable without the original key. Either restore",
                "the original SV_IMAGE_ENCRYPTION_KEY (or data/.image_key), or rebuild",
                "the demo data from scratch:",
                "  python -m api.seed --reset --manifest data/manifest_real.json "
                "--customers 10 --references 3",
            ],
        )
    elif missing:
        report.add(
            "stored images",
            FAIL,
            f"{missing} referenced images are absent from storage",
            [
                "The database and the image store disagree. Rebuild the demo data:",
                "  python -m api.seed --reset --manifest data/manifest_real.json "
                "--customers 10 --references 3",
            ],
        )
    else:
        report.add("stored images", OK, f"{ok} decrypted successfully")


def check_freshness(report: Report, session, current_version: str) -> None:
    if not current_version:
        return

    enrolments = session.execute(select(CustomerEnrolment)).scalars().all()
    customers = session.execute(select(Customer)).scalars().all()
    if not customers:
        return

    stale = [e for e in enrolments if e.model_version != current_version]
    if stale:
        versions = sorted({e.model_version or "(unrecorded)" for e in stale})
        report.add(
            "embedding freshness",
            FAIL,
            f"{len(stale)} customer(s) hold embeddings from {', '.join(versions)}, "
            f"but the loaded model is {current_version}. Scores would look normal "
            "and be meaningless.",
            [
                "python -m api.reenrol --check     # confirm the scope",
                "python -m api.reenrol --apply     # re-embed every stored specimen",
            ],
        )
    else:
        report.add("embedding freshness", OK, f"all enrolments match {current_version}")

    missing_enrolment = [c for c in customers if c.references and c.enrolment is None]
    if missing_enrolment:
        report.add(
            "cohort statistics",
            WARN,
            f"{len(missing_enrolment)} customer(s) lack cached statistics; "
            "verification falls back to raw similarity",
            ["python -m api.reenrol --apply"],
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run() -> Report:
    report = Report()
    check_secrets(report)
    check_artifacts(report)
    version = check_model(report)

    init_db()
    if not check_schema(report):
        # Every remaining check queries these tables. Running them would bury
        # the real fault under a stack trace.
        return report

    session = get_sessionmaker()()
    try:
        customers, _ = check_database(report, session)
        if customers:
            check_storage(report, session)
            check_freshness(report, session, version)
    finally:
        session.close()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the running stack")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as failures",
    )
    args = parser.parse_args()

    report = run()
    print(report.render())

    if report.failed:
        print("\nSomething is broken. Work through the fixes above, top to bottom.")
        sys.exit(1)
    if args.strict and any(c.status == WARN for c in report.checks):
        print("\nWarnings present and --strict was requested.")
        sys.exit(1)
    print("\nStack looks healthy.")


if __name__ == "__main__":
    main()
