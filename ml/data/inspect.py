"""Report the structure of a downloaded dataset before ingesting it.

Third-party signature datasets arrive in wildly inconsistent layouts: one
folder per signer, genuine and forged in sibling folders, everything flat with
the label encoded in the filename, or packed into parquet shards. Guessing
wrong produces a manifest where forgeries are labelled genuine, which trains a
model to accept them and shows up as *good* validation accuracy.

So: look first, ingest second.

    python -m ml.data.inspect --root data/raw/some-dataset

Prints the directory shape, sample filenames, and its best guess at how signer
identity and genuine/forged status are encoded — then tells you the ingest
command to run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

__all__ = ["inspect_root", "GENUINE_HINTS", "FORGERY_HINTS"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

# Substrings that mark genuine vs forged, lowercased.
#
# Forgery hints are checked FIRST and win outright. That is not a stylistic
# choice — several of the genuine hints are substrings of forgery words, and
# without precedence they cancel out and every forgery comes back ambiguous:
#
#   "forgery"   contains "org"  -> would match a genuine hint
#   "full_forg" contains "org"  -> same (this is CEDAR's actual folder name)
#
# Checking forgery first is also the safe direction to be wrong in. A genuine
# sample mislabelled as a forgery makes the model look worse than it is, which
# gets investigated. A forgery mislabelled genuine teaches the model to accept
# forgeries and raises validation accuracy, so nothing catches it.
FORGERY_HINTS = ("forg", "fake", "false", "skilled", "simulated", "_f_", "disguised")
GENUINE_HINTS = ("genuine", "original", "authentic", "real", "true", "org", "_g_", "reference")


# Some datasets use bare single-letter folders. Matched by equality only —
# treating a lone "f" as a substring would classify every filename containing
# the letter f as a forgery.
FORGERY_EXACT = {"f", "forg", "sf"}
GENUINE_EXACT = {"g", "gen", "o", "r"}


def classify(name: str) -> str | None:
    """Guess genuine/forgery from a file or folder name, or None if unclear."""
    lowered = name.lower().strip()
    stem = lowered.rsplit(".", 1)[0] if "." in lowered else lowered

    if stem in FORGERY_EXACT:
        return "skilled_forgery"
    if stem in GENUINE_EXACT:
        return "genuine"
    if any(hint in lowered for hint in FORGERY_HINTS):
        return "skilled_forgery"
    if any(hint in lowered for hint in GENUINE_HINTS):
        return "genuine"
    return None


def inspect_root(root: Path, *, samples: int = 6) -> dict:
    """Walk a dataset directory and describe what is in it."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    images: list[Path] = []
    tabular: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            images.append(path)
        elif suffix in {".parquet", ".arrow", ".csv", ".json", ".jsonl"}:
            tabular.append(path)

    depths = Counter(len(p.relative_to(root).parts) - 1 for p in images)
    by_dir: dict[str, int] = Counter(str(p.parent.relative_to(root)) for p in images)

    # Which path component varies like a signer id, and which like a label?
    label_from_folder = Counter()
    label_from_filename = Counter()
    for path in images:
        parts = path.relative_to(root).parts
        for part in parts[:-1]:
            label = classify(part)
            if label:
                label_from_folder[label] += 1
                break
        label = classify(path.name)
        if label:
            label_from_filename[label] += 1

    return {
        "root": str(root),
        "image_count": len(images),
        "tabular_files": [str(p.relative_to(root)) for p in tabular[:10]],
        "depth_histogram": dict(sorted(depths.items())),
        "distinct_directories": len(by_dir),
        "largest_directories": dict(Counter(by_dir).most_common(8)),
        "labels_from_folder_names": dict(label_from_folder),
        "labels_from_filenames": dict(label_from_filename),
        "sample_paths": [str(p.relative_to(root)) for p in images[:samples]],
        "extensions": dict(Counter(p.suffix.lower() for p in images)),
    }


def _advise(report: dict) -> list[str]:
    """Turn the observations into a concrete next command."""
    lines: list[str] = []

    if report["tabular_files"] and report["image_count"] == 0:
        lines += [
            "This looks like a packed dataset (parquet/arrow), not loose images.",
            "Export it to folders first, for example:",
            "",
            "    from datasets import load_dataset",
            "    ds = load_dataset('<repo id>')",
            "    # then write each split to <root>/<signer>/<genuine|forgery>/*.png",
            "",
            "Then re-run this inspector on the exported folder.",
        ]
        return lines

    folder_labels = report["labels_from_folder_names"]
    filename_labels = report["labels_from_filenames"]

    if folder_labels and sum(folder_labels.values()) > report["image_count"] * 0.5:
        lines.append("Genuine/forged appears to be encoded in FOLDER names.")
        layout = "folder"
    elif filename_labels and sum(filename_labels.values()) > report["image_count"] * 0.5:
        lines.append("Genuine/forged appears to be encoded in FILE names.")
        layout = "filename"
    else:
        lines += [
            "Could NOT determine which files are forgeries.",
            "",
            "This matters more than anything else in the ingest: mislabelling forgeries",
            "as genuine trains the model to accept them, and it shows up as *better*",
            "validation accuracy, not worse. Open the dataset card and confirm the",
            "convention before ingesting. If the dataset genuinely has no forgeries, it",
            "can still be used for writer-identity pretraining via --no-forgeries.",
        ]
        return lines

    depths = report["depth_histogram"]
    likely_depth = max(depths, key=depths.get) if depths else 0
    lines += [
        f"Most images sit {likely_depth} folder level(s) below the root.",
        "",
        "Ingest with:",
        "",
        f"    python -m ml.data.ingest --source <name> --root {report['root']} \\",
        f"        --layout {layout} --script <latin|arabic|mixed> --track <a|b>",
        "",
        "Then verify the counts look right before splitting:",
        "",
        "    python -m ml.data.manifest stats",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe a dataset before ingesting it")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    import json

    report = inspect_root(args.root, samples=args.samples)
    print(json.dumps(report, indent=2))
    print("\n" + "\n".join(_advise(report)))


if __name__ == "__main__":
    main()
