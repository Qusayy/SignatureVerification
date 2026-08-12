import type { ScoreDiagnostics } from '../api'

/**
 * How the similarity became this score, on screen.
 *
 * The whole chain is now two steps — cosine to the stored specimen, then a
 * calibration curve — so this is short by design. It used to be long because
 * the chain was: subtract a baseline whose source varied, clip by a floor,
 * then calibrate, with the score depending on which of those bound. Every
 * number here can be checked against the one beside it.
 *
 * Collapsed by default: an operator at a counter does not need it, and the
 * person investigating a surprising score needs all of it.
 */

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function ScoreBreakdown({ diagnostics: d }: { diagnostics: ScoreDiagnostics }) {
  const clamped = d.calibrator_clamped

  return (
    <details className="breakdown">
      <summary>Score breakdown</summary>

      <dl className="breakdown__rows">
        <div className="is-binding">
          <dt>Matched the stored specimen at</dt>
          <dd>{pct(d.similarity)}</dd>
        </div>
        <div>
          <dt>Which the calibration curve reads as</dt>
          <dd>
            {d.score.toFixed(0)} / 100
            <span className="breakdown__note">
              fitted on {d.calibrator_fit_samples[0]} genuine and{' '}
              {d.calibrator_fit_samples[1]} forged comparisons, at{' '}
              {d.protocol_references} specimen(s) per customer
            </span>
          </dd>
        </div>
        <div>
          <dt>Bands</dt>
          <dd>
            green from {d.green_min.toFixed(0)}, red at or below {d.red_max.toFixed(0)}
            <span className="breakdown__note">
              {d.green_max_far != null
                ? `set so at most ${pct(d.green_max_far)} of forgeries reach green`
                : 'configured default, not derived from measurement'}
            </span>
          </dd>
        </div>
        {d.genuine_share_at_or_above != null && (
          <div>
            <dt>Signatures matching at least this well</dt>
            <dd>
              {pct(d.genuine_share_at_or_above)} of genuine
              {d.impostor_share_at_or_above != null && (
                <>, {pct(d.impostor_share_at_or_above)} of forgeries</>
              )}
              <span className="breakdown__note">measured on held-out signers</span>
            </dd>
          </div>
        )}
      </dl>

      {/* Saturation is what makes every score look identical, so it is stated
          rather than left to be inferred from a number that looks fine. */}
      {clamped && (
        <p className="alert alert--warn">
          A similarity of {pct(d.similarity)} falls {clamped} the range the curve was
          fitted over ({pct(d.calibrator_domain[0])} to {pct(d.calibrator_domain[1])}), so
          the score was clamped to the {clamped === 'above' ? 'top' : 'bottom'} of the
          scale. Scores here cannot distinguish one signature from another.
        </p>
      )}

      {d.calibrator_thin_fit && (
        <p className="alert alert--warn">
          The calibration curve was fitted on fewer than 200 comparisons per class, so it
          is coarse and its ceiling is conservative. Widening the validation set — more
          signers, not more samples per signer — is what improves it.
        </p>
      )}

      <p className="breakdown__footer">
        {d.calibrator_distinct_scores} distinct scores available on this curve. Model{' '}
        {d.model_version}.
      </p>
    </details>
  )
}
