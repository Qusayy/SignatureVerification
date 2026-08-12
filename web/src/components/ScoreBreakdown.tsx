import type { ScoreDiagnostics } from '../api'

/**
 * The arithmetic behind the score, on screen.
 *
 * A score is a similarity minus a baseline, clipped by a floor, pushed through
 * a calibration curve. When one of those stages misbehaves the number looks
 * arbitrary and there is nothing on the panel to argue with. This shows the
 * whole chain, with the step that decided the outcome marked.
 *
 * Collapsed by default: an operator at a counter does not need it, and the
 * person debugging a surprising score needs all of it.
 */

const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const signed = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

function baselineLabel(source: string): string {
  if (source === 'own') return 'this customer’s own specimens'
  if (source === 'population') return 'a typical customer (only one specimen on file)'
  return 'none — the score has no per-customer baseline'
}

export function ScoreBreakdown({ diagnostics: d }: { diagnostics: ScoreDiagnostics }) {
  const clampedHigh = d.calibrator_clamped === 'above'
  const clampedLow = d.calibrator_clamped === 'below'

  return (
    <>
      {/* The one condition that invalidates everything else on the panel. */}
      {d.inconsistent && (
        <p className="alert alert--danger">
          <strong>This score is not trustworthy.</strong> The signature matches at{' '}
          {pct(d.combined_similarity)}, below the {pct(d.similarity_floor)} that genuine
          signatures reach on this model — yet the score is {d.normalised >= 0 ? 'high' : 'high'}.
          Do not act on it. Send this breakdown to whoever maintains the system.
        </p>
      )}

      <details className="breakdown">
        <summary>Score breakdown</summary>

        <dl className="breakdown__rows">
          <div>
            <dt>Similarity to the specimens</dt>
            <dd>{pct(d.combined_similarity)}</dd>
          </div>
          <div>
            <dt>Baseline subtracted</dt>
            <dd>
              {pct(d.baseline)}
              <span className="breakdown__note">{baselineLabel(d.baseline_source)}</span>
            </dd>
          </div>
          <div className={!d.floor_applied && d.baseline_source !== 'none' ? 'is-binding' : ''}>
            <dt>Margin against that baseline</dt>
            <dd>{signed(d.relative_margin)}</dd>
          </div>
          <div className={d.floor_applied ? 'is-binding' : ''}>
            <dt>Margin against the floor</dt>
            <dd>
              {signed(d.absolute_margin)}
              <span className="breakdown__note">
                floor {pct(d.similarity_floor)},{' '}
                {d.floor_measured ? 'measured on this corpus' : 'configured default'}
              </span>
            </dd>
          </div>
          <div>
            <dt>Score is decided by</dt>
            <dd>
              {d.binding_term}
              <span className="breakdown__note">
                {d.baseline_source === 'none'
                  ? 'with no baseline the value is not on the calibration scale at all'
                  : 'whichever of the two is stricter'}
              </span>
            </dd>
          </div>
        </dl>

        {/* Saturation is the failure that makes every score look identical, so
            it gets stated rather than left to be inferred from the numbers. */}
        {(clampedHigh || clampedLow) && (
          <p className="alert alert--warn">
            The value {signed(d.normalised)} falls {clampedHigh ? 'above' : 'below'} the
            calibration curve’s range ({signed(d.calibrator_domain[0])} to{' '}
            {signed(d.calibrator_domain[1])}), so it was clamped to the{' '}
            {clampedHigh ? 'top' : 'bottom'} of the scale. Scores in this region cannot
            distinguish one signature from another — recalibrate before relying on them.
          </p>
        )}

        <p className="breakdown__footer">
          Calibration: {d.calibrator_distinct_scores} distinct scores available, fitted on{' '}
          {d.calibrator_fit_samples[0]} genuine / {d.calibrator_fit_samples[1]} impostor
          comparisons.
          {d.calibrator_fit_samples[0] < 200 && (
            <>
              {' '}
              That is a thin fit — below roughly 200 of each the curve collapses into a few
              coarse steps and stops discriminating near the top of its range.
            </>
          )}{' '}
          Model {d.model_version}.
        </p>
      </details>
    </>
  )
}
