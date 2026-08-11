import { useState } from 'react'
import type { VerificationResult } from '../api'
import { AuthedImage } from './AuthedImage'
import { ScoreGauge } from './ScoreGauge'

interface Props {
  result: VerificationResult
  onDecision: (decision: 'accept' | 'reject', note: string) => void
  submitting: boolean
  decided: 'accept' | 'reject' | null
}

const WARNING_TEXT: Record<string, string> = {
  single_reference_lower_confidence:
    'Only one specimen signature is on file for this customer, so confidence is lower than usual.',
  uncalibrated_score_placeholder:
    'This model has not been calibrated. The number shown is indicative only, not a confidence.',
  score_not_writer_normalised:
    'Only one specimen is on file, so there is no way to measure how consistently this ' +
    'customer signs. This score is less comparable with other customers than usual.',
  ink_outside_detected_region:
    'Some handwriting fell outside the detected region. Check the located area, and draw ' +
    'the box manually if part of the signature was cut off.',
  suspected_photocopy_of_stored_specimen:
    'This looks like a copy of the stored specimen rather than a freshly written signature.',
  very_dark_crop_possible_background_leak:
    'The captured area is unusually dark — background may have been picked up as ink.',
  blank_or_near_blank: 'Very little ink was found in the captured area.',
}

/**
 * Put the margin in words.
 *
 * The score is driven by the gap between how well the query matches the
 * specimens and how well the specimens match each other — not by the raw
 * similarity, which is what an operator's eye goes to first.
 */
function describeMargin(margin: number): string {
  if (margin >= 0.02) return 'Comfortably within this customer’s normal variation.'
  if (margin >= 0) return 'About as consistent as this customer’s own specimens are.'
  if (margin >= -0.02) return 'Slightly outside this customer’s normal variation — worth a look.'
  return 'Noticeably less consistent than this customer’s own specimens.'
}

export function ResultPanel({ result, onDecision, submitting, decided }: Props) {
  const [note, setNote] = useState('')
  const [view, setView] = useState<'overlay' | 'crop'>('overlay')

  return (
    <section className="result" aria-live="polite">
      <div className="result__header">
        <ScoreGauge score={result.score} band={result.band} calibrated={result.calibrated} />
        <div className="result__summary">
          <h2>{result.guidance}</h2>
          <p className="result__reason">{result.reason}</p>

          {result.suspected_copy && (
            <p className="alert alert--danger">
              <strong>Suspected copy.</strong> A genuine signature is never an exact match to the
              stored specimen. Treat this as a possible photocopy or paste, not as a strong match.
            </p>
          )}

          {result.warnings
            .filter((w) => w !== 'suspected_photocopy_of_stored_specimen')
            .map((w) => (
              <p key={w} className="alert alert--warn">
                {WARNING_TEXT[w] ?? w}
              </p>
            ))}

          <dl className="result__meta">
            <div>
              <dt>Specimens compared</dt>
              <dd>{result.comparison.n_references}</dd>
            </div>
            <div>
              <dt>Closest specimen</dt>
              <dd>{(result.comparison.max_similarity * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Average across specimens</dt>
              <dd>{(result.comparison.mean_similarity * 100).toFixed(1)}%</dd>
            </div>
            {result.comparison.writer_normalised && (
              <div>
                <dt>Specimens agree with each other</dt>
                <dd>{(result.comparison.intra_reference_mean * 100).toFixed(1)}%</dd>
              </div>
            )}
            <div>
              <dt>Time</dt>
              <dd>{(result.latency_ms / 1000).toFixed(1)}s</dd>
            </div>
          </dl>

          {/*
            Without the third figure the first two do not explain the score,
            and the panel reads as broken: 89% against the specimens looks like
            a match until you know the customer's own specimens agree at 88.5%.
            The comparison is the point, so state it in words rather than
            leaving the operator to subtract.
          */}
          {result.comparison.writer_normalised && (
            <p className="result__reason">
              This signature matches{' '}
              <strong>
                {(result.comparison.mean_similarity * 100).toFixed(1)}%
              </strong>{' '}
              on average, against stored specimens that match each other{' '}
              <strong>
                {(result.comparison.intra_reference_mean * 100).toFixed(1)}%
              </strong>
              . {describeMargin(
                result.comparison.mean_similarity - result.comparison.intra_reference_mean,
              )}
            </p>
          )}
        </div>
      </div>

      <div className="result__images">
        <div className="tabs" role="tablist">
          <button
            role="tab"
            aria-selected={view === 'overlay'}
            className={view === 'overlay' ? 'active' : ''}
            onClick={() => setView('overlay')}
          >
            Difference overlay
          </button>
          <button
            role="tab"
            aria-selected={view === 'crop'}
            className={view === 'crop' ? 'active' : ''}
            onClick={() => setView('crop')}
          >
            Captured signature
          </button>
        </div>

        {view === 'overlay' ? (
          <figure>
            <AuthedImage src={result.overlay_url} alt="Captured signature overlaid on the stored specimen" />
            <figcaption>
              <span className="swatch swatch--query" /> captured only
              <span className="swatch swatch--reference" /> specimen only
              <span className="swatch swatch--both" /> both agree
            </figcaption>
          </figure>
        ) : (
          <figure>
            <AuthedImage src={result.crop_url} alt="The signature area that was scored" />
            <figcaption>
              {result.detection
                ? `Located automatically (${(result.detection.confidence * 100).toFixed(0)}% confidence)`
                : 'Supplied directly'}
            </figcaption>
          </figure>
        )}

        <div className="references">
          <h3>Stored specimens</h3>
          <div className="references__strip">
            {result.reference_urls.map((url, i) => (
              <figure key={url}>
                <AuthedImage src={url} alt={`Stored specimen ${i + 1}`} />
                <figcaption>{(result.comparison.per_reference[i] * 100).toFixed(0)}%</figcaption>
              </figure>
            ))}
          </div>
        </div>
      </div>

      <div className="decision">
        <h3>Your decision</h3>
        <p className="decision__hint">
          The system does not accept or reject signatures. Record what you decided — it is logged
          against your name and used to improve the model.
        </p>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Optional note (why you decided this)"
          rows={2}
          disabled={decided !== null}
        />
        <div className="decision__buttons">
          <button
            className="btn btn--accept"
            disabled={submitting || decided !== null}
            onClick={() => onDecision('accept', note)}
          >
            {decided === 'accept' ? 'Accepted ✓' : 'Accept signature'}
          </button>
          <button
            className="btn btn--reject"
            disabled={submitting || decided !== null}
            onClick={() => onDecision('reject', note)}
          >
            {decided === 'reject' ? 'Rejected ✓' : 'Reject signature'}
          </button>
        </div>
        {decided && <p className="decision__done">Decision recorded and logged.</p>}
      </div>
    </section>
  )
}
