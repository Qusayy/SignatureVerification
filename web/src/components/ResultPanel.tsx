import { useState } from 'react'
import type { VerificationResult } from '../api'
import { AuthedImage } from './AuthedImage'
import { ScoreBreakdown } from './ScoreBreakdown'
import { ScoreGauge } from './ScoreGauge'

interface Props {
  result: VerificationResult
  onDecision: (decision: 'accept' | 'reject', note: string) => void
  submitting: boolean
  decided: 'accept' | 'reject' | null
}

/**
 * Only warnings an operator can act on mid-transaction.
 *
 * The list used to include several notes about the score itself being less
 * reliable than usual — single specimen, no per-customer baseline, uncalibrated.
 * With one specimen per customer as the design point rather than the exception,
 * those fired on every verification and said nothing the operator could use.
 * Conditions that genuinely make a score meaningless are now refusals at
 * startup, where an engineer sees them.
 */
const WARNING_TEXT: Record<string, string> = {
  ink_outside_detected_region:
    'Some handwriting fell outside the detected region. Check the located area, and draw ' +
    'the box manually if part of the signature was cut off.',
  suspected_photocopy_of_stored_specimen:
    'This looks like a copy of the stored specimen rather than a freshly written signature.',
  very_dark_crop_possible_background_leak:
    'The captured area is unusually dark — background may have been picked up as ink.',
  blank_or_near_blank: 'Very little ink was found in the captured area.',
}

export function ResultPanel({ result, onDecision, submitting, decided }: Props) {
  const [note, setNote] = useState('')
  const [view, setView] = useState<'overlay' | 'crop'>('overlay')

  return (
    <section className="result" aria-live="polite">
      <div className="result__header">
        <ScoreGauge
          score={result.score}
          band={result.band}
          greenMin={result.diagnostics?.green_min}
          redMax={result.diagnostics?.red_max}
        />
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
              <dt>Matched the specimen at</dt>
              <dd>{(result.comparison.similarity * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Specimens on file</dt>
              <dd>{result.comparison.n_references}</dd>
            </div>
            <div>
              <dt>Time</dt>
              <dd>{(result.latency_ms / 1000).toFixed(1)}s</dd>
            </div>
          </dl>

          {/*
            A bare 46 is not actionable. The same 46 beside "1 in 4 genuine
            signatures match this poorly, and 1 in 5 forgeries match this well"
            is. The shares are measured on held-out signers, so they describe
            this model rather than making a general claim about signatures.
          */}
          {result.diagnostics?.genuine_share_at_or_above != null && (
            <p className="result__reason">
              <strong>
                {(result.diagnostics.genuine_share_at_or_above * 100).toFixed(0)}%
              </strong>{' '}
              of genuine signatures match at least this well
              {result.diagnostics.impostor_share_at_or_above != null && (
                <>
                  , and{' '}
                  <strong>
                    {(result.diagnostics.impostor_share_at_or_above * 100).toFixed(0)}%
                  </strong>{' '}
                  of practised forgeries do
                </>
              )}
              .
            </p>
          )}

          {result.diagnostics && <ScoreBreakdown diagnostics={result.diagnostics} />}
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
