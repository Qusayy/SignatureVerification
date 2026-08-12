import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { VerificationResult } from '../api'
import { ResultPanel } from '../components/ResultPanel'
import { ScoreGauge } from '../components/ScoreGauge'

vi.mock('../components/AuthedImage', () => ({
  AuthedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}))

function result(overrides: Partial<VerificationResult> = {}): VerificationResult {
  return {
    event_id: 'e1',
    score: 88,
    band: 'green',
    guidance: 'Consistent with the stored specimen(s).',
    reason: 'Stroke shape and proportions are consistent with all 3 stored specimens.',
    comparison: {
      similarity: 0.86,
      max_similarity: 0.86,
      mean_similarity: 0.86,
      min_similarity: 0.86,
      per_reference: [0.86],
      // One specimen per customer is the deployed protocol, so it is the
      // default the suite exercises.
      n_references: 1,
      single_reference: true,
      scoring_version: 2,
    },
    detection: {
      bbox: { x: 10, y: 20, width: 300, height: 90 },
      confidence: 0.72,
      method: 'heuristic',
    },
    warnings: [],
    suspected_copy: false,
    calibrated: true,
    model_version: 'signet@abc12345',
    latency_ms: 1400,
    crop_url: '/api/images/crop',
    overlay_url: '/api/images/overlay',
    page_url: '/api/images/page',
    reference_urls: ['/api/images/r1', '/api/images/r2', '/api/images/r3'],
    stages: [],
    diagnostics: {
      similarity: 0.86,
      score: 88,
      band: 'green',
      green_min: 71.7,
      red_max: 22.0,
      green_max_far: 0.05,
      red_max_frr: 0.05,
      genuine_share_at_or_above: 0.42,
      impostor_share_at_or_above: 0.06,
      calibrator_domain: [0.73, 0.99] as [number, number],
      calibrator_clamped: null,
      calibrator_distinct_scores: 637,
      calibrator_fit_samples: [500, 600] as [number, number],
      calibrator_thin_fit: false,
      protocol_references: 1,
      model_version: 'signet@abc123',
    },
    advisory_only: true,
    ...overrides,
  }
}

describe('ResultPanel', () => {
  it('always tells the employee the system does not decide', () => {
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)
    expect(screen.getByText(/does not accept or reject/i)).toBeInTheDocument()
    expect(screen.getByText(/Advisory only — you decide/i)).toBeInTheDocument()
  })

  it('offers both accept and reject, neither preselected', () => {
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)
    expect(screen.getByRole('button', { name: /accept signature/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /reject signature/i })).toBeEnabled()
  })

  it('surfaces a suspected copy as a fraud warning rather than a strong match', () => {
    render(
      <ResultPanel
        result={result({ suspected_copy: true, score: 92 })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/Suspected copy/i)).toBeInTheDocument()
    expect(screen.getByText(/never an exact match/i)).toBeInTheDocument()
  })

  it('locks the decision buttons once a decision is recorded', () => {
    render(
      <ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided="accept" />,
    )
    expect(screen.getByRole('button', { name: /accepted/i })).toBeDisabled()
    expect(screen.getByText(/Decision recorded and logged/i)).toBeInTheDocument()
  })

  it('says nothing about the specimen count', () => {
    // One specimen is the design point, not a caveat. A warning that fires on
    // every verification tells the operator nothing they can act on.
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)
    expect(screen.queryByText(/only one specimen/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/less reliable/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/uncalibrated/i)).not.toBeInTheDocument()
  })

  it('renders only warnings an operator can act on', () => {
    render(
      <ResultPanel
        result={result({ warnings: ['ink_outside_detected_region', 'blank_or_near_blank'] })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/draw the box manually/i)).toBeInTheDocument()
    expect(screen.getByText(/Very little ink/i)).toBeInTheDocument()
  })
})

describe('ScoreGauge', () => {
  it('always shows the number', () => {
    // It used to render a dash whenever the score could not be put on a
    // meaningful scale, which after single-specimen became the norm was every
    // verification. A dash at the counter is the least useful thing available.
    render(<ScoreGauge score={46} band="amber" greenMin={71.7} redMax={22} />)
    expect(screen.getByText('46')).toBeInTheDocument()
    expect(screen.getByText('out of 100')).toBeInTheDocument()
    expect(screen.queryByText('—')).not.toBeInTheDocument()
  })

  it('states where the bands begin', () => {
    render(<ScoreGauge score={46} band="amber" greenMin={71.7} redMax={22} />)
    expect(screen.getByText(/green from 72/)).toBeInTheDocument()
  })

  it('omits the edges when the calibrator carries none', () => {
    render(<ScoreGauge score={46} band="amber" />)
    expect(screen.queryByText(/green from/)).not.toBeInTheDocument()
  })
})

describe('score in context', () => {
  it('puts the number next to how often each population reaches it', () => {
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)
    expect(screen.getByText(/match at least this well/i)).toBeInTheDocument()
    expect(screen.getByText('42%')).toBeInTheDocument()
    expect(screen.getByText('6%')).toBeInTheDocument()
  })

  it('shows the similarity the score was computed from', () => {
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)
    expect(screen.getByText('Matched the specimen at')).toBeInTheDocument()
    expect(screen.getAllByText('86.0%').length).toBeGreaterThan(0)
  })
})

describe('score breakdown', () => {
  it('shows the two steps and their provenance', () => {
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)
    expect(screen.getByText('Score breakdown')).toBeInTheDocument()
    expect(screen.getByText('Which the calibration curve reads as')).toBeInTheDocument()
    expect(screen.getByText(/at most 5.0% of forgeries reach green/i)).toBeInTheDocument()
    expect(screen.getByText(/1 specimen\(s\) per customer/)).toBeInTheDocument()
  })

  it('explains saturation when the curve clamped the value', () => {
    render(
      <ResultPanel
        result={result({
          diagnostics: { ...result().diagnostics!, calibrator_clamped: 'above' as const },
        })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/clamped to the top/i)).toBeInTheDocument()
    expect(screen.getByText(/cannot distinguish one signature from another/i)).toBeInTheDocument()
  })

  it('flags a thin calibration fit', () => {
    render(
      <ResultPanel
        result={result({
          diagnostics: {
            ...result().diagnostics!,
            calibrator_thin_fit: true,
            calibrator_fit_samples: [72, 96] as [number, number],
          },
        })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/fewer than 200 comparisons per class/i)).toBeInTheDocument()
  })
})
