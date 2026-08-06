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
      raw: 0.81,
      max_similarity: 0.86,
      mean_similarity: 0.76,
      min_similarity: 0.7,
      per_reference: [0.86, 0.75, 0.7],
      n_references: 3,
      single_reference: false,
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
        result={result({ suspected_copy: true, score: 99.6 })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/Suspected copy/i)).toBeInTheDocument()
    expect(screen.getByText(/never an exact match/i)).toBeInTheDocument()
  })

  it('explains the single-specimen warning in plain language', () => {
    render(
      <ResultPanel
        result={result({ warnings: ['single_reference_lower_confidence'] })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/Only one specimen signature is on file/i)).toBeInTheDocument()
  })

  it('locks the decision buttons once a decision is recorded', () => {
    render(
      <ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided="accept" />,
    )
    expect(screen.getByRole('button', { name: /accepted/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /reject signature/i })).toBeDisabled()
    expect(screen.getByText(/Decision recorded and logged/i)).toBeInTheDocument()
  })
})

describe('ScoreGauge', () => {
  it('hides the number when the model is uncalibrated', () => {
    render(<ScoreGauge score={88} band="green" calibrated={false} />)
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.getByText('uncalibrated')).toBeInTheDocument()
  })

  it('shows the score when calibrated', () => {
    render(<ScoreGauge score={88} band="green" calibrated />)
    expect(screen.getByText('88')).toBeInTheDocument()
    expect(screen.getByText('out of 100')).toBeInTheDocument()
  })
})
