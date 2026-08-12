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
      intra_reference_mean: 0.72,
      writer_normalised: true,
      baseline_source: 'own' as const,
      specimens_disagree: false,
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
    diagnostics: null,
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

describe('specimen agreement', () => {
  it('shows the baseline the score is measured against', () => {
    // Without it the panel reads as broken: a high similarity next to a
    // middling score, with nothing on screen explaining the gap.
    render(<ResultPanel result={result()} onDecision={vi.fn()} submitting={false} decided={null} />)

    expect(screen.getByText('Specimens agree with each other')).toBeInTheDocument()
    // Once in the figures, once in the sentence that explains the margin.
    expect(screen.getAllByText('72.0%')).toHaveLength(2)
    expect(screen.getByText(/Comfortably within/)).toBeInTheDocument()
  })

  it('reads the margin out in words', () => {
    render(
      <ResultPanel
        result={result({
          comparison: {
            ...result().comparison,
            mean_similarity: 0.71,
            intra_reference_mean: 0.72,
          },
        })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/Slightly outside/)).toBeInTheDocument()
  })

  it('omits the comparison when there is only one specimen', () => {
    render(
      <ResultPanel
        result={result({
          comparison: {
            ...result().comparison,
            n_references: 1,
            single_reference: true,
            writer_normalised: false,
            baseline_source: 'none' as const,
            intra_reference_mean: 0,
          },
        })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.queryByText('Specimens agree with each other')).not.toBeInTheDocument()
  })
})

describe('single specimen', () => {
  const oneSpecimen = () =>
    result({
      warnings: ['score_uses_population_baseline'],
      comparison: {
        ...result().comparison,
        n_references: 1,
        single_reference: true,
        writer_normalised: false,
        baseline_source: 'population' as const,
        intra_reference_mean: 0.9575,
      },
    })

  it('labels a population baseline as not being this customer', () => {
    render(
      <ResultPanel result={oneSpecimen()} onDecision={vi.fn()} submitting={false} decided={null} />,
    )
    expect(screen.getByText('Typical customer consistency')).toBeInTheDocument()
    expect(screen.queryByText('Specimens agree with each other')).not.toBeInTheDocument()
  })

  it('warns that the score rests on a weaker basis', () => {
    render(
      <ResultPanel result={oneSpecimen()} onDecision={vi.fn()} submitting={false} decided={null} />,
    )
    expect(screen.getByText(/own consistency cannot be measured/i)).toBeInTheDocument()
  })
})

describe('score breakdown', () => {
  const diagnostics = (over = {}) => ({
    combined_similarity: 0.81,
    baseline: 0.72,
    baseline_source: 'own',
    similarity_floor: 0.6,
    floor_measured: true,
    relative_margin: 0.09,
    absolute_margin: 0.21,
    binding_term: 'relative margin',
    floor_applied: false,
    normalised: 0.09,
    calibrator_domain: [-0.7, 0.15] as [number, number],
    calibrator_clamped: null,
    calibrator_distinct_scores: 42,
    calibrator_fit_samples: [500, 500] as [number, number],
    model_version: 'signet@abc123',
    inconsistent: false,
    ...over,
  })

  it('shows the arithmetic behind the score', () => {
    render(
      <ResultPanel
        result={result({ diagnostics: diagnostics() })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText('Score breakdown')).toBeInTheDocument()
    expect(screen.getByText('Baseline subtracted')).toBeInTheDocument()
    expect(screen.getByText('relative margin')).toBeInTheDocument()
  })

  it('shouts when a low similarity produced a confident score', () => {
    render(
      <ResultPanel
        result={result({ diagnostics: diagnostics({ inconsistent: true, combined_similarity: 0.09 }) })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/not trustworthy/i)).toBeInTheDocument()
    expect(screen.getByText(/Do not act on it/i)).toBeInTheDocument()
  })

  it('explains saturation when the calibrator clamped the value', () => {
    render(
      <ResultPanel
        result={result({ diagnostics: diagnostics({ calibrator_clamped: 'above', normalised: 0.9 }) })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/clamped to the/i)).toBeInTheDocument()
    expect(screen.getByText(/cannot\s+distinguish one signature from another/i)).toBeInTheDocument()
  })

  it('flags a thin calibration fit', () => {
    render(
      <ResultPanel
        result={result({ diagnostics: diagnostics({ calibrator_fit_samples: [72, 96] }) })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/thin fit/i)).toBeInTheDocument()
  })
})

describe('no baseline available', () => {
  const noBaseline = {
    combined_similarity: 0.047,
    baseline: 0,
    baseline_source: 'none',
    similarity_floor: 0.6,
    floor_measured: false,
    relative_margin: 0.047,
    absolute_margin: -0.553,
    binding_term: 'nothing — the score has no baseline and is not comparable',
    floor_applied: true,
    normalised: -0.553,
    calibrator_domain: [-0.7, 0.15] as [number, number],
    calibrator_clamped: null,
    calibrator_distinct_scores: 8,
    calibrator_fit_samples: [72, 96] as [number, number],
    model_version: 'signet@abc123',
    inconsistent: false,
  }

  it('withholds the number rather than guessing one', () => {
    // The reported failure: 4.7% similarity, no baseline, score 69/100.
    render(
      <ResultPanel
        result={result({
          calibrated: false,
          warnings: ['score_scale_unavailable'],
          diagnostics: noBaseline,
        })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.getByText('uncalibrated')).toBeInTheDocument()
    expect(screen.getByText(/no scale on which to express a confidence/i)).toBeInTheDocument()
  })

  it('says plainly that nothing decided the score', () => {
    render(
      <ResultPanel
        result={result({ calibrated: false, diagnostics: noBaseline })}
        onDecision={vi.fn()}
        submitting={false}
        decided={null}
      />,
    )
    expect(screen.getByText(/not on the calibration scale at all/i)).toBeInTheDocument()
  })
})
