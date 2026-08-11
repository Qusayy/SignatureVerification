import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { PipelineStage } from '../api'
import { PipelineTheatre } from '../components/PipelineTheatre'

const fetchImage = vi.hoisted(() => vi.fn())

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchImage,
}))

function stage(overrides: Partial<PipelineStage> & { key: string }): PipelineStage {
  return {
    title: overrides.key,
    caption: `caption for ${overrides.key}`,
    kind: 'image',
    image_url: `/api/images/${overrides.key}`,
    metrics: {},
    ...overrides,
  }
}

const STAGES: PipelineStage[] = [
  stage({ key: 'capture', title: 'Captured image', metrics: { pixels: '690x298' } }),
  stage({ key: 'binarised', title: 'Adaptive threshold', metrics: { ink_fraction: 0.0731 } }),
  stage({
    key: 'compare',
    title: 'Compared to specimens',
    kind: 'compare',
    image_url: null,
    metrics: { per_reference: [0.954, 0.933, 0.944], combined: 0.9489 },
  }),
  stage({
    key: 'calibration',
    title: 'Calibrated to a score',
    kind: 'score',
    image_url: null,
    metrics: { score: 99.5, band: 'green', normalised: 4.5355 },
  }),
]

beforeEach(() => {
  fetchImage.mockReset()
  fetchImage.mockImplementation(async (url: string) => `blob:${url}`)
  // Assigned rather than stubbed wholesale: jsdom has no object-URL support,
  // but replacing the whole URL global also removes the constructor that
  // jsdom itself needs.
  URL.createObjectURL = vi.fn(() => 'blob:stub')
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PipelineTheatre', () => {
  it('renders nothing when there are no stages', () => {
    const { container } = render(<PipelineTheatre stages={[]} runId="e1" />)
    expect(container).toBeEmptyDOMElement()
  })

  it('opens on the first stage and shows its caption', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())
    expect(screen.getByText('caption for capture')).toBeInTheDocument()
    expect(screen.getByText('Step 1 of 4')).toBeInTheDocument()
  })

  it('preloads every stage image exactly once', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)

    // Two of the four stages are data panels with no image to fetch.
    await waitFor(() => expect(fetchImage).toHaveBeenCalledTimes(2))
    expect(fetchImage).toHaveBeenCalledWith('/api/images/capture')
    expect(fetchImage).toHaveBeenCalledWith('/api/images/binarised')
  })

  it('steps forward and back through the stages', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByRole('heading', { name: 'Adaptive threshold' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Back' }))
    expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument()
  })

  it('disables Back on the first stage', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled()
  })

  it('renders per-specimen similarities on the compare stage', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Compared to specimens' }))

    expect(screen.getByText('Specimen 1')).toBeInTheDocument()
    expect(screen.getByText('0.954')).toBeInTheDocument()
    expect(screen.getByText('Combined')).toBeInTheDocument()
  })

  it('shows the final score and band on the last stage', async () => {
    const { container } = render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument(),
    )

    fireEvent.click(screen.getByRole('button', { name: 'Calibrated to a score' }))

    await waitFor(() => expect(screen.getByText('99.5')).toBeInTheDocument())
    expect(screen.getByText('/ 100')).toBeInTheDocument()
    // The band also appears as a metric chip, so target the badge itself.
    expect(container.querySelector('.bignum__band')).toHaveTextContent('green')
    // The end of the replay offers a rewind, not a play.
    expect(screen.getByRole('button', { name: 'Replay' })).toBeInTheDocument()
  })

  it('surfaces stage metrics as chips', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Adaptive threshold' }))

    expect(screen.getByText('ink fraction')).toBeInTheDocument()
    expect(screen.getByText('0.0731')).toBeInTheDocument()
  })

  it('keeps playing when an image fails to load', async () => {
    // A stage image that 404s must not take the whole explanation down with it.
    fetchImage.mockRejectedValueOnce(new Error('Could not load image (500)'))
    render(<PipelineTheatre stages={STAGES} runId="e1" />)

    await waitFor(() =>
      expect(screen.getByText('Image unavailable for this step')).toBeInTheDocument(),
    )
    expect(screen.getByText('caption for capture')).toBeInTheDocument()
  })

  it('collapses and restores', async () => {
    render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Hide' }))
    expect(screen.queryByText('caption for capture')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Show' }))
    expect(screen.getByText('caption for capture')).toBeInTheDocument()
  })

  it('restarts from the top when a new verification arrives', async () => {
    const { rerender } = render(<PipelineTheatre stages={STAGES} runId="e1" />)
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Captured image' })).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByRole('heading', { name: 'Adaptive threshold' })).toBeInTheDocument()

    rerender(<PipelineTheatre stages={STAGES} runId="e2" />)
    expect(screen.getByText('Step 1 of 4')).toBeInTheDocument()
  })
})
