import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { type PipelineStage, fetchImage } from '../api'

/**
 * Replays a verification, stage by stage.
 *
 * The score on its own is a number an operator has to take on trust. This shows
 * the whole chain that produced it — what was captured, what was thrown away,
 * what the network actually saw, and how a similarity became a confidence.
 *
 * Two deliberate choices:
 *
 * - **Every frame is real.** Each panel is an image the pipeline genuinely
 *   produced during this verification, fetched from the server, not a mock-up
 *   or a stylised illustration. If the preprocessing changes, this display
 *   changes with it, because there is nothing here to keep in sync.
 * - **Images are preloaded before playback starts.** Streaming them as the
 *   animation runs produces a stutter exactly where attention is highest.
 */

interface Props {
  stages: PipelineStage[]
  /** Restarts playback whenever this changes — one verification, one replay. */
  runId: string
}

/** Which part of the architecture each stage belongs to. */
const CHAPTERS: { id: string; label: string; keys: string[] }[] = [
  { id: 'capture', label: 'Capture', keys: ['capture', 'deskew', 'detect', 'crop'] },
  {
    id: 'clean',
    label: 'Isolate ink',
    keys: ['grayscale', 'illumination', 'binarised', 'lines_removed', 'denoised'],
  },
  { id: 'geometry', label: 'Normalise', keys: ['normalised', 'model_input'] },
  { id: 'network', label: 'Embed', keys: ['embedding'] },
  { id: 'decide', label: 'Compare', keys: ['compare', 'cohort', 'calibration', 'overlay'] },
]

const STAGE_MS = 2000

function chapterOf(key: string): string {
  return CHAPTERS.find((c) => c.keys.includes(key))?.id ?? 'decide'
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
  )
}

/** Turn a metric key into something readable without a lookup table per stage. */
function labelOf(key: string): string {
  return key.replace(/_/g, ' ').replace(/\bdeg\b/, '°').replace(/\bpx\b/, 'px')
}

function formatMetric(value: string | number | boolean | number[]): string {
  if (Array.isArray(value)) return value.map((v) => v.toFixed(3)).join('  ·  ')
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  return value
}

/**
 * Fetch every stage image once, up front.
 *
 * Object URLs are revoked on unmount; these are decrypted biometric images and
 * leaving them alive in browser memory outlives any reason to hold them.
 */
function useStageImages(stages: PipelineStage[]): { urls: Record<string, string>; ready: boolean } {
  const [urls, setUrls] = useState<Record<string, string>>({})
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    const created: string[] = []
    setUrls({})
    setReady(false)

    const withImages = stages.filter((s) => s.image_url)
    if (withImages.length === 0) {
      setReady(true)
      return
    }

    Promise.all(
      withImages.map(async (stage) => {
        try {
          const url = await fetchImage(stage.image_url as string)
          created.push(url)
          return [stage.key, url] as const
        } catch {
          // A stage that will not load must not stop the replay: the panel
          // falls back to its caption, which still explains the step.
          return [stage.key, ''] as const
        }
      }),
    ).then((pairs) => {
      if (cancelled) {
        created.forEach(URL.revokeObjectURL)
        return
      }
      setUrls(Object.fromEntries(pairs.filter(([, url]) => url)))
      setReady(true)
    })

    return () => {
      cancelled = true
      created.forEach(URL.revokeObjectURL)
    }
  }, [stages])

  return { urls, ready }
}

/** Counts a number up when it first appears. Pure decoration; skipped when reduced motion is set. */
function useCountUp(target: number, active: boolean): number {
  const [value, setValue] = useState(active ? target : 0)

  useEffect(() => {
    if (!active) return
    if (prefersReducedMotion()) {
      setValue(target)
      return
    }
    let frame = 0
    const start = performance.now()
    const duration = 700
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      // Ease-out cubic: fast then settling, which reads as arriving at a value
      // rather than sliding past it.
      setValue(target * (1 - Math.pow(1 - t, 3)))
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, active])

  return value
}

// --------------------------------------------------------------------------
// Non-image stage panels
// --------------------------------------------------------------------------

function ComparePanel({ stage, active }: { stage: PipelineStage; active: boolean }) {
  const per = (stage.metrics.per_reference as number[] | undefined) ?? []
  const combined = Number(stage.metrics.combined ?? 0)

  // Similarities live in a narrow band near the top of the range, so plotting
  // them from zero makes every bar look identical. The axis starts at 0.5.
  const floor = 0.5
  const width = (v: number) => `${Math.max(2, Math.min(100, ((v - floor) / (1 - floor)) * 100))}%`

  return (
    <div className="theatre__data">
      <ul className="simbars">
        {per.map((value, i) => (
          <li key={i}>
            <span className="simbars__label">Specimen {i + 1}</span>
            <span className="simbars__track">
              <span
                className="simbars__fill"
                style={{ width: active ? width(value) : '0%', transitionDelay: `${i * 90}ms` }}
              />
            </span>
            <span className="simbars__value">{value.toFixed(3)}</span>
          </li>
        ))}
        <li className="simbars__combined">
          <span className="simbars__label">Combined</span>
          <span className="simbars__track">
            <span
              className="simbars__fill simbars__fill--accent"
              style={{ width: active ? width(combined) : '0%', transitionDelay: `${per.length * 90}ms` }}
            />
          </span>
          <span className="simbars__value">{combined.toFixed(3)}</span>
        </li>
      </ul>
    </div>
  )
}

function ScorePanel({ stage, active }: { stage: PipelineStage; active: boolean }) {
  const isFinal = stage.key === 'calibration'
  const headline = isFinal ? Number(stage.metrics.score ?? 0) : Number(stage.metrics.normalised ?? 0)
  const shown = useCountUp(headline, active)
  const band = String(stage.metrics.band ?? '')

  return (
    <div className="theatre__data">
      <div className={`bignum ${band ? `bignum--${band}` : ''}`}>
        <span className="bignum__value">{isFinal ? shown.toFixed(1) : shown.toFixed(3)}</span>
        <span className="bignum__unit">{isFinal ? '/ 100' : 'normalised'}</span>
      </div>
      {isFinal && band && <div className={`bignum__band bignum__band--${band}`}>{band}</div>}
      {!isFinal && (
        <div className="flowline">
          <span>{Number(stage.metrics.raw_similarity ?? 0).toFixed(3)}</span>
          <span className="flowline__arrow" aria-hidden="true" />
          <span>{String(stage.metrics.method ?? '')}</span>
          <span className="flowline__arrow" aria-hidden="true" />
          <span>{Number(stage.metrics.normalised ?? 0).toFixed(3)}</span>
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Theatre
// --------------------------------------------------------------------------

export function PipelineTheatre({ stages, runId }: Props) {
  const { urls, ready } = useStageImages(stages)
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [collapsed, setCollapsed] = useState(false)
  const timer = useRef<number | null>(null)

  const reduced = useMemo(prefersReducedMotion, [])
  const total = stages.length

  // A new verification always replays from the top.
  useEffect(() => {
    setIndex(0)
    setPlaying(!reduced)
  }, [runId, reduced])

  useEffect(() => {
    if (!playing || !ready || total === 0) return
    if (index >= total - 1) {
      setPlaying(false)
      return
    }
    timer.current = window.setTimeout(() => setIndex((i) => i + 1), STAGE_MS / speed)
    return () => {
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [playing, ready, index, total, speed])

  const go = useCallback(
    (next: number) => {
      setPlaying(false)
      setIndex(Math.max(0, Math.min(total - 1, next)))
    },
    [total],
  )

  const replay = useCallback(() => {
    setIndex(0)
    setPlaying(true)
  }, [])

  if (total === 0) return null

  const stage = stages[index]
  const activeChapter = chapterOf(stage.key)
  const progress = total > 1 ? (index / (total - 1)) * 100 : 100

  return (
    <section className={`theatre ${collapsed ? 'theatre--collapsed' : ''}`} aria-label="How this result was reached">
      <header className="theatre__head">
        <div>
          <h2>How this result was reached</h2>
          <p className="theatre__sub">
            Every panel is an image this verification actually produced, in the order it was
            produced.
          </p>
        </div>
        <button className="btn btn--ghost" onClick={() => setCollapsed((c) => !c)}>
          {collapsed ? 'Show' : 'Hide'}
        </button>
      </header>

      {!collapsed && (
        <>
          <ol className="chapters" aria-hidden="true">
            {CHAPTERS.map((chapter) => {
              const done =
                CHAPTERS.findIndex((c) => c.id === chapter.id) <
                CHAPTERS.findIndex((c) => c.id === activeChapter)
              return (
                <li
                  key={chapter.id}
                  className={`chapters__item ${chapter.id === activeChapter ? 'is-active' : ''} ${
                    done ? 'is-done' : ''
                  }`}
                >
                  <span className="chapters__dot" />
                  {chapter.label}
                </li>
              )
            })}
          </ol>

          <div className="theatre__stage">
            {!ready && <div className="theatre__loading">Assembling the replay…</div>}

            {ready && (
              <div className={`frame frame--${stage.kind}`} key={stage.key}>
                {stage.image_url && urls[stage.key] ? (
                  <img className="frame__img" src={urls[stage.key]} alt={stage.title} />
                ) : stage.kind === 'compare' ? (
                  <ComparePanel stage={stage} active />
                ) : stage.kind === 'score' ? (
                  <ScorePanel stage={stage} active />
                ) : (
                  <div className="theatre__loading">Image unavailable for this step</div>
                )}
                {!reduced && <span className="frame__sweep" aria-hidden="true" />}
              </div>
            )}
          </div>

          <div className="theatre__caption" aria-live="polite">
            <div className="theatre__step">
              Step {index + 1} of {total}
            </div>
            <h3>{stage.title}</h3>
            <p>{stage.caption}</p>
            {Object.keys(stage.metrics).length > 0 && (
              <ul className="chips">
                {Object.entries(stage.metrics).map(([key, value]) => (
                  <li key={key} className="chips__item">
                    <span className="chips__key">{labelOf(key)}</span>
                    <span className="chips__value">{formatMetric(value)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="theatre__transport">
            <div className="scrub" role="progressbar" aria-valuenow={index + 1} aria-valuemin={1} aria-valuemax={total}>
              <span className="scrub__fill" style={{ width: `${progress}%` }} />
            </div>
            <div className="theatre__buttons">
              <button className="btn btn--ghost" onClick={() => go(index - 1)} disabled={index === 0}>
                Back
              </button>
              {index >= total - 1 ? (
                <button className="btn btn--primary" onClick={replay}>
                  Replay
                </button>
              ) : (
                <button className="btn btn--primary" onClick={() => setPlaying((p) => !p)}>
                  {playing ? 'Pause' : 'Play'}
                </button>
              )}
              <button
                className="btn btn--ghost"
                onClick={() => go(index + 1)}
                disabled={index >= total - 1}
              >
                Next
              </button>
              <div className="speed" role="group" aria-label="Playback speed">
                {[0.5, 1, 2].map((s) => (
                  <button
                    key={s}
                    className={`speed__btn ${speed === s ? 'is-active' : ''}`}
                    onClick={() => setSpeed(s)}
                  >
                    {s}×
                  </button>
                ))}
              </div>
            </div>
          </div>

          <ol className="filmstrip">
            {stages.map((s, i) => (
              <li key={s.key}>
                <button
                  className={`filmstrip__cell ${i === index ? 'is-active' : ''} ${
                    i < index ? 'is-done' : ''
                  }`}
                  onClick={() => go(i)}
                  title={s.title}
                >
                  {urls[s.key] ? (
                    <img src={urls[s.key]} alt="" />
                  ) : (
                    <span className="filmstrip__glyph" aria-hidden="true">
                      {s.kind === 'compare' ? '≈' : s.kind === 'score' ? '#' : '·'}
                    </span>
                  )}
                  <span className="filmstrip__label">{s.title}</span>
                </button>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  )
}
