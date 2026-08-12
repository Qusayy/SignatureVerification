interface Props {
  score: number
  band: 'green' | 'amber' | 'red'
  /** Band edges on the 0-100 scale, from the calibrator. Drawn as ticks. */
  greenMin?: number
  redMax?: number
}

const BAND_LABEL: Record<Props['band'], string> = {
  green: 'Consistent',
  amber: 'Inconclusive',
  red: 'Not consistent',
}

/**
 * The confidence dial.
 *
 * Deliberately labelled "Advisory" and never "Approved" or "Verified": the
 * wording is the main thing stopping an employee from treating the number as a
 * decision the system has already made.
 *
 * **The number is always shown.** It used to be replaced by a dash whenever the
 * scoring layer could not put the value on a meaningful scale — which, once
 * single-specimen customers became the norm, was every verification. A dash at
 * the counter is the worst outcome available: the operator has already captured
 * the signature and waited, and gets nothing to act on. A service that cannot
 * score now refuses at startup instead, where an engineer sees it.
 *
 * The band ticks matter more than they look. Without them a 46 is a bare
 * number; with them it is visibly just below the green threshold, which is the
 * difference between "unclear" and "borderline, look closely".
 */
export function ScoreGauge({ score, band, greenMin, redMax }: Props) {
  const radius = 68
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(100, score))
  const offset = circumference * (1 - clamped / 100)

  // The arc starts at 12 o'clock and runs clockwise, so a score of s sits at
  // (s/100) of the way round from -90 degrees.
  const tick = (value: number) => {
    const angle = ((value / 100) * 360 - 90) * (Math.PI / 180)
    const inner = radius - 10
    const outer = radius + 10
    return {
      x1: 80 + inner * Math.cos(angle),
      y1: 80 + inner * Math.sin(angle),
      x2: 80 + outer * Math.cos(angle),
      y2: 80 + outer * Math.sin(angle),
    }
  }

  return (
    <div className={`gauge gauge--${band}`}>
      <svg viewBox="0 0 160 160" role="img" aria-label={`Advisory confidence ${clamped} out of 100`}>
        <circle className="gauge__track" cx="80" cy="80" r={radius} />
        <circle
          className="gauge__value"
          cx="80"
          cy="80"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 80 80)"
        />
        {redMax !== undefined && redMax > 0 && (
          <line className="gauge__tick gauge__tick--red" {...tick(redMax)} />
        )}
        {greenMin !== undefined && greenMin > 0 && (
          <line className="gauge__tick gauge__tick--green" {...tick(greenMin)} />
        )}
        <text className="gauge__number" x="80" y="76" textAnchor="middle">
          {clamped.toFixed(0)}
        </text>
        <text className="gauge__caption" x="80" y="98" textAnchor="middle">
          out of 100
        </text>
      </svg>
      <div className="gauge__band">{BAND_LABEL[band]}</div>
      {greenMin !== undefined && greenMin > 0 && (
        <div className="gauge__edges">
          green from {greenMin.toFixed(0)} · red at or below {(redMax ?? 0).toFixed(0)}
        </div>
      )}
      <div className="gauge__advisory">Advisory only — you decide</div>
    </div>
  )
}
