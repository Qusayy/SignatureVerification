interface Props {
  score: number
  band: 'green' | 'amber' | 'red'
  calibrated: boolean
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
 */
export function ScoreGauge({ score, band, calibrated }: Props) {
  const radius = 68
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(100, score))
  const offset = circumference * (1 - clamped / 100)

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
        <text className="gauge__number" x="80" y="76" textAnchor="middle">
          {calibrated ? clamped.toFixed(0) : '—'}
        </text>
        <text className="gauge__caption" x="80" y="98" textAnchor="middle">
          {calibrated ? 'out of 100' : 'uncalibrated'}
        </text>
      </svg>
      <div className="gauge__band">{BAND_LABEL[band]}</div>
      <div className="gauge__advisory">Advisory only — you decide</div>
    </div>
  )
}
