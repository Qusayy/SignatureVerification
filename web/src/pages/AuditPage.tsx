import { useEffect, useState } from 'react'
import { type AuditEvent, listAuditEvents } from '../api'

export function AuditPage() {
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listAuditEvents(100)
      .then(setEvents)
      .catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <p className="alert alert--danger">{error}</p>

  return (
    <section className="panel">
      <h2>Verification history</h2>
      <p className="login__sub">
        Append-only. Every check and every decision is kept, including the ones where the employee
        and the model disagreed — those are the most useful records the pilot produces.
      </p>
      <table className="audit">
        <thead>
          <tr>
            <th>When</th>
            <th>Customer</th>
            <th>Employee</th>
            <th>Score</th>
            <th>Advice</th>
            <th>Decision</th>
            <th>Agreed</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.id} className={e.agreed_with_model === false ? 'row--disagree' : undefined}>
              <td>{new Date(e.created_at).toLocaleString()}</td>
              <td>{e.customer_number}</td>
              <td>{e.employee_username}</td>
              <td>{e.score.toFixed(0)}</td>
              <td>
                <span className={`pill pill--${e.band}`}>{e.band}</span>
                {e.suspected_copy && <span className="pill pill--red">copy?</span>}
              </td>
              <td>{e.decision ?? '—'}</td>
              <td>
                {e.agreed_with_model === null ? '—' : e.agreed_with_model ? 'yes' : 'no'}
              </td>
            </tr>
          ))}
          {events.length === 0 && (
            <tr>
              <td colSpan={7}>No verifications recorded yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </section>
  )
}
