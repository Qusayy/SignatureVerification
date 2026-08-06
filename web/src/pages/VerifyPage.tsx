import { useEffect, useRef, useState } from 'react'
import {
  ApiError,
  type Customer,
  type VerificationResult,
  getCustomer,
  recordDecision,
  searchCustomers,
  verify,
} from '../api'
import { AuthedImage } from '../components/AuthedImage'
import { ResultPanel } from '../components/ResultPanel'

export function VerifyPage() {
  const [query, setQuery] = useState('')
  const [matches, setMatches] = useState<Customer[]>([])
  const [customer, setCustomer] = useState<Customer | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isFullPage, setIsFullPage] = useState(true)
  const [result, setResult] = useState<VerificationResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [decided, setDecided] = useState<'accept' | 'reject' | null>(null)
  const shownAt = useRef<number | null>(null)

  // Local preview of the uploaded page, revoked when it changes.
  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (query.trim().length < 1) {
      setMatches([])
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      searchCustomers(query)
        .then((found) => !cancelled && setMatches(found))
        .catch(() => !cancelled && setMatches([]))
    }, 200)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [query])

  async function selectCustomer(customerNumber: string) {
    setError(null)
    setResult(null)
    setDecided(null)
    try {
      setCustomer(await getCustomer(customerNumber))
      setMatches([])
      setQuery(customerNumber)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not load customer')
    }
  }

  async function runVerification() {
    if (!customer || !file) return
    setBusy(true)
    setError(null)
    setResult(null)
    setDecided(null)
    try {
      const outcome = await verify({
        customerNumber: customer.customer_number,
        file,
        isFullPage,
      })
      setResult(outcome)
      shownAt.current = Date.now()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Verification failed')
    } finally {
      setBusy(false)
    }
  }

  async function submitDecision(decision: 'accept' | 'reject', note: string) {
    if (!result) return
    setBusy(true)
    try {
      const seconds = shownAt.current ? Math.round((Date.now() - shownAt.current) / 1000) : null
      await recordDecision(result.event_id, decision, note, seconds)
      setDecided(decision)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not record the decision')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="verify">
      <section className="panel">
        <h2>1. Find the customer</h2>
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Customer number or name"
          autoFocus
        />
        {matches.length > 0 && (
          <ul className="matches">
            {matches.map((m) => (
              <li key={m.id}>
                <button onClick={() => selectCustomer(m.customer_number)}>
                  <strong>{m.customer_number}</strong> {m.full_name}
                  <span className="matches__meta">
                    {m.n_references} specimen{m.n_references === 1 ? '' : 's'} · {m.script}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {customer && (
          <div className="customer">
            <div className="customer__head">
              <div>
                <strong>{customer.customer_number}</strong> — {customer.full_name}
              </div>
              <div className="customer__tag">{customer.script}</div>
            </div>
            {customer.n_references === 0 ? (
              <p className="alert alert--warn">
                No specimen signature on file. Enrol one before verifying.
              </p>
            ) : (
              <div className="references__strip">
                {customer.references.map((r, i) => (
                  <figure key={r.id}>
                    <AuthedImage src={r.canvas_url} alt={`Stored specimen ${i + 1}`} />
                  </figure>
                ))}
              </div>
            )}
            {customer.n_references === 1 && (
              <p className="alert alert--warn">
                Only one specimen on file. Scores will be less reliable than for customers with
                several.
              </p>
            )}
          </div>
        )}
      </section>

      <section className="panel">
        <h2>2. Capture the signed paper</h2>
        <input
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null)
            setResult(null)
            setDecided(null)
          }}
        />
        <label className="checkbox">
          <input
            type="checkbox"
            checked={isFullPage}
            onChange={(e) => setIsFullPage(e.target.checked)}
          />
          This is a full form — find the signature automatically
        </label>

        {preview && (
          <figure className="preview">
            <img src={preview} alt="Uploaded page" />
          </figure>
        )}

        <button
          className="btn btn--primary"
          disabled={!customer || !file || busy || customer.n_references === 0}
          onClick={runVerification}
        >
          {busy ? 'Checking…' : 'Check signature'}
        </button>
        {error && <p className="alert alert--danger">{error}</p>}
      </section>

      {result && (
        <ResultPanel
          result={result}
          onDecision={submitDecision}
          submitting={busy}
          decided={decided}
        />
      )}
    </div>
  )
}
