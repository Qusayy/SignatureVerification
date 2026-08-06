import { useState } from 'react'
import { ApiError, type Employee, login } from '../api'

interface Props {
  onSignedIn: (employee: Employee) => void
}

export function LoginPage({ onSignedIn }: Props) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      onSignedIn(await login(username, password))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Sign-in failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="login" onSubmit={submit}>
      <h1>Signature Verification</h1>
      <p className="login__sub">
        Every check is recorded against your name. The system advises; you decide.
      </p>
      <label>
        Username
        <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required />
      </label>
      <label>
        Password
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </label>
      {error && <p className="alert alert--danger">{error}</p>}
      <button className="btn btn--primary" disabled={busy}>
        {busy ? 'Signing in…' : 'Sign in'}
      </button>
    </form>
  )
}
