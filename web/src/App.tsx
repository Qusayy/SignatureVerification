import { useEffect, useState } from 'react'
import { type Employee, type Health, getHealth, logout } from './api'
import { AuditPage } from './pages/AuditPage'
import { LoginPage } from './pages/LoginPage'
import { VerifyPage } from './pages/VerifyPage'

type Tab = 'verify' | 'audit'

export function App() {
  const [employee, setEmployee] = useState<Employee | null>(null)
  const [tab, setTab] = useState<Tab>('verify')
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    if (!employee) return
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [employee])

  if (!employee) return <LoginPage onSignedIn={setEmployee} />

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">Signature Verification</div>
        <nav className="topbar__nav">
          <button className={tab === 'verify' ? 'active' : ''} onClick={() => setTab('verify')}>
            Verify
          </button>
          <button className={tab === 'audit' ? 'active' : ''} onClick={() => setTab('audit')}>
            History
          </button>
        </nav>
        <div className="topbar__user">
          {employee.full_name || employee.username}
          <button
            className="link"
            onClick={() => {
              logout()
              setEmployee(null)
            }}
          >
            Sign out
          </button>
        </div>
      </header>

      {health && !health.model_loaded && (
        <p className="alert alert--danger banner">
          The model is not loaded — verification is unavailable. {health.error}
        </p>
      )}
      {health?.model_loaded && !health.calibrated && (
        <p className="alert alert--warn banner">
          This model has not been calibrated. Scores are indicative only and must not be treated as
          confidence values.
        </p>
      )}

      <main>{tab === 'verify' ? <VerifyPage /> : <AuditPage />}</main>

      <footer className="footer">
        Advisory system. It never accepts or rejects a signature — the employee always decides, and
        every decision is logged.
        {health?.model_version && <span> Model {health.model_version}.</span>}
      </footer>
    </div>
  )
}
