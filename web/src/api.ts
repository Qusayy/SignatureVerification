/**
 * API client.
 *
 * The token is held in memory only, never in localStorage. A shared terminal is
 * shared between shifts, and a token persisted in browser storage outlives the
 * operator who signed in.
 */

export interface Employee {
  id: string
  username: string
  full_name: string
  location: string
}

export interface Comparison {
  raw: number
  max_similarity: number
  mean_similarity: number
  min_similarity: number
  per_reference: number[]
  n_references: number
  single_reference: boolean
}

export interface Detection {
  bbox: { x: number; y: number; width: number; height: number }
  confidence: number
  method: string
}

export interface VerificationResult {
  event_id: string
  score: number
  band: 'green' | 'amber' | 'red'
  guidance: string
  reason: string
  comparison: Comparison
  detection: Detection | null
  warnings: string[]
  suspected_copy: boolean
  calibrated: boolean
  model_version: string
  latency_ms: number
  crop_url: string
  overlay_url: string
  page_url: string | null
  reference_urls: string[]
  advisory_only: true
}

export interface CustomerReference {
  id: string
  image_url: string
  canvas_url: string
  captured_at: string | null
  source: string
  created_at: string
}

export interface Customer {
  id: string
  customer_number: string
  full_name: string
  script: string
  n_references: number
  enrolled: boolean
  references: CustomerReference[]
}

export interface Health {
  status: string
  model_loaded: boolean
  model_version: string | null
  cohort_normalisation: boolean
  calibrated: boolean
  advisory_only: boolean
  warnings: string[]
  error: string | null
}

export interface AuditEvent {
  id: string
  customer_number: string
  employee_username: string
  score: number
  band: string
  suspected_copy: boolean
  n_references: number
  warnings: string[]
  model_version: string
  created_at: string
  decision: string | null
  decision_note: string | null
  agreed_with_model: boolean | null
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

let accessToken: string | null = null

export function setToken(token: string | null): void {
  accessToken = token
}

export function getToken(): string | null {
  return accessToken
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)

  const response = await fetch(path, { ...init, headers })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new ApiError(detail, response.status)
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

export async function login(username: string, password: string): Promise<Employee> {
  const body = new URLSearchParams({ username, password })
  const response = await fetch('/api/auth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!response.ok) {
    throw new ApiError(response.status === 401 ? 'Incorrect username or password' : 'Sign-in failed', response.status)
  }
  const payload = await response.json()
  setToken(payload.access_token)
  return payload.employee
}

export function logout(): void {
  setToken(null)
}

export const getHealth = (): Promise<Health> => request<Health>('/api/health')

export const searchCustomers = (q: string): Promise<Customer[]> =>
  request<Customer[]>(`/api/customers?q=${encodeURIComponent(q)}`)

export const getCustomer = (customerNumber: string): Promise<Customer> =>
  request<Customer>(`/api/customers/${encodeURIComponent(customerNumber)}`)

export interface VerifyOptions {
  customerNumber: string
  file: File
  isFullPage: boolean
  bbox?: { x: number; y: number; width: number; height: number }
}

export async function verify(options: VerifyOptions): Promise<VerificationResult> {
  const form = new FormData()
  form.append('customer_number', options.customerNumber)
  form.append('file', options.file)
  form.append('is_full_page', String(options.isFullPage))
  if (options.bbox) {
    form.append('bbox_x', String(Math.round(options.bbox.x)))
    form.append('bbox_y', String(Math.round(options.bbox.y)))
    form.append('bbox_width', String(Math.round(options.bbox.width)))
    form.append('bbox_height', String(Math.round(options.bbox.height)))
  }
  return request<VerificationResult>('/api/verify', { method: 'POST', body: form })
}

export function recordDecision(
  eventId: string,
  decision: 'accept' | 'reject',
  note: string,
  secondsToDecide: number | null,
): Promise<unknown> {
  return request(`/api/verify/${eventId}/decision`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, note, seconds_to_decide: secondsToDecide }),
  })
}

export const listAuditEvents = (limit = 50): Promise<AuditEvent[]> =>
  request<AuditEvent[]>(`/api/audit/events?limit=${limit}`)

/**
 * Fetch an authenticated image as an object URL.
 *
 * Signature images require a bearer token, so a plain <img src> cannot load
 * them. Callers must revoke the returned URL when done.
 */
export async function fetchImage(url: string): Promise<string> {
  const headers = new Headers()
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  const response = await fetch(url, { headers })
  if (!response.ok) throw new ApiError(`Could not load image (${response.status})`, response.status)
  return URL.createObjectURL(await response.blob())
}
