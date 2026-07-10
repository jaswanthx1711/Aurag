const API_URL = ''

// ── token ─────────────────────────────────────────────────────────────────────
export const getToken  = () => (globalThis.window === undefined ? null : globalThis.localStorage.getItem('aurag_token'))
export const setToken  = (t: string) => localStorage.setItem('aurag_token', t)
export const clearToken = () => localStorage.removeItem('aurag_token')

// ── base request ──────────────────────────────────────────────────────────────
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken()
  const isForm = options.body instanceof FormData
  const headers: Record<string, string> = {
    ...(isForm ? {} : { 'Content-Type': 'application/json' }),
    ...(token   ? { Authorization: `Bearer ${token}` } : {}),
  }
  const res = await fetch(`${API_URL}${path}`, { headers, ...options })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

// ── auth ──────────────────────────────────────────────────────────────────────
export const register = (email: string, name: string, password: string) =>
  request<{ access_token: string; user: User }>('/api/auth/register', {
    method: 'POST', body: JSON.stringify({ email, name, password }),
  })

export const login = (email: string, password: string) =>
  request<{ access_token: string; user: User }>('/api/auth/login', {
    method: 'POST', body: JSON.stringify({ email, password }),
  })

export const getMe = () => request<User>('/api/auth/me')

// ── meetings ──────────────────────────────────────────────────────────────────
export const getMeetings        = () => request<MeetingListItem[]>('/api/meetings')
export const getMeeting         = (id: string) => request<MeetingDetail>(`/api/meetings/${id}`)
export const getDashboard       = () => request<unknown>('/api/dashboard')
export const cancelTranscription = (id: string) =>
  request<{ status: string }>(`/api/meetings/${id}/cancel`, { method: 'POST' })
export const searchMeetings     = (query: string) =>
  request<SearchResponse>(`/api/search?query=${encodeURIComponent(query)}`)
export const deleteQuestion     = (meetingId: string, index: number) =>
  request<{ status: string }>(`/api/meetings/${meetingId}/questions/${index}`, { method: 'DELETE' })
export const deleteMeeting      = (id: string) =>
  request<{ status: string }>(`/api/meetings/${id}`, { method: 'DELETE' })

export function uploadAudio(file: File, title?: string) {
  const form = new FormData()
  form.append('file', file)
  if (title) form.append('title', title)
  return request<UploadResponse>('/api/upload', { method: 'POST', body: form })
}

export async function* streamQuestion(meetingId: string, question: string, forceWeb = false, editIndex?: number) {
  const token = getToken()
  const res = await fetch(`${API_URL}/api/meetings/${meetingId}/questions/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ question, force_web: forceWeb, edit_index: editIndex }),
  })
  if (!res.ok) throw new Error(await res.text())
  if (!res.body) throw new Error('Streaming response body is empty')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const payload = line.slice(6)
      if (payload === '[DONE]') return
      yield JSON.parse(payload) as StreamEvent
    }
  }
}

// ── report URLs (token as query param for direct browser downloads) ───────────
function reportUrl(path: string) {
  const token = getToken()
  const tokenQuery = token ? `?token=${token}` : ''
  return `${API_URL}${path}${tokenQuery}`
}

export const getRedactionReportUrl = (id: string) => reportUrl(`/api/meetings/${id}/redaction-report`)
export const getSummaryReportUrl   = (id: string) => reportUrl(`/api/meetings/${id}/summary-report`)
export const getSummaryDocxUrl     = (id: string) => reportUrl(`/api/meetings/${id}/summary-docx`)
export const getSummaryTxtUrl      = (id: string) => reportUrl(`/api/meetings/${id}/summary-txt`)

// ── types ─────────────────────────────────────────────────────────────────────
export interface User {
  id: string
  email: string
  name: string
}

export interface MeetingListItem {
  id: string
  title: string
  created_at: string
  processing_status: string
  quality_flag: string
  wer_estimate: number
  pii_count: number
  chunk_count: number
}

export interface QuestionAnswer {
  question: string
  answer: string
  confidence: number
  routing: string
  sources?: Source[]
}

export interface Source {
  chunk_id?: string
  meeting_title: string
  snippet: string
  source_type?: string
  url?: string
}

export interface TimelineItem {
  label: string
  detail: string
}

export interface MeetingDetail extends MeetingListItem {
  summary?: string
  action_items?: string[]
  topics?: string[]
  timeline?: TimelineItem[]
  sanitized_transcript?: string
  speaker_transcript?: string
  redactions?: Redaction[]
  lecture_notes?: LectureNote[]
  questions?: QuestionAnswer[]
  chunks?: unknown[]
  snr_estimate?: number
}

export interface Redaction {
  entity_type: string
  text: string
  start?: number
  confidence?: number
  method?: string
  replacement?: string
}

export interface LectureNote {
  segment: number
  heading: string
  summary: string
}

export interface SearchResponse {
  hits: SearchHit[]
  corrected_query?: string
}

export interface SearchHit {
  chunk_id: string
  meeting_title: string
  snippet: string
  score: number
}

export interface UploadResponse {
  meeting: MeetingDetail
  message: string
}

export interface StreamEvent {
  type: 'meta' | 'token'
  text?: string
  routing?: string
  confidence?: number
  sources?: Source[]
}
