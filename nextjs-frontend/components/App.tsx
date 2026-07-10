'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  cancelTranscription, clearToken, deleteQuestion, deleteMeeting, getDashboard, getMeeting, getMeetings,
  getMe, getRedactionReportUrl, getSummaryTxtUrl, getSummaryDocxUrl,
  getSummaryReportUrl, getToken, login, register, searchMeetings, setToken,
  streamQuestion, uploadAudio,
  type MeetingDetail, type MeetingListItem, type User, type SearchHit,
} from '@/lib/api'

/* ── helpers ─────────────────────────────────────────────────────────────── */
function fmt(v: string) {
  return new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(v))
}

function detectExportFormat(q: string): string | null {
  const l = q.toLowerCase()
  if (/\bpdf\b/.test(l))                return 'pdf'
  if (/\b(docx?|docs|word)\b/.test(l))  return 'docx'
  if (/\bcsv\b/.test(l))                return 'csv'
  if (/\b(txt|text)\b/.test(l))         return 'txt'
  return null
}

function buildQAText(title: string, questions: { question: string; answer: string }[]) {
  const body = [...questions].reverse().map((q, i) => `Q${i + 1}: ${q.question}\nA: ${q.answer}\n`).join('\n---\n\n')
  return `Q&A - ${title}\n\n${body}`
}

function getInputPlaceholder(selectedId: string | null, isProcessing: boolean) {
  if (!selectedId) return 'Select a meeting to start…'
  if (isProcessing)  return 'Processing audio…'
  return 'Ask anything… (Enter to send)'
}

const SPEAKER_COLORS = ['#2dd4bf', '#a78bfa', '#fbbf24', '#f87171']

/* ── tiny components ─────────────────────────────────────────────────────── */
function IconBtn({ icon, label, active, onClick }: Readonly<{ icon: string; label: string; active?: boolean; onClick: () => void }>) {
  return (
    <button type="button" onClick={onClick} title={label}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
        active
          ? 'bg-teal-400/15 text-teal-300 border border-teal-400/30 shadow-sm shadow-teal-400/10'
          : 'text-slate-400 hover:text-slate-100 hover:bg-white/[0.07] border border-white/[0.1] hover:border-white/[0.18]'
      }`}>
      <span>{icon}</span>
      <span className="hidden sm:inline">{label}</span>
    </button>
  )
}

function SectionCard({ title, icon, onClose, children }: Readonly<{
  title: string; icon: string; onClose: () => void; children: React.ReactNode
}>) {
  return (
    <div className="mb-6 rounded-2xl border border-white/[0.06] bg-[#0c1424]/40 backdrop-blur-md overflow-hidden slide-up shadow-2xl shadow-black/35">
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.04] bg-white/[0.015]">
        <div className="flex items-center gap-2.5 text-base font-bold text-white">
          <span className="text-lg">{icon}</span>{title}
        </div>
        <button type="button" onClick={onClose}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-white/[0.06] bg-white/[0.02] text-xs font-semibold text-slate-400 hover:text-teal-400 hover:bg-teal-400/8 hover:border-teal-400/20 transition-all">
          ← Back to Chat
        </button>
      </div>
      <div className="p-6">{children}</div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════════════
   Auth Screen
   ════════════════════════════════════════════════════════════════════════════ */
function AuthScreen({ onAuth }: Readonly<{ onAuth: (u: User) => void }>) {
  const [view, setView]         = useState<'login' | 'register'>('login')
  const [email, setEmail]       = useState('')
  const [name, setName]         = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  async function submit(e: React.SyntheticEvent) {
    e.preventDefault(); setLoading(true); setError('')
    try {
      const res = view === 'login' ? await login(email, password) : await register(email, name, password)
      setToken(res.access_token); onAuth(res.user)
    } catch (err) { setError(err instanceof Error ? err.message : 'Something went wrong') }
    finally { setLoading(false) }
  }

  const inp = 'w-full bg-[#0f1a2e] border border-white/[0.08] rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/10 transition'

  return (
    <div className="h-full bg-[#060912] flex items-center justify-center p-4 overflow-y-auto">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-400 to-teal-600 inline-flex items-center justify-center text-[#05090f] font-black text-2xl shadow-xl shadow-teal-400/20 mb-4">A</div>
          <h1 className="text-xl font-bold text-white">AuRAG</h1>
          <p className="text-sm text-slate-500 mt-1">Meeting Intelligence Platform</p>
        </div>
        <div className="bg-[#0b1420] border border-white/[0.08] rounded-2xl p-7">
          <h2 className="text-sm font-semibold text-white mb-5">{view === 'login' ? 'Sign in' : 'Create account'}</h2>
          {error && <div className="mb-4 px-3 py-2.5 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400">{error}</div>}
          <form onSubmit={submit} className="flex flex-col gap-3">
            {view === 'register' && (
              <input className={inp} type="text" required placeholder="Full name" value={name} onChange={e => setName(e.target.value)} />
            )}
            <input className={inp} type="email" required placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
            <input className={inp} type="password" required placeholder={view === 'register' ? 'Password (6+ chars)' : 'Password'} value={password} onChange={e => setPassword(e.target.value)} />
            {(() => {
              let submitLabel = 'Sign in'
              if (view !== 'login') submitLabel = 'Create account'
              return (
                <button type="submit" disabled={loading}
                  className="mt-1 w-full py-2.5 rounded-xl text-sm font-semibold bg-teal-500 hover:bg-teal-400 disabled:opacity-50 text-[#05090f] transition-all flex items-center justify-center gap-2">
                  {loading
                    ? <><span className="w-3.5 h-3.5 border-2 border-[#05090f]/30 border-t-[#05090f] rounded-full spin" />Wait…</>
                    : submitLabel}
                </button>
              )
            })()}
          </form>
          <p className="text-center text-xs text-slate-600 mt-4">
            {view === 'login' ? "No account? " : 'Have an account? '}
            <button className="text-teal-400 hover:text-teal-300 font-medium" onClick={() => { setView(v => v === 'login' ? 'register' : 'login'); setError('') }}>
              {view === 'login' ? 'Create one' : 'Sign in'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════════════
   Main App
   ════════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [user, setUser]               = useState<User | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [meetings, setMeetings]       = useState<MeetingListItem[]>([])
  const [selectedId, setSelectedId]   = useState<string | null>(null)
  const [meeting, setMeeting]         = useState<MeetingDetail | null>(null)
  const [uploadTitle, setUploadTitle] = useState('')
  const [uploadFile, setUploadFile]   = useState<File | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTitle, setRecordingTitle] = useState('')
  const [question, setQuestion]       = useState('')
  const [questionLoading, setQuestionLoading] = useState(false)
  const [streamText, setStreamText]   = useState('')
  const [streamMeta, setStreamMeta]   = useState<{ routing: string; confidence: number } | null>(null)
  const [streamQ, setStreamQ]         = useState('')
  const [webMode, setWebMode]         = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchHit[]>([])
  const [searchCorrected, setSearchCorrected] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [error, setError]             = useState('')
  const [exportNote, setExportNote]   = useState('')
  const [showUpload, setShowUpload]   = useState(false)
  const [feedback, setFeedback]       = useState<Record<number, 'up' | 'down' | null>>({})
  const [copied, setCopied]           = useState<Record<number, boolean>>({})
  const [sourcesOpen, setSourcesOpen] = useState<Record<number, boolean>>({})
  const [readingIdx, setReadingIdx]   = useState<number | null>(null)
  const [exportUrls, setExportUrls]   = useState<Record<string, string>>({})
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editingText, setEditingText]   = useState('')
  const [streamingEditIndex, setStreamingEditIndex] = useState<number | null>(null)
  const [theme, setTheme]             = useState<'dark' | 'light-modern' | 'quiet-light' | 'solarized-light' | 'tokyo-night-light' | 'monokai' | 'solarized-dark' | 'synthwave84' | 'tokyo-night'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('aurag_theme') as 'dark' | 'light-modern' | 'quiet-light' | 'solarized-light' | 'tokyo-night-light' | 'monokai' | 'solarized-dark' | 'synthwave84' | 'tokyo-night') || 'dark'
    }
    return 'dark'
  })
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    localStorage.setItem('aurag_theme', theme)
  }, [theme])

  const [panels, setPanels]           = useState({ summary: false, transcript: false, notes: false, timeline: false, search: false })
  const [sidebarOpen, setSidebarOpen] = useState(true)

  useEffect(() => {
    if (typeof window !== 'undefined' && window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }, [])

  const activePanelKey = Object.keys(panels).find(key => panels[key as keyof typeof panels])

  const chatEndRef  = useRef<HTMLDivElement>(null)
  const scrollRef   = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const mediaRecRef = useRef<MediaRecorder | null>(null)
  const audioChunks = useRef<Blob[]>([])

  function resetPanels() {
    setPanels({ summary: false, transcript: false, notes: false, timeline: false, search: false })
  }

  /* ── auth ── */
  useEffect(() => {
    if (authChecked) return
    getMe()
      .then(u => { setUser(u); setAuthChecked(true) })
      .catch(() => { clearToken(); setAuthChecked(true) })
  }, [authChecked])

  /* ── data ── */
  const refresh = useCallback(async (nextId = selectedId) => {
    const [, list] = await Promise.all([getDashboard(), getMeetings()])
    setMeetings(list)
    const id = nextId ?? list[0]?.id ?? null
    setSelectedId(id)
    if (id) setMeeting(await getMeeting(id))
    else setMeeting(null)
  }, [selectedId])

  useEffect(() => {
    if (!user) return
    const t = setTimeout(() => {
      refresh().catch(e => setError(String(e)))
    }, 0)
    return () => clearTimeout(t)
  }, [refresh, user])

  useEffect(() => {
    if (!selectedId) {
      queueMicrotask(() => setMeeting(null))
      return
    }
    getMeeting(selectedId).then(setMeeting).catch(e => setError(String(e)))
    queueMicrotask(resetPanels)
  }, [selectedId])

  useEffect(() => {
    if (meeting?.processing_status !== 'processing' || !selectedId) return
    const t = setInterval(async () => {
      try {
        const up = await getMeeting(selectedId)
        setMeeting(up)
        if (up.processing_status !== 'processing') {
          const [, m] = await Promise.all([getDashboard(), getMeetings()])
          setMeetings(m)
        }
      } catch (err) { setError(String(err)) }
    }, 4000)
    return () => clearInterval(t)
  }, [meeting?.processing_status, selectedId])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      })
    }
  }, [meeting?.questions?.length, streamText])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0
    }
  }, [activePanelKey])

  /* ── actions ── */
  async function handleUpload(e: React.SyntheticEvent) {
    e.preventDefault(); if (!uploadFile) return
    setUploadLoading(true); setError('')
    try {
      const up = await uploadAudio(uploadFile, uploadTitle)
      setUploadFile(null); setUploadTitle(''); setShowUpload(false)
      await refresh(up.meeting.id)
    } catch (err) { setError(String(err)) } finally { setUploadLoading(false) }
  }

  /** Pre-build a one-time download URL silently after streaming completes */
  async function prepareExportUrl(question: string, answer: string, meetingTitle: string) {
    const fmt = detectExportFormat(question)
    if (!fmt || !answer) return
    try {
      const token = getToken()
      const res = await fetch('/api/export-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ title: meetingTitle, text: answer, format: fmt }),
      })
      if (!res.ok) return
      const { url } = await res.json()
      setExportUrls(prev => ({ ...prev, [question]: url }))
    } catch { /* silent */ }
  }

  async function triggerExport(fmt: string, overrideText?: string) {
    if (!meeting) return
    const id = meeting.id, base = meeting.title.replace(/[^a-z0-9]/gi, '_').slice(0, 30)
    
    if (overrideText) {
      setExportNote(`Creating ${fmt.toUpperCase()}…`)
      try {
        const token = getToken()
        // Step 1: POST content → get a one-time download URL (authenticated)
        const res = await fetch('/api/export-url', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {})
          },
          body: JSON.stringify({
            title: meeting.title,
            text: overrideText,
            format: fmt
          })
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
        const { url } = await res.json()
        // Step 2: Trigger download via hidden iframe - does NOT navigate the page away
        const iframe = document.createElement('iframe')
        iframe.style.display = 'none'
        iframe.src = url
        document.body.appendChild(iframe)
        setTimeout(() => document.body.removeChild(iframe), 5000)
        setExportNote(`✅ Downloading…`)
        setTimeout(() => setExportNote(''), 3000)
      } catch (err) {
        setError(`Failed to export: ${err instanceof Error ? err.message : String(err)}`)
        setExportNote('')
      }
      return
    }

    let url = ''
    let filename = ''
    if (fmt === 'pdf') {
      url = getSummaryReportUrl(id)
      filename = `summary-${base}.pdf`
    } else if (fmt === 'docx') {
      url = getSummaryDocxUrl(id)
      filename = `summary-${base}.docx`
    } else if (fmt === 'txt') {
      url = getSummaryTxtUrl(id)
      filename = `summary-${base}.txt`
    }

    if (url) {
      setExportNote(`Downloading ${fmt.toUpperCase()}…`)
      try {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const blob = await res.blob()
        const localUrl = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = localUrl
        a.download = filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        setTimeout(() => URL.revokeObjectURL(localUrl), 1000)
      } catch (err) {
        setError(`Failed to download ${fmt.toUpperCase()}: ${err instanceof Error ? err.message : String(err)}`)
      } finally {
        setTimeout(() => setExportNote(''), 4000)
      }
    }
  }

  async function handleAsk(e: React.SyntheticEvent) {
    e.preventDefault(); if (!selectedId || !question.trim()) return
    resetPanels()
    setQuestionLoading(true); setError('')
    setStreamText(''); setStreamMeta(null)
    const asked = question; setStreamQ(asked); setQuestion('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    let fullAnswer = ''
    try {
      for await (const ev of streamQuestion(selectedId, asked, webMode)) {
        if (ev.type === 'meta')       setStreamMeta({ routing: ev.routing ?? '', confidence: ev.confidence ?? 0 })
        else if (ev.type === 'token') { fullAnswer += (ev.text ?? ''); setStreamText(t => t + (ev.text ?? '')) }
      }
      setStreamText(''); setStreamMeta(null); setStreamQ('')
      await refresh(selectedId)
      // Pre-build the export download URL silently (no user gesture needed later)
      if (fullAnswer && detectExportFormat(asked) && meeting) {
        prepareExportUrl(asked, fullAnswer, meeting.title)
      }
    } catch (err) { setError(String(err)); setStreamQ('') }
    finally { setQuestionLoading(false) }
  }


  async function handleSaveEdit(chatIndex: number, dbIndex: number, originalWasWeb: boolean) {
    if (!selectedId || !editingText.trim()) return
    const newAsked = editingText
    setEditingIndex(null)
    setStreamingEditIndex(chatIndex)
    if (meeting && meeting.questions) {
      const slicedQuestions = meeting.questions.slice(dbIndex)
      setMeeting({
        ...meeting,
        questions: slicedQuestions
      })
    }
    setQuestionLoading(true); setError('')
    setStreamText(''); setStreamMeta(null)
    setStreamQ(newAsked)
    let fullAnswer = ''
    const useWebSearch = originalWasWeb || webMode
    try {
      for await (const ev of streamQuestion(selectedId, newAsked, useWebSearch, dbIndex)) {
        if (ev.type === 'meta')       setStreamMeta({ routing: ev.routing ?? '', confidence: ev.confidence ?? 0 })
        else if (ev.type === 'token') { fullAnswer += (ev.text ?? ''); setStreamText(t => t + (ev.text ?? '')) }
      }
      setStreamText(''); setStreamMeta(null); setStreamQ(''); setStreamingEditIndex(null)
      await refresh(selectedId)
      if (fullAnswer && detectExportFormat(newAsked) && meeting) {
        prepareExportUrl(newAsked, fullAnswer, meeting.title)
      }
    } catch (err) { setError(String(err)); setStreamQ(''); setStreamingEditIndex(null) }
    finally { setQuestionLoading(false) }
  }

  async function handleSearch(e: React.SyntheticEvent) {
    e.preventDefault(); if (!searchQuery.trim()) return
    setSearchLoading(true); setError('')
    try {
      const r = await searchMeetings(searchQuery)
      setSearchResults(r.hits ?? []); setSearchCorrected(r.corrected_query ?? searchQuery)
    } catch (err) { setError(String(err)) } finally { setSearchLoading(false) }
  }

  async function handleStartRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunks.current = []
      const rec = new MediaRecorder(stream)
      rec.ondataavailable = e => { if (e.data.size > 0) audioChunks.current.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' })
        const file = new File([blob], `recording-${Date.now()}.webm`, { type: 'audio/webm' })
        setUploadLoading(true)
        try { const up = await uploadAudio(file, recordingTitle || 'Browser Recording'); setRecordingTitle(''); await refresh(up.meeting.id) }
        catch (err) { setError(String(err)) } finally { setUploadLoading(false) }
      }
      mediaRecRef.current = rec; rec.start(); setIsRecording(true)
    } catch (err) { setError('Microphone denied: ' + String(err)) }
  }
  function handleStopRecording() { mediaRecRef.current?.stop(); setIsRecording(false) }

  function exportQA() {
    if (!meeting?.questions?.length) return
    const txt = buildQAText(meeting.title, meeting.questions)
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob([txt], { type: 'text/plain' })),
      download: `${meeting.title.replace(/[^a-z0-9]/gi, '_')}-qa.txt`,
    }); a.click(); URL.revokeObjectURL(a.href)
  }

  function togglePanel(key: string) {
    setPanels(p => {
      const next = { summary: false, transcript: false, notes: false, timeline: false, search: false }
      next[key as keyof typeof next] = !p[key as keyof typeof p]
      return next
    })
  }
  function closePanel(key: string)  {
    setPanels({ summary: false, transcript: false, notes: false, timeline: false, search: false })
  }

  async function handleCancel() {
    try { await cancelTranscription(selectedId ?? ''); await refresh(selectedId) }
    catch (e) { setError(String(e)) }
  }
  async function handleRetry() {
    try { await refresh(selectedId) }
    catch (e) { setError(String(e)) }
  }

  const isProcessing = meeting?.processing_status === 'processing'
  const isErr        = meeting?.processing_status === 'error'
  const chatMessages = [...(meeting?.questions ?? [])].reverse()

  if (!authChecked) return (
    <div className="h-full bg-[#060912] flex items-center justify-center">
      <span className="w-6 h-6 border-2 border-teal-400/30 border-t-teal-400 rounded-full spin" />
    </div>
  )
  if (!user) return <AuthScreen onAuth={u => setUser(u)} />

  return (
    <div className={`fixed inset-0 flex bg-[#060912] overflow-hidden text-slate-100 theme-${theme}`} style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>

      {/* Ambient glows */}
      <div className="absolute top-[-10%] left-[10%] w-[750px] h-[750px] rounded-full bg-teal-500/[0.08] blur-[140px] pointer-events-none z-0" />
      <div className="absolute bottom-[-20%] right-[0%] w-[650px] h-[650px] rounded-full bg-indigo-500/[0.07] blur-[120px] pointer-events-none z-0" />
      <div className="absolute top-[50%] left-[50%] w-[400px] h-[400px] rounded-full bg-cyan-500/[0.04] blur-[100px] pointer-events-none z-0" />

      {/* Sidebar Overlay for Mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-20 md:hidden transition-opacity duration-300"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ════════════════════ SIDEBAR ════════════════════ */}
      <aside className={`fixed inset-y-0 left-0 w-64 shrink-0 flex flex-col bg-[#080d1a] border-r border-white/[0.06] z-30 transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full md:hidden'
      }`}>

        {/* Brand + new upload */}
        <div className="px-4 pt-5 pb-4 border-b border-white/[0.05]">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-300 via-teal-400 to-teal-600 flex items-center justify-center text-[#04080f] font-black text-base shadow-lg shadow-teal-500/30">A</div>
              <div>
                <p className="font-bold text-sm text-white leading-tight">AuRAG</p>
                <p className="text-[0.6rem] text-slate-500 tracking-wide uppercase">Meeting Intelligence</p>
              </div>
            </div>
            {/* Close Sidebar Button for Mobile */}
            <button 
              type="button" 
              onClick={() => setSidebarOpen(false)}
              className="p-1 rounded-lg text-slate-500 hover:text-white hover:bg-white/[0.05] transition-all md:hidden shrink-0"
              title="Close sidebar"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="flex flex-col gap-2">
            <button type="button" onClick={() => setShowUpload(v => !v)}
              className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-white/[0.09] bg-white/[0.03] text-slate-300 hover:text-white hover:bg-white/[0.07] hover:border-white/[0.15] text-sm font-medium transition-all">
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
              New Meeting
            </button>
          </div>

          {/* Upload / Record panel */}
          {showUpload && (
            <div className="mt-3 p-3 rounded-xl bg-[#0f1c34] border border-white/[0.09] flex flex-col gap-3 slide-up shadow-lg shadow-black/30">
              <form onSubmit={handleUpload} className="flex flex-col gap-2">
                <p className="text-[0.65rem] font-bold tracking-widest uppercase text-slate-500">Upload File</p>
                <input value={uploadTitle} onChange={e => setUploadTitle(e.target.value)} placeholder="Title (optional)"
                  className="w-full bg-[#0f1a2e] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 outline-none focus:border-teal-400/40 transition" />
                <label className="flex items-center gap-2 border border-dashed border-white/[0.12] rounded-lg px-3 py-2 text-xs text-slate-500 hover:border-teal-400/30 hover:text-teal-400/70 cursor-pointer transition-all">
                  📁 <span className="truncate">{uploadFile ? uploadFile.name : 'Choose audio…'}</span>
                  <input type="file" accept="audio/*" className="hidden" onChange={e => setUploadFile(e.target.files?.[0] ?? null)} />
                </label>
                <button type="submit" disabled={!uploadFile || uploadLoading}
                  className="w-full py-1.5 rounded-lg text-xs font-semibold bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-[#05090f] transition-all">
                  {uploadLoading ? 'Processing…' : 'Upload & Process'}
                </button>
              </form>

              <div className="border-t border-white/[0.06] pt-3">
                <p className="text-[0.65rem] font-bold tracking-widest uppercase text-slate-500 mb-2">Record Live</p>
                <input value={recordingTitle} onChange={e => setRecordingTitle(e.target.value)} placeholder="Title (optional)"
                  disabled={isRecording}
                  className="w-full bg-[#0f1a2e] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 outline-none focus:border-teal-400/40 transition mb-2 disabled:opacity-50" />
                {isRecording
                  ? <button type="button" onClick={handleStopRecording}
                      className="w-full py-1.5 rounded-lg text-xs font-semibold bg-red-500/10 border border-red-500/30 text-red-400 transition-all flex items-center justify-center gap-2">
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75 pulse-ring" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-red-400" />
                      </span>{'Stop & Upload'}
                    </button>
                  : <button type="button" onClick={handleStartRecording} disabled={uploadLoading}
                      className="w-full py-1.5 rounded-lg text-xs font-semibold border border-white/[0.08] text-slate-400 hover:text-teal-400 hover:border-teal-400/30 disabled:opacity-40 transition-all">
                      🎙 Start Recording
                    </button>
                }
              </div>
            </div>
          )}
        </div>

        {/* Meeting list */}
        <div className="flex-1 overflow-y-auto px-3 pb-3">
          <p className="text-[0.6rem] font-bold tracking-widest uppercase text-slate-600 px-2 mb-2 mt-3">Meetings</p>
          {meetings.length === 0 && <p className="text-xs text-slate-600 px-2 py-2">No meetings yet.</p>}
          {meetings.map(m => {
            const sel = m.id === selectedId
            let qualityDot = 'bg-teal-400'
            if (m.processing_status === 'processing') qualityDot = 'bg-amber-400 animate-pulse'
            else if (m.processing_status === 'error') qualityDot = 'bg-red-500'
            else if (m.quality_flag === 'red')        qualityDot = 'bg-red-500'
            else if (m.quality_flag === 'amber')      qualityDot = 'bg-amber-400'
            return (
              <div key={m.id}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-xl transition-all duration-150 group mb-0.5 border ${
                  sel
                    ? 'bg-teal-500/[0.1] border-teal-500/[0.18] text-white shadow-sm shadow-teal-500/5'
                    : 'text-slate-400 hover:bg-white/[0.05] hover:text-slate-200 border-transparent'
                }`}>
                <button type="button" onClick={() => {
                    setSelectedId(m.id);
                    if (typeof window !== 'undefined' && window.innerWidth < 768) {
                      setSidebarOpen(false);
                    }
                  }}
                  className="flex-1 text-left min-w-0 pr-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[0.82rem] font-medium truncate">{m.title}</span>
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${qualityDot}`} title={m.processing_status === 'processing' ? 'Processing…' : m.quality_flag ?? 'green'} />
                  </div>
                  <p className="text-[0.65rem] text-slate-600 mt-0.5 group-hover:text-slate-500 transition">{fmt(m.created_at)}</p>
                </button>
                
                {/* Delete button (visible on hover) */}
                <button type="button" 
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (confirm(`Are you sure you want to delete '${m.title}'?`)) {
                      try {
                        await deleteMeeting(m.id);
                        if (selectedId === m.id) {
                          setSelectedId(null);
                          setMeeting(null);
                        }
                        await refresh(null);
                      } catch (err) {
                        setError('Failed to delete meeting: ' + String(err));
                      }
                    }
                  }}
                  title="Delete meeting"
                  className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-slate-600 hover:text-red-400 hover:bg-red-400/10 transition-all shrink-0 ml-1"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            )
          })}
        </div>

        {/* User */}
        <div className="border-t border-white/[0.06] px-4 py-3 flex items-center gap-2.5 bg-black/[0.15]">
          <button
            onClick={() => setShowSettings(true)}
            title="Open Settings"
            className="flex flex-1 items-center gap-2.5 min-w-0 text-left hover:opacity-85 transition-opacity"
          >
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-[#05090f] font-bold text-xs shrink-0 shadow-md shadow-teal-500/20">
              {user.name?.[0]?.toUpperCase() ?? '?'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-slate-300 truncate">{user.name}</p>
              <p className="text-[0.6rem] text-slate-500 truncate">Settings</p>
            </div>
          </button>
          <button onClick={() => { clearToken(); setUser(null); setMeetings([]); setMeeting(null); setSelectedId(null) }}
            title="Sign out" className="p-1.5 rounded-lg text-slate-600 hover:text-red-400 hover:bg-red-400/10 transition-all">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h6a2 2 0 012 2v1" />
            </svg>
          </button>
        </div>
      </aside>

      {/* ════════════════════ MAIN ════════════════════ */}
      <div className="flex-1 flex flex-col overflow-hidden z-10">

        {/* ── Top bar ── */}
        <header className="flex-none border-b border-white/[0.08] bg-[#0a1422]/95 backdrop-blur-xl z-20 shadow-lg shadow-black/20">
          {/* Title row */}
          <div className="flex items-center justify-between px-5 h-14">
            <div className="flex items-center gap-3 min-w-0">
              {/* Toggle Sidebar Button */}
              <button 
                type="button" 
                onClick={() => setSidebarOpen(v => !v)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.05] transition-all shrink-0"
                title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              
              {!meeting && (
                <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-[#04080f] font-black text-xs shrink-0 shadow-md shadow-teal-500/25">A</div>
              )}
              <span className={`font-semibold truncate ${meeting ? 'text-sm text-white' : 'text-sm text-slate-400'}`}>
                {meeting ? meeting.title : 'Select a meeting to get started'}
              </span>
            </div>
            {isProcessing && (
              <span className="flex items-center gap-1.5 text-xs text-amber-400 shrink-0 ml-3 bg-amber-400/8 border border-amber-400/20 px-2.5 py-1 rounded-full">
                <span className="w-2.5 h-2.5 border-2 border-amber-400/30 border-t-amber-400 rounded-full spin" />
                Processing…
              </span>
            )}

          </div>

          {/* Toolbar row - only when a meeting is loaded */}
          {meeting && (
            <div className="flex flex-wrap items-center gap-1.5 px-4 pb-3 pt-0">
              {meeting.summary && (
                <IconBtn icon="📋" label="Summary"   active={panels.summary}    onClick={() => togglePanel('summary')} />
              )}
              <IconBtn icon="📝" label="Transcript"  active={panels.transcript} onClick={() => togglePanel('transcript')} />
              {(meeting.lecture_notes ?? []).length > 0 && (
                <IconBtn icon="📚" label="Notes"     active={panels.notes}      onClick={() => togglePanel('notes')} />
              )}
              {(meeting.timeline ?? []).length > 0 && (
                <IconBtn icon="⏱" label="Timeline"  active={panels.timeline}   onClick={() => togglePanel('timeline')} />
              )}
              <IconBtn icon="🔍" label="Search"      active={panels.search}     onClick={() => togglePanel('search')} />
            </div>
          )}
        </header>

        {/* ── Single scroll area ── */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto relative bg-[#070c16]">

          {/* Ambient glows behind panel content */}
          <div className="absolute top-[-5%] left-[5%] w-[600px] h-[600px] rounded-full bg-teal-500/[0.06] blur-[130px] pointer-events-none z-0" />
          <div className="absolute bottom-[-5%] right-[5%] w-[500px] h-[500px] rounded-full bg-indigo-500/[0.06] blur-[110px] pointer-events-none z-0" />

          <div className={`${activePanelKey ? 'max-w-4xl' : 'max-w-2xl'} mx-auto px-4 py-6 relative z-10`}>

            {/* ─ error banner ─ */}
            {error && (
              <div className="flex items-center justify-between gap-3 mb-4 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                {error}
                <button onClick={() => setError('')} className="text-red-400/50 hover:text-red-400">✕</button>
              </div>
            )}

            {/* ─ error banner (failed transcription) ─ */}
            {isErr && (
              <div className="flex items-center justify-between mb-4 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                <span>❌ Transcription failed. Try uploading the file again.</span>
                <button className="text-xs px-2.5 py-1 rounded-lg bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 transition"
                  onClick={handleRetry}>
                  Retry
                </button>
              </div>
            )}

            {/* ─ processing banner ─ */}
            {isProcessing && (
              <div className="flex items-center justify-between mb-4 px-4 py-3 rounded-xl bg-amber-400/8 border border-amber-400/20 text-sm text-amber-400">
                <span>⏳ Transcribing - updates automatically.</span>
                <button className="text-xs px-2.5 py-1 rounded-lg bg-amber-400/10 hover:bg-amber-400/20 border border-amber-400/20 transition"
                  onClick={handleCancel}>
                  Cancel
                </button>
              </div>
            )}

            {/* ─ inline panels ─ */}
            {panels.summary && meeting?.summary && (
              <SectionCard title="Summary" icon="📋" onClose={() => closePanel('summary')}>
                <p className="text-sm text-slate-300 leading-relaxed mb-4">{meeting.summary}</p>
                {(meeting.action_items ?? []).length > 0 && (
                  <div className="mb-4">
                    <p className="text-[0.65rem] font-bold tracking-widest uppercase text-slate-500 mb-2">Action Items</p>
                    <ul className="flex flex-col gap-2">
                      {meeting.action_items!.map((item) => (
                        <li key={item} className="flex items-start gap-2 text-sm text-slate-400">
                          <span className="mt-2 w-1.5 h-1.5 rounded-full bg-teal-400/60 shrink-0" />{item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {(meeting.topics ?? []).length > 0 && (
                  <div className="mb-4">
                    <p className="text-[0.65rem] font-bold tracking-widest uppercase text-slate-500 mb-2">Topics</p>
                    <div className="flex flex-wrap gap-1.5">
                      {meeting.topics!.map((t) => (
                        <span key={t} className="px-2.5 py-0.5 rounded-full text-xs bg-teal-400/8 text-teal-400/80 border border-teal-400/15">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="flex gap-2 pt-3 border-t border-white/[0.06]">
                  <a href={getSummaryReportUrl(meeting.id)} download className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-teal-400 hover:bg-teal-400/8 border border-white/[0.06] hover:border-teal-400/20 transition-all">↓ PDF</a>
                  <a href={getSummaryDocxUrl(meeting.id)} download className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-teal-400 hover:bg-teal-400/8 border border-white/[0.06] hover:border-teal-400/20 transition-all">↓ DOCX</a>
                  <a href={getSummaryTxtUrl(meeting.id)} download className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-teal-400 hover:bg-teal-400/8 border border-white/[0.06] hover:border-teal-400/20 transition-all">↓ TXT</a>
                </div>
              </SectionCard>
            )}

            {panels.transcript && (
              <SectionCard title="Transcript" icon="📝" onClose={() => closePanel('transcript')}>
                {(meeting?.redactions ?? []).length > 0 && (
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-xs text-amber-400">{meeting!.pii_count} PII redactions</span>
                    <a href={getRedactionReportUrl(meeting!.id)} download className="text-xs text-slate-400 hover:text-teal-400 transition">↓ PII Report</a>
                  </div>
                )}
                <div className="bg-[#080f1c]/30 rounded-xl p-4 border border-white/[0.03]">
                  {meeting?.speaker_transcript
                    ? meeting.speaker_transcript.split('\n\n').map((line) => {
                        const m = line.match(/^\[([^\]]+)\]\s*(.*)$/s)
                        if (!m) return <p key={line} className="text-sm text-slate-500 mb-2 leading-relaxed">{line}</p>
                        const [, speaker, text] = m
                        const idx   = ['A','B','C','D'].indexOf(speaker.slice(-1))
                        const color = SPEAKER_COLORS[idx] ?? SPEAKER_COLORS[0]
                        return (
                          <div key={line} className="flex items-start gap-3 mb-3 last:mb-0">
                            <span style={{ background: color + '15', color, border: `1px solid ${color}30` }}
                              className="px-2.5 py-0.5 rounded-full text-[0.62rem] font-bold shrink-0 mt-0.5">{speaker}</span>
                            <p className="text-sm text-slate-300 leading-relaxed">{text}</p>
                          </div>
                        )
                      })
                    : <p className="text-sm text-slate-400 leading-relaxed">{meeting?.sanitized_transcript ?? 'No transcript available.'}</p>
                  }
                </div>
              </SectionCard>
            )}

            {panels.notes && (meeting?.lecture_notes ?? []).length > 0 && (
              <SectionCard title="Lecture Notes" icon="📚" onClose={() => closePanel('notes')}>
                <div className="flex flex-col gap-3">
                  {meeting!.lecture_notes!.map(note => (
                    <div key={note.segment} className="flex gap-3 p-3 rounded-xl bg-[#080f1c]/35 border border-white/[0.03] hover:border-teal-500/10 transition-all duration-300">
                      <div className="w-6 h-6 rounded-lg bg-teal-400/10 border border-teal-400/20 flex items-center justify-center text-[0.65rem] font-bold text-teal-400 shrink-0 mt-0.5">{note.segment}</div>
                      <div>
                        <p className="text-sm font-bold text-white">{note.heading}</p>
                        <p className="text-xs text-slate-400 mt-1 leading-relaxed">{note.summary}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </SectionCard>
            )}

            {panels.timeline && (meeting?.timeline ?? []).length > 0 && (
              <SectionCard title="Timeline" icon="⏱" onClose={() => closePanel('timeline')}>
                <div className="relative pl-5 py-1">
                  <div className="absolute left-2 top-0 bottom-0 w-px bg-white/[0.08]" />
                  {meeting!.timeline!.map((item) => (
                    <div key={item.label} className="relative flex gap-3 mb-5 last:mb-0">
                      <div className="absolute -left-[1.1rem] top-1.5 w-2 h-2 rounded-full bg-teal-400/70 ring-4 ring-[#070b13]" />
                      <div>
                        <p className="text-xs font-bold text-teal-400 mb-0.5">{item.label}</p>
                        <p className="text-sm text-slate-300 leading-relaxed">{item.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </SectionCard>
            )}

            {panels.search && (
              <SectionCard title="Search Across Meetings" icon="🔍" onClose={() => closePanel('search')}>
                <form onSubmit={handleSearch} className="flex gap-2 mb-4">
                  <input value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                    placeholder="Search all meeting transcripts…"
                    className="flex-1 bg-[#080f1c] border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-slate-300 placeholder-slate-600 outline-none focus:border-teal-400/40 focus:ring-1 focus:ring-teal-400/20 transition" />
                  <button type="submit" disabled={searchLoading}
                    className="px-4 py-2 rounded-xl text-sm font-semibold bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-[#05090f] transition-all">
                    {searchLoading ? '…' : 'Search'}
                  </button>
                </form>
                {searchResults.length > 0 && (
                  <div className="flex flex-col gap-2">
                    {searchCorrected !== searchQuery && (
                      <p className="text-xs text-slate-500">Results for: <strong className="text-slate-400">{searchCorrected}</strong></p>
                    )}
                    {searchResults.map(hit => (
                      <div key={hit.chunk_id} className="p-3 rounded-xl bg-[#080f1c]/50 border border-white/[0.05] hover:border-teal-500/15 transition-all">
                        <p className="text-xs font-semibold text-teal-400 mb-1">{hit.meeting_title} <span className="text-slate-600 font-normal ml-2">{hit.score.toFixed(2)} score</span></p>
                        <p className="text-xs text-slate-450 leading-relaxed">{hit.snippet}</p>
                      </div>
                    ))}
                  </div>
                )}
              </SectionCard>
            )}

            {!activePanelKey && (
              <>
                {/* ─ audio quality badge ─ */}
                {meeting?.processing_status === 'ready' && (() => {
                  let dotColor = 'bg-teal-400'
                  let qualityLabel = 'Good'
                  if (meeting.quality_flag === 'red')   { dotColor = 'bg-red-500';   qualityLabel = 'Poor' }
                  else if (meeting.quality_flag === 'amber') { dotColor = 'bg-amber-400'; qualityLabel = 'Fair' }
                  return (
                    <div className="flex items-center gap-3 mb-6 px-4 py-2.5 rounded-xl bg-white/[0.02] border border-white/[0.05] text-xs text-slate-500 shadow-sm">
                      <span className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />
                      <span>Audio quality: <strong className="text-slate-400">{qualityLabel}</strong></span>
                      {meeting.wer_estimate != null && <span>· WER {(meeting.wer_estimate * 100).toFixed(0)}%</span>}
                      {meeting.snr_estimate != null && <span>· SNR {meeting.snr_estimate.toFixed(1)} dB</span>}
                      {meeting.pii_count > 0 && <span>· {meeting.pii_count} PII redacted</span>}
                    </div>
                  )
                })()}

                {/* ─ empty states ─ */}
                {!meeting && (
                  <div className="flex flex-col items-center justify-center py-32 text-center">
                    <div className="w-18 h-18 rounded-2xl bg-teal-400/[0.08] border border-teal-400/[0.15] flex items-center justify-center text-4xl mb-6 shadow-2xl shadow-teal-500/10 w-[72px] h-[72px]">🎙</div>
                    <h2 className="text-lg font-bold text-slate-200 mb-2">No meeting selected</h2>
                    <p className="text-sm text-slate-500 max-w-xs leading-relaxed">Click <strong className="text-slate-400">New Meeting</strong> in the sidebar to upload audio, or pick an existing meeting from the list.</p>
                  </div>
                )}

                {meeting && chatMessages.length === 0 && !streamText && (
                  <div className="flex flex-col items-center justify-center py-24 text-center">
                    <div className="relative mb-6">
                      <div className="absolute inset-0 rounded-full bg-teal-400/20 blur-xl scale-150" />
                      <div className="relative w-14 h-14 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-[#04080f] font-black text-xl shadow-2xl shadow-teal-500/30">A</div>
                    </div>
                    <h2 className="text-lg font-bold text-white mb-1">What would you like to know?</h2>
                    <p className="text-sm text-slate-500 max-w-sm mb-7 leading-relaxed">Ask anything about <strong className="text-slate-300">{meeting.title}</strong>. You can also ask me to export a PDF, DOCX or TXT.</p>
                    <div className="flex flex-wrap gap-2 justify-center max-w-md">
                      {['Summarise the key decisions', 'List all action items', 'Who were the main speakers?', 'Give me a PDF summary'].map(s => (
                        <button key={s} type="button" onClick={() => setQuestion(s)}
                          className="px-3.5 py-2 rounded-xl border border-white/[0.09] bg-white/[0.03] text-xs text-slate-400 hover:text-white hover:bg-white/[0.07] hover:border-white/[0.16] transition-all">
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* ─ chat messages ─ */}
                {chatMessages.map((qa, index) => {
                  const isEditingStreaming = streamingEditIndex === index
                  const displayQuestion = isEditingStreaming ? streamQ : qa.question
                  const displayAnswer = isEditingStreaming ? streamText : qa.answer
                  const displayRouting = isEditingStreaming ? (streamMeta?.routing ?? '') : qa.routing
                  const displayConfidence = isEditingStreaming ? (streamMeta?.confidence ?? 0) : qa.confidence
                  const displaySources = isEditingStreaming ? [] : (qa.sources ?? [])

                  const exportFmt = detectExportFormat(displayQuestion)
                  return (
                  <div key={index} className="mb-8 slide-up">
                    {/* User bubble */}
                    <div className="flex justify-end items-center gap-2 mb-4 group">
                      {editingIndex !== index && !questionLoading && (
                        <button
                          type="button"
                          onClick={() => {
                            setEditingIndex(index)
                            setEditingText(qa.question)
                          }}
                          className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-teal-400 p-1.5 rounded-lg hover:bg-white/[0.04] transition-all duration-200 shrink-0"
                          title="Edit question"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                            <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.83 20.089a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
                          </svg>
                        </button>
                      )}

                      {editingIndex === index ? (
                        <div className="max-w-[75%] flex flex-col gap-2 bg-[#121c2c] rounded-2xl border border-white/[0.12] p-3 w-full">
                          <textarea
                            value={editingText}
                            onChange={(e) => setEditingText(e.target.value)}
                            className="bg-[#0f1725] text-slate-100 text-sm rounded-lg p-2 border border-white/[0.08] focus:border-teal-400/50 outline-none w-full resize-none"
                            rows={3}
                          />
                          <div className="flex justify-end gap-2 text-xs">
                            <button
                              type="button"
                              onClick={() => setEditingIndex(null)}
                              className="px-2.5 py-1.5 rounded-lg border border-white/[0.08] text-slate-400 hover:text-white transition-colors"
                            >
                              Cancel
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                const dbIndex = (meeting?.questions?.length ?? 0) - 1 - index
                                const wasWeb = qa.routing === 'agentic_web'
                                handleSaveEdit(index, dbIndex, wasWeb)
                              }}
                              className="px-2.5 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-[#05090f] font-semibold transition-colors"
                            >
                              Save
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="max-w-[75%] bg-gradient-to-br from-[#1a2d4a] to-[#132238] rounded-3xl rounded-tr-lg px-5 py-3.5 border border-white/[0.07] shadow-lg shadow-black/20">
                          <p className="text-[0.9rem] text-slate-100 leading-relaxed">{displayQuestion}</p>
                        </div>
                      )}
                    </div>

                    {/* AI response */}
                    <div className="flex gap-3.5">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-[#04080f] font-black text-sm shrink-0 mt-0.5 shadow-md shadow-teal-500/25">A</div>
                      <div className="flex-1 min-w-0 pt-1">
                        <div className="flex items-center gap-2 mb-2.5">
                          {displayRouting === 'agentic_web'
                            ? <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[0.65rem] font-semibold bg-violet-400/10 text-violet-400 ring-1 ring-violet-400/20">🌐 Web search</span>
                            : <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[0.65rem] font-semibold bg-teal-400/10 text-teal-400 ring-1 ring-teal-400/20">📄 Transcript</span>
                          }
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[0.65rem] font-semibold bg-white/5 text-slate-400 ring-1 ring-white/10">
                            {Math.round((displayConfidence ?? 0) * 100)}% confidence
                          </span>
                        </div>
                        {displayRouting === 'agentic_web' && (
                          <div className="flex items-start gap-2 mb-3 px-3 py-2 rounded-lg bg-amber-400/8 border border-amber-400/20 text-xs text-amber-400/90">
                            <span className="shrink-0 mt-0.5">⚠️</span>
                            <span>This answer came from a <strong>web search</strong>, not your meeting transcript - the transcript didn&apos;t have enough information on this topic. Results may be unrelated to your meeting.</span>
                          </div>
                        )}
                        <p className="text-[0.925rem] text-slate-200 leading-[1.75] whitespace-pre-wrap">{displayAnswer}</p>
                        {(displaySources ?? []).length > 0 && (
                          sourcesOpen[index] ? (
                            <div className="mt-4 flex flex-col gap-1.5">
                              <button type="button" onClick={() => setSourcesOpen(p => ({ ...p, [index]: false }))}
                                className="self-start text-xs font-medium text-slate-500 hover:text-slate-300 transition-colors mb-0.5">
                                Hide sources ({displaySources.length})
                              </button>
                              {displaySources.map((s) => (
                                <div key={s.chunk_id ?? s.url ?? s.snippet} className={`px-3 py-2 rounded-lg text-xs border ${s.source_type === 'web' ? 'bg-violet-400/5 border-violet-400/15' : 'bg-white/[0.02] border-white/[0.06]'}`}>
                                  <p className="font-medium text-slate-400 mb-0.5">
                                    {s.source_type === 'web' && '🌐 '}
                                    {s.url ? <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-violet-400 hover:underline">{s.meeting_title}</a> : s.meeting_title}
                                    {s.source_type === 'transcript' && ` (Chunk ${s.chunk_id})`}
                                  </p>
                                  <p className="text-slate-550 leading-relaxed">{s.snippet}</p>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <button type="button" onClick={() => setSourcesOpen(p => ({ ...p, [index]: true }))}
                              className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-300 transition-colors">
                              📎 Show sources ({displaySources.length})
                            </button>
                          )
                        )}
                        {(() => {
                          const fmt = exportFmt
                          if (!fmt) return null
                          const extLabel = fmt === 'pdf' ? 'PDF'
                            : fmt === 'docx' ? 'Word document'
                            : 'Text file'
                          const fileLabel = fmt === 'txt' ? 'the text summary'
                            : 'the report'
                          const tok = getToken() ?? ''
                          const formUrl = `/api/export-form?token=${encodeURIComponent(tok)}`
                          return (
                            <div className="mt-3">
                              <p className="text-[0.9rem] text-slate-200 leading-relaxed mb-2">I&apos;ve created the {extLabel} for you:</p>
                              <form method="POST" action={formUrl} style={{ display: 'inline' }}>
                                <input type="hidden" name="title" value={meeting?.title || ''} />
                                <input type="hidden" name="text" value={displayAnswer} />
                                <input type="hidden" name="format" value={fmt} />
                                <button type="submit" className="inline-flex items-center gap-1.5 text-[0.9rem] font-medium text-teal-400 hover:text-teal-300 underline underline-offset-2 decoration-teal-400/50 hover:decoration-teal-300 transition-colors">
                                  <span>📄</span>
                                  <span>Download {fileLabel} ({extLabel})</span>
                                  <span className="text-xs">↗</span>
                                </button>
                              </form>
                            </div>
                          )
                        })()}
                        {/* ── ChatGPT-style action bar ── */}
                        <div className="flex items-center gap-0.5 mt-3 -ml-1">
                          {/* Copy */}
                          <button
                            type="button"
                            title="Copy"
                            onClick={() => {
                              navigator.clipboard.writeText(displayAnswer).then(() => {
                                setCopied(p => ({ ...p, [index]: true }))
                                setTimeout(() => setCopied(p => ({ ...p, [index]: false })), 2000)
                              })
                            }}
                            className="flex items-center justify-center w-8 h-8 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-white/[0.07] transition-all"
                          >
                            {copied[index]
                              ? <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                              : <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
                            }
                          </button>

                          {/* Read aloud / Stop */}
                          <button
                            type="button"
                            title={readingIdx === index ? 'Stop reading' : 'Read aloud'}
                            onClick={() => {
                              if (readingIdx === index) {
                                speechSynthesis.cancel()
                                setReadingIdx(null)
                              } else {
                                speechSynthesis.cancel()
                                const utter = new SpeechSynthesisUtterance(displayAnswer)
                                utter.onend = () => setReadingIdx(null)
                                utter.onerror = () => setReadingIdx(null)
                                speechSynthesis.speak(utter)
                                setReadingIdx(index)
                              }
                            }}
                            className={`flex items-center justify-center w-8 h-8 rounded-lg transition-all ${
                              readingIdx === index
                                ? 'text-teal-400 bg-teal-400/10'
                                : 'text-slate-500 hover:text-slate-200 hover:bg-white/[0.07]'
                            }`}
                          >
                            {readingIdx === index
                              ? <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                              : <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
                            }
                          </button>

                          {/* Divider */}
                          <div className="w-px h-4 bg-white/[0.08] mx-1" />

                          {/* Thumbs up */}
                          <button
                            type="button"
                            title="Good response"
                            onClick={() => setFeedback(p => ({ ...p, [index]: p[index] === 'up' ? null : 'up' }))}
                            className={`flex items-center justify-center w-8 h-8 rounded-lg transition-all ${
                              feedback[index] === 'up'
                                ? 'text-teal-400 bg-teal-400/10'
                                : 'text-slate-500 hover:text-slate-200 hover:bg-white/[0.07]'
                            }`}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill={feedback[index] === 'up' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/></svg>
                          </button>

                          {/* Thumbs down */}
                          <button
                            type="button"
                            title="Bad response"
                            onClick={() => setFeedback(p => ({ ...p, [index]: p[index] === 'down' ? null : 'down' }))}
                            className={`flex items-center justify-center w-8 h-8 rounded-lg transition-all ${
                              feedback[index] === 'down'
                                ? 'text-red-400 bg-red-400/10'
                                : 'text-slate-500 hover:text-slate-200 hover:bg-white/[0.07]'
                            }`}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill={feedback[index] === 'down' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/></svg>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )})}

                {/* ─ streaming ─ */}
                {(streamText || (questionLoading && streamQ)) && streamingEditIndex === null && (
                  <div className="mb-8 slide-up">
                    <div className="flex justify-end mb-4">
                      <div className="max-w-[75%] bg-gradient-to-br from-[#1a2d4a] to-[#132238] rounded-3xl rounded-tr-lg px-5 py-3.5 border border-white/[0.07] shadow-lg shadow-black/20">
                        <p className="text-[0.9rem] text-slate-100 leading-relaxed">{streamQ}</p>
                      </div>
                    </div>
                    <div className="flex gap-3.5">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-400 to-teal-600 flex items-center justify-center text-[#04080f] font-black text-sm shrink-0 mt-0.5 shadow-md shadow-teal-500/25">A</div>
                      <div className="flex-1 min-w-0 pt-1">
                        {streamMeta && (
                          <div className="mb-2.5">
                            <div className="flex items-center gap-2 mb-2">
                              {streamMeta.routing === 'agentic_web'
                                ? <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[0.65rem] font-semibold bg-violet-400/10 text-violet-400 ring-1 ring-violet-400/20">🌐 Web search</span>
                                : <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[0.65rem] font-semibold bg-teal-400/10 text-teal-400 ring-1 ring-teal-400/20">📄 Transcript</span>
                              }
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[0.65rem] font-semibold bg-white/5 text-slate-400 ring-1 ring-white/10">{Math.round(streamMeta.confidence * 100)}% confidence</span>
                            </div>
                            {streamMeta.routing === 'agentic_web' && (
                              <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-amber-400/8 border border-amber-400/20 text-xs text-amber-400/90">
                                <span className="shrink-0 mt-0.5">⚠️</span>
                                <span>Answering from <strong>web search</strong> - transcript confidence was too low. Results may be unrelated to your meeting.</span>
                              </div>
                            )}
                          </div>
                        )}
                        {streamText
                          ? <p className="text-[0.925rem] text-slate-200 leading-[1.75]">{streamText}<span className="text-teal-400 blink">▌</span></p>
                          : <div className="flex items-center gap-2 text-slate-500 text-sm">
                              <span className="w-4 h-4 border-2 border-teal-400/30 border-t-teal-400 rounded-full spin" />{'Thinking…'}
                            </div>
                        }
                      </div>
                    </div>
                  </div>
                )}

            {/* ─ export note / QA export ─ */}
            {exportNote && (
              <p className="text-xs text-teal-400 mb-4">⬇ {exportNote}</p>
            )}
            {chatMessages.length > 1 && (
              <div className="flex justify-center mb-6">
                <button type="button" onClick={exportQA}
                  className="text-xs text-slate-600 hover:text-slate-400 transition-all">↓ Export Q&amp;A history</button>
              </div>
            )}
          </>
        )}

            <div ref={chatEndRef} className="h-4" />
          </div>
        </div>

        {/* ── Input bar ── */}
        {!activePanelKey && (
          <div className="flex-none bg-[#080d1a]/95 border-t border-white/[0.07] backdrop-blur-xl px-4 pb-5 pt-3 z-10">
              <div className="max-w-2xl mx-auto">

                {/* Web mode banner — shown when web search is enabled */}
                {webMode && (
                  <div className="flex items-center gap-2 mb-2 px-3 py-2 rounded-xl bg-violet-500/10 border border-violet-400/25 text-xs text-violet-300">
                    <span>🌐</span>
                    <span className="flex-1"><strong>Web search is ON</strong> - answers may come from the internet, not your meeting transcript.</span>
                    <button type="button" onClick={() => setWebMode(false)}
                      className="shrink-0 text-violet-400/60 hover:text-violet-300 transition-colors text-base leading-none"
                      title="Turn off web search">✕</button>
                  </div>
                )}

                <form onSubmit={handleAsk}>
                  <div className="flex items-end gap-2 bg-[#0f1d35] border border-white/[0.1] rounded-2xl px-4 py-3 focus-within:border-teal-400/50 focus-within:shadow-lg focus-within:shadow-teal-400/[0.06] transition-all shadow-lg shadow-black/20">
                    <textarea ref={textareaRef} rows={1} value={question}
                      onChange={e => { setQuestion(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px' }}
                      onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(e) } }}
                      disabled={!selectedId || isProcessing}
                      placeholder={getInputPlaceholder(selectedId, isProcessing)}
                      className="flex-1 bg-transparent text-[0.9rem] text-slate-200 placeholder-slate-600 outline-none resize-none max-h-40 overflow-y-auto leading-relaxed py-0.5" />
                    <button type="submit" disabled={!selectedId || !question.trim() || questionLoading || isProcessing}
                      className="shrink-0 w-8 h-8 rounded-xl bg-teal-500 hover:bg-teal-400 disabled:opacity-25 disabled:cursor-not-allowed text-[#04080f] transition-all flex items-center justify-center shadow-md shadow-teal-500/30 hover:shadow-teal-400/40">
                      {questionLoading
                        ? <span className="w-3.5 h-3.5 border-2 border-[#05090f]/30 border-t-[#05090f] rounded-full spin" />
                        : <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
                          </svg>
                      }
                    </button>
                  </div>
                  <div className="flex items-center justify-between mt-1.5 px-0.5">
                    <button type="button" onClick={() => setWebMode(v => !v)}
                      className={`flex items-center gap-1.5 text-[0.68rem] font-medium rounded-lg px-2 py-1 transition-all ${webMode ? 'bg-violet-400/15 text-violet-400 ring-1 ring-violet-400/30' : 'text-slate-600 hover:text-slate-400 hover:bg-white/5'}`}>
                      <span>🌐</span>
                      <span>{webMode ? 'Web search: ON' : 'Web search: OFF'}</span>
                    </button>
                    <p className="text-[0.62rem] text-slate-700">Shift+Enter for new line</p>
                  </div>
                </form>
              </div>
            </div>
        )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-md flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-[#0b1329] border border-white/[0.08] rounded-3xl p-6 w-full max-w-md shadow-2xl relative slide-up">
            <button
              onClick={() => setShowSettings(false)}
              className="absolute top-4 right-4 text-slate-500 hover:text-white transition-colors"
              title="Close Settings"
            >
              ✕
            </button>
            <h2 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
              <span>⚙️</span> Settings
            </h2>

            <div className="flex flex-col gap-5">
              {/* Theme Settings */}
              <div>
                <p className="text-xs font-semibold tracking-wider text-slate-500 uppercase mb-3">Appearance</p>
                <div className="flex flex-col gap-4">
                  {/* Dark Theme Dropdown */}
                  <div>
                    <label className="block text-[0.7rem] font-bold tracking-wider uppercase text-slate-500 mb-1.5">
                      Dark Themes
                    </label>
                    <select
                      value={['dark', 'tokyo-night', 'monokai', 'solarized-dark', 'synthwave84'].includes(theme) ? theme : ''}
                      onChange={(e) => {
                        if (e.target.value) setTheme(e.target.value as any)
                      }}
                      className="w-full bg-[#0f1d35] border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-slate-200 outline-none focus:border-teal-400/40 focus:ring-1 focus:ring-teal-400/20 transition cursor-pointer"
                    >
                      <option value="" disabled>Select Dark Theme...</option>
                      <option value="dark">🌙 Dark+ (Default)</option>
                      <option value="tokyo-night">🌌 Tokyo Night</option>
                      <option value="monokai">🎨 Monokai</option>
                      <option value="solarized-dark">🌲 Solarized Dark</option>
                      <option value="synthwave84">⚡ SynthWave '84</option>
                    </select>
                  </div>

                  {/* Light Theme Dropdown */}
                  <div>
                    <label className="block text-[0.7rem] font-bold tracking-wider uppercase text-slate-500 mb-1.5">
                      Light Themes
                    </label>
                    <select
                      value={['light-modern', 'quiet-light', 'solarized-light', 'tokyo-night-light'].includes(theme) ? theme : ''}
                      onChange={(e) => {
                        if (e.target.value) setTheme(e.target.value as any)
                      }}
                      className="w-full bg-[#0f1d35] border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-slate-200 outline-none focus:border-teal-400/40 focus:ring-1 focus:ring-teal-400/20 transition cursor-pointer"
                    >
                      <option value="" disabled>Select Light Theme...</option>
                      <option value="light-modern">☀️ Light Modern</option>
                      <option value="quiet-light">🌸 Quiet Light</option>
                      <option value="solarized-light">🌾 Solarized Light</option>
                      <option value="tokyo-night-light">🎐 Tokyo Night Light</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* User details read-only */}
              <div className="border-t border-white/[0.06] pt-4">
                <p className="text-xs font-semibold tracking-wider text-slate-500 uppercase mb-2">User Profile</p>
                <div className="bg-[#0f1a2e] rounded-xl p-3 border border-white/[0.04]">
                  <p className="text-xs text-slate-400">Signed in as</p>
                  <p className="text-sm font-semibold text-white mt-0.5">{user.name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{user.email}</p>
                </div>
              </div>

              {/* Save/Close button */}
              <button
                onClick={() => setShowSettings(false)}
                className="w-full mt-2 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-[#05090f] font-semibold text-sm transition-colors shadow-md shadow-teal-500/20"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      </div>
    </div>
  )
}
