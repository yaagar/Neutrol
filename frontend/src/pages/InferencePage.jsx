// InferencePage.jsx — live EEG inference display
//
// The user picks a trained model and a pipeline run, then clicks Start.
// The backend streams one prediction per window at the selected speed.
// The page shows the current prediction, per-class confidence bars,
// and a scrolling ribbon of the last 40 predictions.
//
// WebSocket protocol:
//   Client → { type:'start', job_id, run_id, delay_ms }
//   Server → { type:'ready', class_names, total }
//   Server → { type:'prediction', window_index, predicted_class, confidence, scores, total }
//   Server → { type:'done', total }
//   Client → { type:'stop' }  (sent any time to halt)

import { useState, useEffect, useRef, useCallback } from 'react'
import { useSession } from '../hooks/useSession'
import { API_BASE_URL, WS_BASE_URL } from '../config'

const API = API_BASE_URL
const WS  = WS_BASE_URL

// One color per class — up to 10 classes
const CLASS_COLORS = [
  { bg: 'bg-blue-500',   text: 'text-blue-400',   border: 'border-blue-500',   hex: '#3b82f6' },
  { bg: 'bg-green-500',  text: 'text-green-400',  border: 'border-green-500',  hex: '#22c55e' },
  { bg: 'bg-red-500',    text: 'text-red-400',    border: 'border-red-500',    hex: '#ef4444' },
  { bg: 'bg-yellow-500', text: 'text-yellow-400', border: 'border-yellow-500', hex: '#eab308' },
  { bg: 'bg-purple-500', text: 'text-purple-400', border: 'border-purple-500', hex: '#a855f7' },
  { bg: 'bg-pink-500',   text: 'text-pink-400',   border: 'border-pink-500',   hex: '#ec4899' },
  { bg: 'bg-cyan-500',   text: 'text-cyan-400',   border: 'border-cyan-500',   hex: '#06b6d4' },
  { bg: 'bg-orange-500', text: 'text-orange-400', border: 'border-orange-500', hex: '#f97316' },
  { bg: 'bg-teal-500',   text: 'text-teal-400',   border: 'border-teal-500',   hex: '#14b8a6' },
  { bg: 'bg-rose-500',   text: 'text-rose-400',   border: 'border-rose-500',   hex: '#f43f5e' },
]

const RIBBON_LENGTH = 50  // how many recent predictions to show in the history ribbon

export default function InferencePage() {
  const { currentSession } = useSession()

  // --- Settings ---
  const [jobs, setJobs]           = useState([])    // completed training jobs
  const [runs, setRuns]           = useState([])    // pipeline runs for current session
  const [selectedJob, setSelectedJob] = useState('')
  const [selectedRun, setSelectedRun] = useState('')
  const [delayMs, setDelayMs]     = useState(500)

  // --- Stream state ---
  const [streaming, setStreaming]     = useState(false)
  const [streamStatus, setStreamStatus] = useState('idle')  // idle | connecting | streaming | done | error
  const [classNames, setClassNames]   = useState([])
  const [total, setTotal]             = useState(0)
  const [current, setCurrent]         = useState(null)    // latest prediction msg
  const [history, setHistory]         = useState([])      // last RIBBON_LENGTH predictions
  const [errorMsg, setErrorMsg]       = useState(null)

  const wsRef = useRef(null)

  // --- Load jobs on mount ---
  useEffect(() => {
    fetchJobs()
  }, [])

  // --- Load pipeline runs when session changes ---
  useEffect(() => {
    if (currentSession) fetchRuns()
  }, [currentSession?.session_id])

  async function fetchJobs() {
    try {
      const res  = await fetch(`${API}/api/train/jobs`)
      const json = await res.json()
      if (json.success) {
        const done = json.data.jobs.filter(j => j.status === 'done' && j.model_path)
        setJobs(done)
        if (done.length > 0) setSelectedJob(done[0].job_id)
      }
    } catch { /* backend not reachable */ }
  }

  async function fetchRuns() {
    if (!currentSession) return
    try {
      const res  = await fetch(`${API}/api/pipeline/runs`)
      const json = await res.json()
      if (json.success) {
        const sessionRuns = json.data.filter(r => r.session_id === currentSession.session_id)
        setRuns(sessionRuns)
        if (sessionRuns.length > 0) setSelectedRun(sessionRuns[sessionRuns.length - 1].run_id)
      }
    } catch { /* ignore */ }
  }

  // -------------------------------------------------------------------------
  // WebSocket streaming
  // -------------------------------------------------------------------------

  function startStream() {
    if (!selectedJob || !selectedRun) return

    // Reset display state
    setCurrent(null)
    setHistory([])
    setClassNames([])
    setTotal(0)
    setErrorMsg(null)
    setStreamStatus('connecting')
    setStreaming(true)

    const ws = new WebSocket(`${WS}/api/inference/ws/stream-run`)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type:     'start',
        job_id:   selectedJob,
        run_id:   selectedRun,
        delay_ms: delayMs,
      }))
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        handleMessage(msg)
      } catch { /* ignore malformed */ }
    }

    ws.onerror = () => {
      setErrorMsg('WebSocket connection failed. Is the backend running?')
      setStreamStatus('error')
      setStreaming(false)
    }

    ws.onclose = () => {
      setStreaming(false)
    }
  }

  function stopStream() {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }))
      wsRef.current.close()
    }
    setStreaming(false)
    setStreamStatus('idle')
  }

  function handleMessage(msg) {
    if (msg.type === 'ready') {
      setClassNames(msg.class_names)
      setTotal(msg.total)
      setStreamStatus('streaming')
    } else if (msg.type === 'prediction') {
      setCurrent(msg)
      setHistory(prev => {
        const next = [...prev, msg]
        return next.length > RIBBON_LENGTH ? next.slice(next.length - RIBBON_LENGTH) : next
      })
    } else if (msg.type === 'done') {
      setStreamStatus('done')
      setStreaming(false)
    } else if (msg.type === 'error') {
      setErrorMsg(msg.message)
      setStreamStatus('error')
      setStreaming(false)
    }
  }

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => wsRef.current?.close()
  }, [])

  // -------------------------------------------------------------------------
  // Derived display values
  // -------------------------------------------------------------------------

  const colorMap = Object.fromEntries(
    classNames.map((name, i) => [name, CLASS_COLORS[i % CLASS_COLORS.length]])
  )

  const selectedJobInfo = jobs.find(j => j.job_id === selectedJob)

  const progress = total > 0 && current
    ? Math.round(((current.window_index + 1) / total) * 100)
    : 0

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="max-w-5xl">
      <h1 className="text-2xl font-bold text-gray-100 mb-1">Inference</h1>
      <p className="text-gray-400 text-sm mb-8">
        Stream a pipeline run through a trained model and watch predictions in real time.
      </p>

      <div className="grid grid-cols-[280px_1fr] gap-6">

        {/* ---------------------------------------------------------------- */}
        {/* LEFT: settings panel                                              */}
        {/* ---------------------------------------------------------------- */}
        <div className="space-y-5">

          {/* Model selector */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Trained Model</p>
            {jobs.length === 0 ? (
              <p className="text-yellow-400 text-xs">
                No trained models found.{' '}
                <a href="/train" className="underline hover:text-yellow-200">Train one first.</a>
              </p>
            ) : (
              <select
                value={selectedJob}
                onChange={e => setSelectedJob(e.target.value)}
                disabled={streaming}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
              >
                {jobs.map(j => (
                  <option key={j.job_id} value={j.job_id}>
                    {j.model_name
                      ? `${j.model_name} (${j.classes?.join('/')})`
                      : j.classes?.join(' / ')
                    }{' '}
                    — {j.final_val_acc != null ? `${(j.final_val_acc * 100).toFixed(1)}% val` : 'no metrics'}
                  </option>
                ))}
              </select>
            )}
            {selectedJobInfo && (
              <p className="text-xs text-gray-600 mt-1 font-mono truncate">
                {selectedJobInfo.model_name
                  ? <span className="text-purple-500 not-italic">{selectedJobInfo.model_name}</span>
                  : null
                }
                {' '}{selectedJobInfo.job_id.slice(0, 16)}…
              </p>
            )}
          </div>

          {/* Pipeline run selector */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Data (Pipeline Run)</p>
            {!currentSession ? (
              <p className="text-yellow-400 text-xs">
                <a href="/import" className="underline">Load a session</a> first.
              </p>
            ) : runs.length === 0 ? (
              <p className="text-yellow-400 text-xs">
                No pipeline runs for this session.{' '}
                <a href="/preprocess" className="underline">Run the pipeline first.</a>
              </p>
            ) : (
              <select
                value={selectedRun}
                onChange={e => setSelectedRun(e.target.value)}
                disabled={streaming}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
              >
                {runs.map(r => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.name
                      ? `${r.name} — ${r.n_windows} windows`
                      : `${r.n_windows} windows — ${r.created_at?.slice(0, 10)}`
                    }
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Speed control */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
              Speed — {delayMs} ms / window
            </p>
            <input
              type="range"
              min={100} max={2000} step={100}
              value={delayMs}
              onChange={e => setDelayMs(Number(e.target.value))}
              disabled={streaming}
              className="w-full accent-purple-500 disabled:opacity-50"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>Fast (100ms)</span>
              <span>Slow (2s)</span>
            </div>
          </div>

          {/* Start / Stop button */}
          <button
            onClick={streaming ? stopStream : startStream}
            disabled={!selectedJob || !selectedRun}
            className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 ${
              streaming
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-purple-600 hover:bg-purple-500 text-white'
            }`}
          >
            {streaming ? 'Stop' : streamStatus === 'done' ? 'Run Again' : 'Start Inference'}
          </button>

          {errorMsg && (
            <p className="text-red-400 text-xs">{errorMsg}</p>
          )}

          {/* Status indicator */}
          {streamStatus === 'streaming' && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500" />
              </span>
              Window {current ? current.window_index + 1 : 0} / {total}
            </div>
          )}
          {streamStatus === 'done' && (
            <p className="text-green-400 text-xs font-medium">Stream complete — {total} windows</p>
          )}
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* RIGHT: live display                                               */}
        {/* ---------------------------------------------------------------- */}
        <div className="space-y-4">

          {/* Current prediction — large display */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            {!current ? (
              <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
                {streamStatus === 'connecting' ? 'Connecting…' : 'Waiting for predictions…'}
              </div>
            ) : (
              <>
                <p className="text-xs text-gray-500 mb-3">Current Prediction</p>
                {/* Big class badge */}
                <div className="flex items-center gap-4 mb-4">
                  <div className={`px-6 py-3 rounded-xl text-2xl font-bold text-white ${
                    colorMap[current.predicted_class]?.bg || 'bg-gray-700'
                  }`}>
                    {current.predicted_class}
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-gray-100">
                      {(current.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">confidence</p>
                  </div>
                </div>

                {/* Per-class confidence bars */}
                <div className="space-y-2">
                  {Object.entries(current.scores).map(([cls, score]) => {
                    const color = colorMap[cls]
                    return (
                      <div key={cls} className="flex items-center gap-3">
                        <span className={`text-xs w-16 text-right font-medium ${color?.text || 'text-gray-400'}`}>
                          {cls}
                        </span>
                        <div className="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-200 ${color?.bg || 'bg-gray-600'}`}
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 w-10 text-right font-mono">
                          {(score * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>

                {/* Progress bar */}
                {total > 0 && (
                  <div className="mt-4">
                    <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-600 rounded-full transition-all duration-200"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-600 mt-1 text-right">{progress}% complete</p>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Prediction history ribbon */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-3">
              Prediction History
              {history.length > 0 && <span className="ml-2 text-gray-600">— last {history.length}</span>}
            </p>

            {history.length === 0 ? (
              <div className="h-8 flex items-center text-gray-700 text-xs">No predictions yet</div>
            ) : (
              <>
                {/* Ribbon of colored blocks, one per prediction */}
                <div className="flex gap-0.5 flex-wrap mb-3">
                  {history.map((pred, i) => {
                    const color = colorMap[pred.predicted_class]
                    return (
                      <div
                        key={i}
                        title={`${pred.predicted_class} (${(pred.confidence * 100).toFixed(0)}%)`}
                        className={`h-6 w-3 rounded-sm ${color?.bg || 'bg-gray-600'}`}
                        style={{ opacity: 0.4 + pred.confidence * 0.6 }}
                      />
                    )
                  })}
                </div>

                {/* Class legend + counts */}
                <div className="flex flex-wrap gap-3">
                  {classNames.map(cls => {
                    const count = history.filter(p => p.predicted_class === cls).length
                    const pct   = history.length > 0 ? Math.round((count / history.length) * 100) : 0
                    const color = colorMap[cls]
                    return (
                      <div key={cls} className="flex items-center gap-1.5">
                        <div className={`w-3 h-3 rounded-sm ${color?.bg || 'bg-gray-600'}`} />
                        <span className="text-xs text-gray-400">{cls}</span>
                        <span className="text-xs text-gray-600">{count} ({pct}%)</span>
                      </div>
                    )
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
