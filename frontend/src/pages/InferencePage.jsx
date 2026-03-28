// InferencePage.jsx — live EEG inference with three input sources + two output destinations
//
// Source toggle (top of page):
//   Simulated    — existing behaviour: stream stored pipeline-run windows
//   File Playback — replay any loaded DataSession's raw signal in real time
//   Hardware      — stream from a live BrainFlow device (boards/connect/disconnect UI)
//
// Output section (below source):
//   WebSocket Output — Neutrol starts a WS server; external code connects and receives predictions
//   Serial Output    — sends a configurable string to a COM port after each prediction
//
// All three sources send to the same /ws/stream-run WebSocket (with a mode field)
// except Hardware which uses /ws/hardware. The prediction display (badge + bars + ribbon)
// is fully shared — it handles messages from all three sources identically.

import { useState, useEffect, useRef } from 'react'
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

const RIBBON_LENGTH = 50

export default function InferencePage() {
  const { currentSession, sessions } = useSession()

  // Source mode
  const [source, setSource] = useState('simulated')  // 'simulated' | 'playback' | 'hardware'

  // Shared — model
  const [jobs, setJobs]               = useState([])
  const [selectedJob, setSelectedJob] = useState('')

  // Simulated + playback — pipeline run
  const [runs, setRuns]               = useState([])
  const [selectedRun, setSelectedRun] = useState('')
  const [delayMs, setDelayMs]         = useState(500)

  // Playback — session selector
  const [playbackSession, setPlaybackSession] = useState('')

  // Hardware
  const [boards, setBoards]                     = useState([])
  const [brainflowAvailable, setBrainflowAvailable] = useState(false)
  const [selectedBoard, setSelectedBoard]       = useState('synthetic')
  const [hwSerialPort, setHwSerialPort]         = useState('')
  const [hwIpAddress, setHwIpAddress]           = useState('')
  const [hwStatus, setHwStatus]                 = useState('disconnected')  // 'disconnected' | 'connecting' | 'connected'
  const [hwInfo, setHwInfo]                     = useState(null)   // { n_channels, sample_rate }
  const [hwError, setHwError]                   = useState(null)
  const [hwNSamples, setHwNSamples]             = useState(256)

  // Output config
  const [wsOutEnabled, setWsOutEnabled] = useState(false)
  const [wsOutPort, setWsOutPort]       = useState(8765)
  const [wsOutRunning, setWsOutRunning] = useState(false)
  const [wsOutClients, setWsOutClients] = useState(0)

  const [serialEnabled, setSerialEnabled]     = useState(false)
  const [serialPorts, setSerialPorts]         = useState([])
  const [selectedSerialPort, setSelectedSerialPort] = useState('')
  const [serialBaud, setSerialBaud]           = useState(9600)
  const [serialMap, setSerialMap]             = useState({})   // label → send string

  // Stream state
  const [streaming, setStreaming]         = useState(false)
  const [streamStatus, setStreamStatus]   = useState('idle')
  const [classNames, setClassNames]       = useState([])
  const [total, setTotal]                 = useState(0)
  const [current, setCurrent]             = useState(null)
  const [history, setHistory]             = useState([])
  const [errorMsg, setErrorMsg]           = useState(null)

  const wsRef = useRef(null)

  // ---- Load on mount ----
  useEffect(() => {
    fetchJobs()
    fetchBoards()
    fetchSerialPorts()
    fetchOutputConfig()
  }, [])

  useEffect(() => {
    fetchRuns()
  }, [currentSession?.session_id])

  // When class names arrive, seed the serialMap with sensible defaults
  useEffect(() => {
    if (classNames.length > 0) {
      setSerialMap(prev => {
        const updated = { ...prev }
        classNames.forEach((cls, i) => {
          if (!updated[cls]) updated[cls] = `${i}\n`
        })
        return updated
      })
    }
  }, [classNames])

  // -------------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------------

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
    try {
      const res  = await fetch(`${API}/api/pipeline/runs`)
      const json = await res.json()
      if (json.success) {
        const all = json.data
        setRuns(all)
        // Default to runs for the current session if available, else all
        const relevant = currentSession
          ? all.filter(r => r.session_id === currentSession.session_id)
          : all
        if (relevant.length > 0) setSelectedRun(relevant[relevant.length - 1].run_id)
        else if (all.length > 0) setSelectedRun(all[all.length - 1].run_id)
      }
    } catch { /* ignore */ }
  }

  async function fetchBoards() {
    try {
      const res  = await fetch(`${API}/api/inference/hardware/boards`)
      const json = await res.json()
      if (json.success) {
        setBoards(json.data.boards)
        setBrainflowAvailable(json.data.brainflow_available)
      }
    } catch { /* ignore */ }
  }

  async function fetchSerialPorts() {
    try {
      const res  = await fetch(`${API}/api/inference/output/serial-ports`)
      const json = await res.json()
      if (json.success) {
        setSerialPorts(json.data.ports)
        if (json.data.ports.length > 0 && !selectedSerialPort) {
          setSelectedSerialPort(json.data.ports[0].port)
        }
      }
    } catch { /* ignore */ }
  }

  async function fetchOutputConfig() {
    try {
      const res  = await fetch(`${API}/api/inference/output/config`)
      const json = await res.json()
      if (json.success) {
        const d = json.data
        setWsOutEnabled(d.ws_enabled)
        setWsOutPort(d.ws_port || 8765)
        setWsOutRunning(d.ws_running)
        setWsOutClients(d.ws_clients)
        setSerialEnabled(d.serial_enabled)
      }
    } catch { /* ignore */ }
  }

  // ---- Hardware connect / disconnect ----

  async function connectHardware() {
    setHwStatus('connecting')
    setHwError(null)
    try {
      const res  = await fetch(`${API}/api/inference/hardware/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          board_name:  selectedBoard,
          serial_port: hwSerialPort,
          ip_address:  hwIpAddress,
        }),
      })
      const json = await res.json()
      if (json.success) {
        setHwStatus('connected')
        setHwInfo({ n_channels: json.data.n_channels, sample_rate: json.data.sample_rate })
      } else {
        setHwStatus('disconnected')
        setHwError(json.error)
      }
    } catch {
      setHwStatus('disconnected')
      setHwError('Could not reach backend.')
    }
  }

  async function disconnectHardware() {
    await fetch(`${API}/api/inference/hardware/disconnect`, { method: 'POST' })
    setHwStatus('disconnected')
    setHwInfo(null)
  }

  // ---- Apply output config ----

  async function applyOutputConfig() {
    const body = {
      ws_enabled:     wsOutEnabled,
      ws_port:        wsOutPort,
      serial_enabled: serialEnabled,
      serial_port:    selectedSerialPort,
      serial_baud:    serialBaud,
      serial_map:     serialMap,
    }
    try {
      const res  = await fetch(`${API}/api/inference/output/config`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      })
      const json = await res.json()
      if (json.success) {
        setWsOutRunning(json.data.ws_running)
        setWsOutClients(json.data.ws_clients)
      }
    } catch { /* ignore */ }
  }

  // -------------------------------------------------------------------------
  // WebSocket streaming
  // -------------------------------------------------------------------------

  function startStream() {
    setCurrent(null)
    setHistory([])
    setClassNames([])
    setTotal(0)
    setErrorMsg(null)
    setStreamStatus('connecting')
    setStreaming(true)

    let wsUrl, startMsg

    if (source === 'hardware') {
      wsUrl    = `${WS}/api/inference/ws/hardware`
      startMsg = { type: 'start', job_id: selectedJob, n_samples: hwNSamples, delay_ms: delayMs }
    } else {
      wsUrl    = `${WS}/api/inference/ws/stream-run`
      if (source === 'playback') {
        startMsg = { type: 'start', mode: 'playback', job_id: selectedJob,
                     session_id: playbackSession || currentSession?.session_id,
                     run_id: selectedRun, delay_ms: delayMs }
      } else {
        startMsg = { type: 'start', mode: 'run', job_id: selectedJob, run_id: selectedRun, delay_ms: delayMs }
      }
    }

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen    = () => ws.send(JSON.stringify(startMsg))
    ws.onmessage = (event) => {
      try { handleMessage(JSON.parse(event.data)) } catch { /* ignore malformed */ }
    }
    ws.onerror   = () => {
      setErrorMsg('WebSocket connection failed. Is the backend running?')
      setStreamStatus('error')
      setStreaming(false)
    }
    ws.onclose   = () => setStreaming(false)
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
      setTotal(msg.total ?? 0)
      setStreamStatus('streaming')
    } else if (msg.type === 'prediction') {
      const pred = { predicted_class: msg.predicted_class, confidence: msg.confidence, scores: msg.scores, window_index: msg.window_index ?? 0 }
      setCurrent(pred)
      setHistory(prev => {
        const next = [...prev, pred]
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

  useEffect(() => { return () => wsRef.current?.close() }, [])

  // -------------------------------------------------------------------------
  // Derived display values
  // -------------------------------------------------------------------------

  const colorMap = Object.fromEntries(classNames.map((n, i) => [n, CLASS_COLORS[i % CLASS_COLORS.length]]))

  const progress = total > 0 && current?.window_index != null
    ? Math.round(((current.window_index + 1) / total) * 100)
    : 0

  const selectedBoardInfo = boards.find(b => b.name === selectedBoard)
  const needsSerial  = ['cyton', 'cyton_daisy', 'ganglion'].includes(selectedBoard)
  const needsIp      = false  // no WiFi boards in current list

  const canStart = selectedJob && (
    source === 'simulated' ? !!selectedRun :
    source === 'playback'  ? !!(playbackSession || currentSession?.session_id) && !!selectedRun :
    /* hardware */           hwStatus === 'connected'
  )

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="max-w-5xl">
      <h1 className="text-2xl font-bold text-gray-100 mb-1">Inference</h1>
      <p className="text-gray-400 text-sm mb-6">
        Stream EEG through a trained model and watch predictions in real time.
      </p>

      {/* ------------------------------------------------------------------ */}
      {/* Source Toggle                                                        */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex gap-1 mb-6 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
        {[
          { key: 'simulated', label: 'Simulated' },
          { key: 'playback',  label: 'File Playback' },
          { key: 'hardware',  label: 'Hardware' },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => { if (!streaming) setSource(key) }}
            disabled={streaming}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:cursor-not-allowed
              ${source === key
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-gray-200'
              }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-[300px_1fr] gap-6">

        {/* ---------------------------------------------------------------- */}
        {/* LEFT: settings panel                                              */}
        {/* ---------------------------------------------------------------- */}
        <div className="space-y-4">

          {/* Model selector — shared across all modes */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Trained Model</p>
            {jobs.length === 0 ? (
              <p className="text-yellow-400 text-xs">
                No trained models.{' '}
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
                    {j.model_name ? `${j.model_name}` : j.classes?.join('/')}
                    {j.final_val_acc != null ? ` — ${(j.final_val_acc * 100).toFixed(1)}% val` : ''}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* ---- Source-specific settings ---- */}

          {source === 'simulated' && (
            <SimulatedPanel
              runs={runs}
              selectedRun={selectedRun}
              setSelectedRun={setSelectedRun}
              streaming={streaming}
            />
          )}

          {source === 'playback' && (
            <PlaybackPanel
              sessions={sessions}
              playbackSession={playbackSession}
              setPlaybackSession={setPlaybackSession}
              runs={runs}
              selectedRun={selectedRun}
              setSelectedRun={setSelectedRun}
              streaming={streaming}
            />
          )}

          {source === 'hardware' && (
            <HardwarePanel
              boards={boards}
              brainflowAvailable={brainflowAvailable}
              selectedBoard={selectedBoard}
              setSelectedBoard={setSelectedBoard}
              hwSerialPort={hwSerialPort}
              setHwSerialPort={setHwSerialPort}
              hwIpAddress={hwIpAddress}
              setHwIpAddress={setHwIpAddress}
              hwStatus={hwStatus}
              hwInfo={hwInfo}
              hwError={hwError}
              hwNSamples={hwNSamples}
              setHwNSamples={setHwNSamples}
              needsSerial={needsSerial}
              onConnect={connectHardware}
              onDisconnect={disconnectHardware}
              streaming={streaming}
            />
          )}

          {/* Speed slider — shown for simulated and playback */}
          {source !== 'hardware' && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
                Speed — {delayMs} ms / window
              </p>
              <input
                type="range" min={100} max={2000} step={100}
                value={delayMs}
                onChange={e => setDelayMs(Number(e.target.value))}
                disabled={streaming}
                className="w-full accent-purple-500 disabled:opacity-50"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>Fast (100ms)</span><span>Slow (2s)</span>
              </div>
            </div>
          )}

          {/* Hardware — polling interval */}
          {source === 'hardware' && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
                Poll interval — {delayMs} ms
              </p>
              <input
                type="range" min={100} max={2000} step={100}
                value={delayMs}
                onChange={e => setDelayMs(Number(e.target.value))}
                disabled={streaming}
                className="w-full accent-purple-500 disabled:opacity-50"
              />
            </div>
          )}

          {/* Start / Stop */}
          <button
            onClick={streaming ? stopStream : startStream}
            disabled={!canStart}
            className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 ${
              streaming
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-purple-600 hover:bg-purple-500 text-white'
            }`}
          >
            {streaming ? 'Stop' : streamStatus === 'done' ? 'Run Again' : 'Start Inference'}
          </button>

          {errorMsg && <p className="text-red-400 text-xs">{errorMsg}</p>}

          {streamStatus === 'streaming' && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500" />
              </span>
              {source === 'hardware' ? 'Streaming from hardware…' : `Window ${(current?.window_index ?? 0) + 1} / ${total}`}
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

          {/* Current prediction */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            {!current ? (
              <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
                {streamStatus === 'connecting' ? 'Connecting…' : 'Waiting for predictions…'}
              </div>
            ) : (
              <>
                <p className="text-xs text-gray-500 mb-3">Current Prediction</p>
                <div className="flex items-center gap-4 mb-4">
                  <div className={`px-6 py-3 rounded-xl text-2xl font-bold text-white ${colorMap[current.predicted_class]?.bg || 'bg-gray-700'}`}>
                    {current.predicted_class}
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-gray-100">{(current.confidence * 100).toFixed(1)}%</p>
                    <p className="text-xs text-gray-500">confidence</p>
                  </div>
                </div>

                <div className="space-y-2">
                  {Object.entries(current.scores).map(([cls, score]) => {
                    const color = colorMap[cls]
                    return (
                      <div key={cls} className="flex items-center gap-3">
                        <span className={`text-xs w-16 text-right font-medium ${color?.text || 'text-gray-400'}`}>{cls}</span>
                        <div className="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full transition-all duration-200 ${color?.bg || 'bg-gray-600'}`} style={{ width: `${score * 100}%` }} />
                        </div>
                        <span className="text-xs text-gray-500 w-10 text-right font-mono">{(score * 100).toFixed(1)}%</span>
                      </div>
                    )
                  })}
                </div>

                {total > 0 && source !== 'hardware' && (
                  <div className="mt-4">
                    <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full bg-purple-600 rounded-full transition-all duration-200" style={{ width: `${progress}%` }} />
                    </div>
                    <p className="text-xs text-gray-600 mt-1 text-right">{progress}% complete</p>
                  </div>
                )}
              </>
            )}
          </div>

          {/* History ribbon */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-3">
              Prediction History
              {history.length > 0 && <span className="ml-2 text-gray-600">— last {history.length}</span>}
            </p>
            {history.length === 0 ? (
              <div className="h-8 flex items-center text-gray-700 text-xs">No predictions yet</div>
            ) : (
              <>
                <div className="flex gap-0.5 flex-wrap mb-3">
                  {history.map((pred, i) => (
                    <div
                      key={i}
                      title={`${pred.predicted_class} (${(pred.confidence * 100).toFixed(0)}%)`}
                      className={`h-6 w-3 rounded-sm ${colorMap[pred.predicted_class]?.bg || 'bg-gray-600'}`}
                      style={{ opacity: 0.4 + pred.confidence * 0.6 }}
                    />
                  ))}
                </div>
                <div className="flex flex-wrap gap-3">
                  {classNames.map(cls => {
                    const count = history.filter(p => p.predicted_class === cls).length
                    const pct   = history.length > 0 ? Math.round((count / history.length) * 100) : 0
                    return (
                      <div key={cls} className="flex items-center gap-1.5">
                        <div className={`w-3 h-3 rounded-sm ${colorMap[cls]?.bg || 'bg-gray-600'}`} />
                        <span className="text-xs text-gray-400">{cls}</span>
                        <span className="text-xs text-gray-600">{count} ({pct}%)</span>
                      </div>
                    )
                  })}
                </div>
              </>
            )}
          </div>

          {/* ---------------------------------------------------------------- */}
          {/* Output section                                                    */}
          {/* ---------------------------------------------------------------- */}
          <div>
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Output</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

              {/* WebSocket Output */}
              <OutputCard title="WebSocket Server" badge="ws://" badgeColor="blue">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-400">Send predictions to connected clients</span>
                  <ToggleSwitch enabled={wsOutEnabled} onChange={v => setWsOutEnabled(v)} />
                </div>

                {wsOutEnabled && (
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-12">Port</span>
                      <input
                        type="number"
                        value={wsOutPort}
                        onChange={e => setWsOutPort(Number(e.target.value))}
                        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 w-20 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${wsOutRunning ? 'bg-green-500' : 'bg-gray-600'}`} />
                      <span className="text-xs text-gray-500">
                        {wsOutRunning ? `Running on ws://localhost:${wsOutPort} — ${wsOutClients} client${wsOutClients !== 1 ? 's' : ''}` : 'Stopped'}
                      </span>
                    </div>
                  </div>
                )}

                {wsOutEnabled && (
                  <details className="mb-3">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">Show Python client example</summary>
                    <pre className="mt-2 text-xs text-green-400 bg-gray-950 rounded p-2 overflow-x-auto leading-relaxed">{
`import asyncio, websockets, json

async def receive():
    async with websockets.connect(
        "ws://localhost:${wsOutPort}"
    ) as ws:
        async for msg in ws:
            data = json.loads(msg)
            print(data["label"], data["confidence"])

asyncio.run(receive())`
                    }</pre>
                  </details>
                )}

                <button
                  onClick={applyOutputConfig}
                  className="w-full py-1.5 rounded bg-blue-700 hover:bg-blue-600 text-xs text-white font-medium transition-colors"
                >
                  Apply
                </button>
              </OutputCard>

              {/* Serial Output */}
              <OutputCard title="Serial Port" badge="COM" badgeColor="green">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-400">Send string to microcontroller</span>
                  <ToggleSwitch enabled={serialEnabled} onChange={v => setSerialEnabled(v)} />
                </div>

                {serialEnabled && (
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-12">Port</span>
                      <select
                        value={selectedSerialPort}
                        onChange={e => setSelectedSerialPort(e.target.value)}
                        className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-green-500"
                      >
                        {serialPorts.length === 0 && (
                          <option value="">No ports found</option>
                        )}
                        {serialPorts.map(p => (
                          <option key={p.port} value={p.port}>{p.port} — {p.description}</option>
                        ))}
                      </select>
                      <button onClick={fetchSerialPorts} className="text-xs text-gray-600 hover:text-gray-400">↻</button>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-12">Baud</span>
                      <select
                        value={serialBaud}
                        onChange={e => setSerialBaud(Number(e.target.value))}
                        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-green-500"
                      >
                        {[9600, 57600, 115200].map(b => <option key={b} value={b}>{b}</option>)}
                      </select>
                    </div>

                    {/* Per-class string map */}
                    {classNames.length > 0 && (
                      <div>
                        <p className="text-xs text-gray-600 mb-1">Send string per class:</p>
                        <div className="space-y-1">
                          {classNames.map(cls => (
                            <div key={cls} className="flex items-center gap-2">
                              <span className={`text-xs w-16 font-medium ${colorMap[cls]?.text || 'text-gray-400'}`}>{cls}</span>
                              <span className="text-gray-600 text-xs">→</span>
                              <input
                                value={serialMap[cls] ?? `${classNames.indexOf(cls)}\n`}
                                onChange={e => setSerialMap(prev => ({ ...prev, [cls]: e.target.value }))}
                                className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-green-500"
                                placeholder={`${classNames.indexOf(cls)}\\n`}
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {classNames.length === 0 && (
                      <p className="text-xs text-gray-600">Run inference first to see class labels.</p>
                    )}
                  </div>
                )}

                <button
                  onClick={applyOutputConfig}
                  className="w-full py-1.5 rounded bg-green-700 hover:bg-green-600 text-xs text-white font-medium transition-colors"
                >
                  Apply
                </button>
              </OutputCard>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


// ---------------------------------------------------------------------------
// Source-specific panels
// ---------------------------------------------------------------------------

function SimulatedPanel({ runs, selectedRun, setSelectedRun, streaming }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Pipeline Run</p>
      {runs.length === 0 ? (
        <p className="text-yellow-400 text-xs">
          No pipeline runs.{' '}
          <a href="/preprocess" className="underline">Run pipeline first.</a>
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
              {r.name || `${r.n_windows} windows`} — {r.created_at?.slice(0, 10)}
            </option>
          ))}
        </select>
      )}
      <p className="text-xs text-gray-600 mt-2">Replays stored windows from a pipeline run at the chosen speed.</p>
    </div>
  )
}

function PlaybackPanel({ sessions, playbackSession, setPlaybackSession, runs, selectedRun, setSelectedRun, streaming }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
      <p className="text-xs text-gray-500 uppercase tracking-wider">Session to Replay</p>
      {sessions.length === 0 ? (
        <p className="text-yellow-400 text-xs">No sessions loaded. <a href="/import" className="underline">Import one first.</a></p>
      ) : (
        <select
          value={playbackSession}
          onChange={e => setPlaybackSession(e.target.value)}
          disabled={streaming}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
        >
          {sessions.map(s => (
            <option key={s.session_id} value={s.session_id}>
              {s.file_path?.split(/[\\/]/).pop() || s.session_id.slice(0, 8)} — {s.n_channels}ch {s.sample_rate}Hz
            </option>
          ))}
        </select>
      )}

      <p className="text-xs text-gray-500 uppercase tracking-wider">Window Shape (from run)</p>
      {runs.length === 0 ? (
        <p className="text-yellow-400 text-xs">No pipeline runs. <a href="/preprocess" className="underline">Run pipeline first.</a></p>
      ) : (
        <select
          value={selectedRun}
          onChange={e => setSelectedRun(e.target.value)}
          disabled={streaming}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
        >
          {runs.map(r => (
            <option key={r.run_id} value={r.run_id}>
              {r.name || `${r.n_windows} windows`} — {r.window_shape ? `${r.window_shape[0]}ch × ${r.window_shape[1]}samp` : ''}
            </option>
          ))}
        </select>
      )}
      <p className="text-xs text-gray-600">Slices the raw session signal into windows matching the selected run's shape.</p>
    </div>
  )
}

function HardwarePanel({
  boards, brainflowAvailable, selectedBoard, setSelectedBoard,
  hwSerialPort, setHwSerialPort, hwIpAddress, setHwIpAddress,
  hwStatus, hwInfo, hwError, hwNSamples, setHwNSamples,
  needsSerial, onConnect, onDisconnect, streaming,
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Hardware Device</p>
        {/* Status badge */}
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${hwStatus === 'connected' ? 'bg-green-500' : hwStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 'bg-gray-600'}`} />
          <span className="text-xs text-gray-500 capitalize">{hwStatus}</span>
        </div>
      </div>

      {!brainflowAvailable && (
        <p className="text-yellow-400 text-xs">BrainFlow not installed. Run: <span className="font-mono">pip install brainflow</span></p>
      )}

      <select
        value={selectedBoard}
        onChange={e => setSelectedBoard(e.target.value)}
        disabled={hwStatus !== 'disconnected' || streaming}
        className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
      >
        {boards.map(b => (
          <option key={b.name} value={b.name}>
            {b.name} — {b.n_channels}ch {b.sample_rate}Hz
          </option>
        ))}
        {boards.length === 0 && <option value="synthetic">synthetic (no BrainFlow)</option>}
      </select>

      {needsSerial && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 w-16">COM port</span>
          <input
            value={hwSerialPort}
            onChange={e => setHwSerialPort(e.target.value)}
            placeholder="COM3 or /dev/ttyUSB0"
            disabled={hwStatus !== 'disconnected'}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
          />
        </div>
      )}

      {hwInfo && (
        <p className="text-xs text-green-400">
          {hwInfo.n_channels} channels · {hwInfo.sample_rate} Hz
        </p>
      )}

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 w-16">Window</span>
        <input
          type="number" min={64} max={2048} step={64}
          value={hwNSamples}
          onChange={e => setHwNSamples(Number(e.target.value))}
          disabled={streaming}
          className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-purple-500 disabled:opacity-50"
        />
        <span className="text-xs text-gray-600">samples</span>
      </div>

      {hwError && <p className="text-red-400 text-xs">{hwError}</p>}

      {hwStatus === 'disconnected' ? (
        <button
          onClick={onConnect}
          disabled={!brainflowAvailable || streaming}
          className="w-full py-2 rounded bg-green-700 hover:bg-green-600 disabled:opacity-40 text-xs text-white font-medium transition-colors"
        >
          Connect
        </button>
      ) : hwStatus === 'connecting' ? (
        <button disabled className="w-full py-2 rounded bg-gray-700 text-xs text-gray-400">Connecting…</button>
      ) : (
        <button
          onClick={onDisconnect}
          disabled={streaming}
          className="w-full py-2 rounded bg-red-800 hover:bg-red-700 disabled:opacity-40 text-xs text-white font-medium transition-colors"
        >
          Disconnect
        </button>
      )}
    </div>
  )
}


// ---------------------------------------------------------------------------
// Output UI helpers
// ---------------------------------------------------------------------------

function OutputCard({ title, badge, badgeColor, children }) {
  const badgeColors = {
    blue:  'bg-blue-900/40 border-blue-700 text-blue-300',
    green: 'bg-green-900/40 border-green-700 text-green-300',
  }
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm font-medium text-gray-200">{title}</span>
        <span className={`text-xs px-1.5 py-0.5 rounded border font-mono ${badgeColors[badgeColor]}`}>{badge}</span>
      </div>
      {children}
    </div>
  )
}

function ToggleSwitch({ enabled, onChange }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${enabled ? 'bg-purple-600' : 'bg-gray-700'}`}
    >
      <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${enabled ? 'translate-x-5' : 'translate-x-1'}`} />
    </button>
  )
}
