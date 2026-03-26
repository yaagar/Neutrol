// LabelPage.jsx — EEG segment labeling tool
//
// Workflow on this page:
//   Step 1: If no session loaded → tell user to go to Import
//   Step 2: If no pipeline run exists for this session → show a "Process Signal" panel
//           so the user can run a default pipeline right here
//   Step 3: Show windows one by one with a waveform chart
//           User clicks a label button → label is stored in local state
//           After labeling windows, user hits "Save Labels" to persist to backend
//
// All API calls use async/await as required by CLAUDE.md.
import { useState, useEffect, useCallback } from 'react'
import { useSession } from '../hooks/useSession'
import WaveformChart from '../components/WaveformChart'

const API = 'http://localhost:8000'

// Predefined label options — the most common BCI classes.
// The user can also type a custom label.
const PRESET_LABELS = ['Rest', 'Left', 'Right', 'Blink']

// Default pipeline config used when the user clicks "Process Signal".
// Covers the standard EEG preprocessing chain.
const DEFAULT_PIPELINE = {
  steps: [
    { step: 'bandpass', low_hz: 1.0,  high_hz: 40.0 },
    { step: 'notch',    freq_hz: 60.0, quality_factor: 30.0 },
    { step: 'normalize',method: 'zscore' },
    { step: 'window',   window_sec: 1.0, overlap_ratio: 0.5 },
  ],
}

export default function LabelPage() {
  const { currentSession } = useSession()

  // --- Pipeline run state ---
  const [runs, setRuns]           = useState([])       // all runs for current session
  const [activeRun, setActiveRun] = useState(null)     // the run we're labeling

  // --- Pipeline setup state (shown when no run exists) ---
  const [pipelineConfig, setPipelineConfig] = useState({
    lowHz:       1.0,
    highHz:      40.0,
    notchHz:     60.0,
    windowSec:   1.0,
    overlapRatio: 0.5,
  })
  const [processing, setProcessing]   = useState(false)
  const [processError, setProcessError] = useState(null)

  // --- Window navigation state ---
  const [windowIndex, setWindowIndex]   = useState(0)
  const [windowData, setWindowData]     = useState(null)   // Recharts data for current window
  const [windowLoading, setWindowLoading] = useState(false)

  // --- Labeling state ---
  // labels is a map of windowIndex → label string
  // We use an object so accessing and updating a specific index is O(1)
  const [labels, setLabels]           = useState({})
  const [customLabel, setCustomLabel] = useState('')
  const [saving, setSaving]           = useState(false)
  const [saveResult, setSaveResult]   = useState(null)   // {success, message}

  // --- Fetch runs when session changes ---
  useEffect(() => {
    if (!currentSession) return
    setLabels({})        // clear labels when switching sessions
    setWindowIndex(0)
    setWindowData(null)
    setActiveRun(null)
    fetchRuns()
  }, [currentSession?.session_id])

  // --- Fetch window data when index or active run changes ---
  useEffect(() => {
    if (activeRun && currentSession) {
      fetchWindow(activeRun.run_id, windowIndex)
    }
  }, [activeRun?.run_id, windowIndex])

  // --- Load existing labels from backend when a run is selected ---
  useEffect(() => {
    if (activeRun && currentSession) {
      loadExistingLabels()
    }
  }, [activeRun?.run_id])

  // -------------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------------

  async function fetchRuns() {
    try {
      const res = await fetch(`${API}/api/pipeline/runs`)
      const json = await res.json()
      if (json.success) {
        // Filter to only runs belonging to the current session
        const sessionRuns = json.data.filter(
          r => r.session_id === currentSession.session_id
        )
        setRuns(sessionRuns)
        // Auto-select the most recent run if one exists
        if (sessionRuns.length > 0) {
          setActiveRun(sessionRuns[sessionRuns.length - 1])
        }
      }
    } catch {
      // Backend not reachable — runs stay empty, user sees the pipeline setup panel
    }
  }

  async function runPipeline() {
    setProcessing(true)
    setProcessError(null)
    try {
      const config = {
        steps: [
          { step: 'bandpass', low_hz: pipelineConfig.lowHz,  high_hz: pipelineConfig.highHz },
          { step: 'notch',    freq_hz: pipelineConfig.notchHz, quality_factor: 30.0 },
          { step: 'normalize', method: 'zscore' },
          { step: 'window',   window_sec: pipelineConfig.windowSec, overlap_ratio: pipelineConfig.overlapRatio },
        ],
      }
      const res = await fetch(`${API}/api/pipeline/run`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: currentSession.session_id, pipeline_config: config }),
      })
      const json = await res.json()
      if (json.success) {
        // Refresh the run list — the new run will be auto-selected
        await fetchRuns()
        setWindowIndex(0)
      } else {
        setProcessError(json.error || 'Pipeline failed.')
      }
    } catch {
      setProcessError('Could not reach the backend. Is it running on port 8000?')
    } finally {
      setProcessing(false)
    }
  }

  async function fetchWindow(runId, index) {
    setWindowLoading(true)
    try {
      const res = await fetch(`${API}/api/pipeline/runs/${runId}/window/${index}`)
      const json = await res.json()
      if (json.success) {
        setWindowData(json.data)
      }
    } catch {
      setWindowData(null)
    } finally {
      setWindowLoading(false)
    }
  }

  async function loadExistingLabels() {
    try {
      const res = await fetch(`${API}/api/labels/${currentSession.session_id}`)
      const json = await res.json()
      if (json.success && json.data.labels.length > 0) {
        // Reconstruct the {windowIndex: label} map from saved labels.
        // We match by start_time_sec order since window_index isn't stored directly.
        // For simplicity, we just prefill labels in order.
        const rebuilt = {}
        json.data.labels.forEach((lbl, i) => { rebuilt[i] = lbl.label })
        setLabels(rebuilt)
      }
    } catch {
      // No existing labels — start fresh
    }
  }

  async function saveLabels() {
    if (!activeRun) return
    const labelEntries = Object.entries(labels)
    if (labelEntries.length === 0) {
      setSaveResult({ success: false, message: 'No labels to save. Label at least one window first.' })
      return
    }

    setSaving(true)
    setSaveResult(null)

    // Build the payload: one entry per labeled window
    const labelsPayload = labelEntries.map(([idxStr, label]) => {
      const idx = parseInt(idxStr)
      // Approximate start time using window_shape and index
      const windowSamples = activeRun.window_shape[1]
      const approxStartSec = parseFloat((idx * windowSamples * 0.5 / (windowData?.sample_rate || 256)).toFixed(3))
      return { window_index: idx, label, start_time_sec: approxStartSec }
    })

    try {
      const res = await fetch(`${API}/api/labels/save`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          session_id: currentSession.session_id,
          run_id:     activeRun.run_id,
          labels:     labelsPayload,
        }),
      })
      const json = await res.json()
      if (json.success) {
        setSaveResult({
          success: true,
          message: `Saved ${json.data.total_labeled} labels — ${
            Object.entries(json.data.label_counts)
              .map(([l, c]) => `${c}× ${l}`)
              .join(', ')
          }`,
        })
      } else {
        setSaveResult({ success: false, message: json.error || 'Save failed.' })
      }
    } catch {
      setSaveResult({ success: false, message: 'Could not reach the backend.' })
    } finally {
      setSaving(false)
    }
  }

  // -------------------------------------------------------------------------
  // Label assignment helpers
  // -------------------------------------------------------------------------

  function assignLabel(label) {
    setLabels(prev => ({ ...prev, [windowIndex]: label }))
    // Auto-advance to the next unlabeled window after assigning
    if (activeRun && windowIndex < activeRun.n_windows - 1) {
      const nextUnlabeled = findNextUnlabeled(windowIndex + 1, activeRun.n_windows)
      if (nextUnlabeled !== null) setWindowIndex(nextUnlabeled)
    }
  }

  // Find the next window that hasn't been labeled yet, starting from `from`
  function findNextUnlabeled(from, total) {
    for (let i = from; i < total; i++) {
      if (!labels[i]) return i
    }
    return null  // all labeled from here
  }

  const labeledCount  = Object.keys(labels).length
  const totalWindows  = activeRun?.n_windows || 0
  const progressPct   = totalWindows > 0 ? Math.round((labeledCount / totalWindows) * 100) : 0

  // -------------------------------------------------------------------------
  // Guard: no session loaded
  // -------------------------------------------------------------------------

  if (!currentSession) {
    return (
      <div className="max-w-xl">
        <h1 className="text-2xl font-bold text-gray-100 mb-2">Label</h1>
        <p className="text-gray-400 text-sm">
          No session loaded. Go to{' '}
          <a href="/import" className="text-purple-400 hover:underline">Import</a>{' '}
          and upload an EEG file first.
        </p>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Guard: no pipeline run — show the pipeline setup panel
  // -------------------------------------------------------------------------

  if (runs.length === 0) {
    return (
      <div className="max-w-2xl">
        <h1 className="text-2xl font-bold text-gray-100 mb-1">Label</h1>
        <p className="text-gray-400 text-sm mb-8">
          Session: <span className="font-mono text-gray-300">
            {currentSession.file_path?.split(/[\\/]/).pop()}
          </span> — {currentSession.n_channels} channels, {currentSession.sample_rate} Hz,{' '}
          {currentSession.duration_seconds?.toFixed(1)}s
        </p>

        {/* No runs exist — user must process the signal first */}
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-6">
          <h2 className="text-base font-semibold text-gray-200 mb-1">
            Process Signal First
          </h2>
          <p className="text-gray-500 text-sm mb-6">
            Before labeling, the signal must be filtered and cut into windows.
            Configure the pipeline below and click "Process Signal".
          </p>

          {/* Pipeline configuration form */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Bandpass low (Hz)</span>
              <input type="number" step="0.1" value={pipelineConfig.lowHz}
                onChange={e => setPipelineConfig(p => ({ ...p, lowHz: parseFloat(e.target.value) }))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Bandpass high (Hz)</span>
              <input type="number" step="0.1" value={pipelineConfig.highHz}
                onChange={e => setPipelineConfig(p => ({ ...p, highHz: parseFloat(e.target.value) }))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Notch filter (Hz)</span>
              <input type="number" step="1" value={pipelineConfig.notchHz}
                onChange={e => setPipelineConfig(p => ({ ...p, notchHz: parseFloat(e.target.value) }))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Window size (sec)</span>
              <input type="number" step="0.1" value={pipelineConfig.windowSec}
                onChange={e => setPipelineConfig(p => ({ ...p, windowSec: parseFloat(e.target.value) }))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Overlap ratio (0–0.9)</span>
              <input type="number" step="0.05" min="0" max="0.95" value={pipelineConfig.overlapRatio}
                onChange={e => setPipelineConfig(p => ({ ...p, overlapRatio: parseFloat(e.target.value) }))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500" />
            </label>
          </div>

          {processError && (
            <p className="text-red-400 text-sm mb-4">{processError}</p>
          )}

          <button
            onClick={runPipeline}
            disabled={processing}
            className="bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg transition-colors"
          >
            {processing ? 'Processing…' : 'Process Signal'}
          </button>
        </div>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Main labeling interface
  // -------------------------------------------------------------------------

  return (
    <div className="max-w-4xl">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-100 mb-1">Label</h1>
          <p className="text-gray-500 text-sm">
            {currentSession.file_path?.split(/[\\/]/).pop()} —{' '}
            {currentSession.n_channels} ch, {currentSession.sample_rate} Hz
          </p>
        </div>

        {/* Run selector (if multiple runs exist) */}
        {runs.length > 1 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs text-gray-500">Pipeline run</span>
            <select
              value={activeRun?.run_id || ''}
              onChange={e => {
                const run = runs.find(r => r.run_id === e.target.value)
                if (run) { setActiveRun(run); setWindowIndex(0); setLabels({}) }
              }}
              className="bg-gray-800 border border-gray-700 text-sm text-gray-200 rounded px-2 py-1 focus:outline-none"
            >
              {runs.map(r => (
                <option key={r.run_id} value={r.run_id}>
                  {r.n_windows} windows — {r.created_at?.slice(0, 10)}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Progress bar                                                      */}
      {/* ---------------------------------------------------------------- */}
      <div className="mb-6">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>{labeledCount} of {totalWindows} windows labeled</span>
          <span>{progressPct}%</span>
        </div>
        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-purple-500 rounded-full transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Window navigation                                                 */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setWindowIndex(i => Math.max(0, i - 1))}
          disabled={windowIndex === 0}
          className="text-sm px-4 py-1.5 border border-gray-700 rounded-lg text-gray-300 hover:border-gray-500 disabled:opacity-30 transition-colors"
        >
          ← Previous
        </button>

        <div className="text-center">
          <span className="text-gray-200 font-mono text-sm">
            Window {windowIndex + 1} / {totalWindows}
          </span>
          {/* Show the label already assigned to this window, if any */}
          {labels[windowIndex] && (
            <span className="ml-3 text-xs bg-purple-700 text-white px-2 py-0.5 rounded-full">
              {labels[windowIndex]}
            </span>
          )}
          {windowData && (
            <span className="ml-2 text-xs text-gray-600">
              ~{windowData.start_time_sec}s
            </span>
          )}
        </div>

        <button
          onClick={() => setWindowIndex(i => Math.min(totalWindows - 1, i + 1))}
          disabled={windowIndex === totalWindows - 1}
          className="text-sm px-4 py-1.5 border border-gray-700 rounded-lg text-gray-300 hover:border-gray-500 disabled:opacity-30 transition-colors"
        >
          Next →
        </button>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Waveform chart for current window                                 */}
      {/* ---------------------------------------------------------------- */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-6">
        {windowLoading ? (
          <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
            Loading window…
          </div>
        ) : windowData ? (
          <WaveformChart
            data={windowData.points}
            channelNames={windowData.channel_names}
            height={220}
          />
        ) : (
          <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
            No data loaded.
          </div>
        )}
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Label assignment buttons                                          */}
      {/* ---------------------------------------------------------------- */}
      <div className="mb-6">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
          Assign label to window {windowIndex + 1}
        </p>
        <div className="flex flex-wrap gap-2 mb-3">
          {PRESET_LABELS.map(label => (
            <button
              key={label}
              onClick={() => assignLabel(label)}
              className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                labels[windowIndex] === label
                  ? 'bg-purple-600 border-purple-500 text-white'
                  : 'bg-transparent border-gray-700 text-gray-300 hover:border-purple-600 hover:text-purple-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Custom label input */}
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Custom label…"
            value={customLabel}
            onChange={e => setCustomLabel(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && customLabel.trim()) {
                assignLabel(customLabel.trim())
                setCustomLabel('')
              }
            }}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-purple-500 w-44"
          />
          <button
            onClick={() => {
              if (customLabel.trim()) {
                assignLabel(customLabel.trim())
                setCustomLabel('')
              }
            }}
            disabled={!customLabel.trim()}
            className="text-sm px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-gray-300 hover:border-gray-500 disabled:opacity-40 transition-colors"
          >
            Assign
          </button>
        </div>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Label summary + save                                              */}
      {/* ---------------------------------------------------------------- */}
      <div className="border-t border-gray-800 pt-6 flex items-center justify-between">
        <div>
          {/* Show a compact count per label class */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(
              Object.values(labels).reduce((acc, l) => {
                acc[l] = (acc[l] || 0) + 1; return acc
              }, {})
            ).map(([label, count]) => (
              <span key={label} className="text-xs bg-gray-800 border border-gray-700 text-gray-400 px-2 py-0.5 rounded-full">
                {count}× {label}
              </span>
            ))}
          </div>
          {saveResult && (
            <p className={`text-sm mt-2 ${saveResult.success ? 'text-green-400' : 'text-red-400'}`}>
              {saveResult.message}
            </p>
          )}
        </div>

        <button
          onClick={saveLabels}
          disabled={saving || labeledCount === 0}
          className="bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white text-sm font-medium px-6 py-2 rounded-lg transition-colors"
        >
          {saving ? 'Saving…' : `Save ${labeledCount} Label${labeledCount !== 1 ? 's' : ''}`}
        </button>
      </div>
    </div>
  )
}
