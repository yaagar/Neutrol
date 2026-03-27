// PreprocessPage.jsx — Preprocessing pipeline builder + Signal analysis tabs
//
// This page has two panels:
//   Left:  Pipeline builder — add/remove/configure steps, run pipeline
//   Right: Signal analysis tabs — Time, Frequency, Band Power, Spectrogram
//          with a Before/After toggle comparing raw vs preprocessed signal
//
// Pipeline step state (pipelineSteps, activeTab, activeRunId) is persisted
// in Zustand so navigating away and back doesn't reset the pipeline config.

import { useState, useEffect, useCallback } from 'react'
import { useSession } from '../hooks/useSession'
import useAppStore from '../store/appStore'
import WaveformChart from '../components/WaveformChart'
import FrequencyChart from '../components/FrequencyChart'
import BandPowerChart from '../components/BandPowerChart'
import SpectrogramChart from '../components/SpectrogramChart'

import { API_BASE_URL } from '../config'
const API = API_BASE_URL

// Available steps the user can add to the pipeline.
// Each entry defines the label shown in the dropdown and the default config
// used when the step is first added.
const STEP_PRESETS = [
  {
    label: 'Truncate Signal',
    config: { step: 'truncate', start_sec: 0.0, end_sec: null },
    fields: [
      { key: 'start_sec', label: 'Start (s)', type: 'number', min: 0,   step: 1 },
      { key: 'end_sec',   label: 'End (s)',   type: 'number', min: 0.1, step: 1 },
    ],
  },
  {
    label: 'Bandpass Filter',
    config: { step: 'bandpass', low_hz: 1.0, high_hz: 40.0 },
    fields: [
      { key: 'low_hz',  label: 'Low (Hz)',  type: 'number', min: 0.1, step: 0.5 },
      { key: 'high_hz', label: 'High (Hz)', type: 'number', min: 1,   step: 1   },
    ],
  },
  {
    label: 'Notch Filter',
    config: { step: 'notch', freq_hz: 60.0, quality_factor: 30.0 },
    fields: [
      { key: 'freq_hz',        label: 'Freq (Hz)', type: 'number', min: 1, step: 1 },
      { key: 'quality_factor', label: 'Q factor',  type: 'number', min: 1, step: 1 },
    ],
  },
  {
    label: 'Channel Selector',
    // channels is a list — rendered as checkboxes using session.channel_names
    config: { step: 'channels', channels: [] },
    fields: [],  // no regular fields — handled specially in render
  },
  {
    label: 'Re-reference',
    config: { step: 'reref', method: 'average' },
    fields: [
      { key: 'method', label: 'Method', type: 'select', options: ['average', 'linked_mastoid'] },
    ],
  },
  {
    label: 'Normalize',
    config: { step: 'normalize', method: 'zscore' },
    fields: [
      { key: 'method', label: 'Method', type: 'select', options: ['zscore', 'minmax'] },
    ],
  },
  {
    label: 'Sliding Window',
    config: { step: 'window', window_sec: 1.0, overlap_ratio: 0.5 },
    fields: [
      { key: 'window_sec',   label: 'Window (s)', type: 'number', min: 0.1, step: 0.1 },
      { key: 'overlap_ratio', label: 'Overlap',   type: 'number', min: 0,   max: 0.99, step: 0.1 },
    ],
  },
  {
    label: 'Epoch Rejection',
    config: { step: 'reject', threshold_uv: 100.0 },
    fields: [
      { key: 'threshold_uv', label: 'Threshold (µV)', type: 'number', min: 1, step: 10 },
    ],
  },
]

const ANALYSIS_TABS = [
  { id: 'time',        label: 'Time' },
  { id: 'frequency',   label: 'Frequency' },
  { id: 'bandpower',   label: 'Band Power' },
  { id: 'spectrogram', label: 'Spectrogram' },
]

export default function PreprocessPage() {
  const { currentSession } = useSession()

  // --- Zustand persistent state ---
  const preprocessPageState    = useAppStore(s => s.preprocessPageState)
  const setPreprocessPageState = useAppStore(s => s.setPreprocessPageState)
  const { pipelineSteps, activeTab, activeRunId, runName } = preprocessPageState

  // --- Local UI state (not worth persisting) ---
  const [selectedPreset, setSelectedPreset] = useState('')
  const [running, setRunning]               = useState(false)
  const [runResult, setRunResult]           = useState(null)
  const [runError, setRunError]             = useState(null)
  const [runs, setRuns]                     = useState([])

  // Before/after toggle and channel selector (local)
  const [showRaw, setShowRaw]               = useState(true)
  const [selectedChannel, setSelectedChannel] = useState(0)

  // Analysis data
  const [waveformData, setWaveformData]         = useState(null)
  const [psdData, setPsdData]                   = useState(null)
  const [bandData, setBandData]                 = useState(null)
  const [spectrogramData, setSpectrogramData]   = useState(null)
  const [analysisLoading, setAnalysisLoading]   = useState(false)
  const [analysisError, setAnalysisError]       = useState(null)

  // -------------------------------------------------------------------------
  // Load existing runs when session changes
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!currentSession) {
      setRuns([])
      setRunResult(null)
      setPsdData(null)
      setBandData(null)
      setSpectrogramData(null)
      setWaveformData(null)
      return
    }
    fetchRuns()
    loadWaveform()
  }, [currentSession?.session_id])

  async function fetchRuns() {
    if (!currentSession) return
    try {
      const res  = await fetch(`${API}/api/pipeline/runs`)
      const json = await res.json()
      if (json.success) {
        const sessionRuns = json.data.filter(r => r.session_id === currentSession.session_id)
        setRuns(sessionRuns)
        // Auto-select most recent run if none selected
        if (sessionRuns.length > 0 && !activeRunId) {
          setPreprocessPageState({ activeRunId: sessionRuns[sessionRuns.length - 1].run_id })
        }
      }
    } catch (err) {
      console.error('Failed to fetch runs:', err)
    }
  }

  async function loadWaveform() {
    if (!currentSession) return
    try {
      const res  = await fetch(`${API}/api/data/${currentSession.session_id}/preview?max_samples=512`)
      const json = await res.json()
      if (json.success) setWaveformData(json.data)
    } catch (err) {
      console.error('Failed to load waveform:', err)
    }
  }

  // -------------------------------------------------------------------------
  // Run pipeline
  // -------------------------------------------------------------------------
  async function runPipeline() {
    if (!currentSession) return
    setRunning(true)
    setRunError(null)
    setRunResult(null)

    try {
      const res  = await fetch(`${API}/api/pipeline/run`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:      currentSession.session_id,
          pipeline_config: { steps: pipelineSteps },
          name:            runName,
        }),
      })
      const json = await res.json()
      if (json.success) {
        setRunResult(json.data)
        setPreprocessPageState({ activeRunId: json.data.run_id })
        await fetchRuns()
        // Clear analysis cache so After view reflects new run
        setPsdData(null)
        setBandData(null)
        setSpectrogramData(null)
      } else {
        setRunError(json.error || 'Pipeline run failed.')
      }
    } catch (err) {
      setRunError(`Network error: ${err.message}`)
    } finally {
      setRunning(false)
    }
  }

  // -------------------------------------------------------------------------
  // Analysis fetch functions
  // -------------------------------------------------------------------------
  async function fetchPsd() {
    if (!currentSession) return
    setAnalysisLoading(true)
    setAnalysisError(null)
    try {
      const body = {
        session_id: currentSession.session_id,
        run_id: showRaw ? null : (activeRunId || null),
      }
      const res  = await fetch(`${API}/api/analysis/psd`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const json = await res.json()
      if (json.success) setPsdData(json.data)
      else setAnalysisError(json.error)
    } catch (err) {
      setAnalysisError(`Network error: ${err.message}`)
    } finally {
      setAnalysisLoading(false)
    }
  }

  async function fetchBandPower() {
    if (!currentSession) return
    setAnalysisLoading(true)
    setAnalysisError(null)
    try {
      const body = {
        session_id: currentSession.session_id,
        run_id: showRaw ? null : (activeRunId || null),
      }
      const res  = await fetch(`${API}/api/analysis/bandpower`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const json = await res.json()
      if (json.success) setBandData(json.data)
      else setAnalysisError(json.error)
    } catch (err) {
      setAnalysisError(`Network error: ${err.message}`)
    } finally {
      setAnalysisLoading(false)
    }
  }

  async function fetchSpectrogram() {
    if (!currentSession) return
    setAnalysisLoading(true)
    setAnalysisError(null)
    try {
      const body = {
        session_id:    currentSession.session_id,
        run_id:        showRaw ? null : (activeRunId || null),
        channel_index: selectedChannel,
      }
      const res  = await fetch(`${API}/api/analysis/spectrogram`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const json = await res.json()
      if (json.success) setSpectrogramData(json.data)
      else setAnalysisError(json.error)
    } catch (err) {
      setAnalysisError(`Network error: ${err.message}`)
    } finally {
      setAnalysisLoading(false)
    }
  }

  // Fetch analysis data when tab changes (auto-load on first switch)
  useEffect(() => {
    if (!currentSession) return
    if (activeTab === 'frequency' && !psdData) fetchPsd()
    if (activeTab === 'bandpower' && !bandData) fetchBandPower()
    if (activeTab === 'spectrogram' && !spectrogramData) fetchSpectrogram()
  }, [activeTab, currentSession?.session_id])

  // Re-fetch when before/after or channel selection changes
  useEffect(() => {
    if (!currentSession) return
    if (activeTab === 'frequency') fetchPsd()
    if (activeTab === 'bandpower') fetchBandPower()
    if (activeTab === 'spectrogram') fetchSpectrogram()
  }, [showRaw, activeRunId])

  useEffect(() => {
    if (!currentSession) return
    if (activeTab === 'spectrogram') fetchSpectrogram()
  }, [selectedChannel])

  // -------------------------------------------------------------------------
  // Pipeline step management
  // -------------------------------------------------------------------------
  function addStep() {
    if (!selectedPreset) return
    const preset = STEP_PRESETS.find(p => p.label === selectedPreset)
    if (!preset) return
    const newConfig = { ...preset.config }
    // For Channel Selector: pre-populate with all current session channels
    if (newConfig.step === 'channels' && currentSession) {
      newConfig.channels = [...(currentSession.channel_names || [])]
    }
    setPreprocessPageState({ pipelineSteps: [...pipelineSteps, newConfig] })
    setSelectedPreset('')
  }

  function removeStep(idx) {
    setPreprocessPageState({ pipelineSteps: pipelineSteps.filter((_, i) => i !== idx) })
  }

  function moveStepUp(idx) {
    if (idx === 0) return
    const updated = [...pipelineSteps]
    ;[updated[idx - 1], updated[idx]] = [updated[idx], updated[idx - 1]]
    setPreprocessPageState({ pipelineSteps: updated })
  }

  function moveStepDown(idx) {
    if (idx === pipelineSteps.length - 1) return
    const updated = [...pipelineSteps]
    ;[updated[idx], updated[idx + 1]] = [updated[idx + 1], updated[idx]]
    setPreprocessPageState({ pipelineSteps: updated })
  }

  function duplicateStep(idx) {
    const updated = [...pipelineSteps]
    updated.splice(idx + 1, 0, { ...pipelineSteps[idx] })
    setPreprocessPageState({ pipelineSteps: updated })
  }

  function updateStepField(idx, key, value) {
    const updated = [...pipelineSteps]
    const numVal  = parseFloat(value)
    updated[idx]  = { ...updated[idx], [key]: isNaN(numVal) ? value : numVal }
    setPreprocessPageState({ pipelineSteps: updated })
  }

  // Toggle a channel on/off for ChannelSelector steps
  function toggleChannel(idx, channelName) {
    const updated  = [...pipelineSteps]
    const current  = updated[idx].channels || []
    const newChans = current.includes(channelName)
      ? current.filter(c => c !== channelName)
      : [...current, channelName]
    updated[idx] = { ...updated[idx], channels: newChans }
    setPreprocessPageState({ pipelineSteps: updated })
  }

  function getStepPreset(stepConfig) {
    return STEP_PRESETS.find(p => p.config.step === stepConfig.step)
  }

  // -------------------------------------------------------------------------
  // No session selected
  // -------------------------------------------------------------------------
  if (!currentSession) {
    return (
      <div className="p-6 max-w-lg">
        <h1 className="text-2xl font-bold text-gray-100 mb-2">Preprocess</h1>
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-8 text-center mt-6">
          <p className="text-gray-400 text-sm mb-2">No session loaded.</p>
          <p className="text-gray-600 text-xs">
            Go to{' '}
            <a href="/import" className="text-purple-400 hover:underline">Import</a>{' '}
            and upload an EEG file first.
          </p>
        </div>
      </div>
    )
  }

  const channelNames = currentSession.channel_names || []
  const hasRun       = runs.length > 0

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="flex h-full gap-4 p-4 overflow-hidden">

      {/* ------------------------------------------------------------------ */}
      {/* LEFT PANEL — Pipeline Builder                                       */}
      {/* ------------------------------------------------------------------ */}
      <div className="w-72 flex-shrink-0 flex flex-col gap-3 overflow-y-auto">
        {/* Session info */}
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs font-semibold text-gray-400 uppercase mb-1">Session</p>
          <p className="text-sm text-gray-200 truncate">{currentSession.session_id.slice(0, 8)}…</p>
          <p className="text-xs text-gray-400">
            {currentSession.n_channels} ch · {currentSession.sample_rate} Hz · {currentSession.duration_seconds?.toFixed(1)}s
          </p>
        </div>

        {/* Pipeline steps */}
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-gray-400 uppercase">Pipeline Steps</p>
            {pipelineSteps.length > 0 && (
              <button
                onClick={() => setPreprocessPageState({ pipelineSteps: [] })}
                className="text-xs text-gray-600 hover:text-red-400 transition-colors"
              >
                Clear all
              </button>
            )}
          </div>

          {pipelineSteps.length === 0 && (
            <p className="text-xs text-gray-500 mb-2">No steps. Add steps below.</p>
          )}

          <div className="flex flex-col gap-2 mb-3">
            {pipelineSteps.map((stepConfig, idx) => {
              const preset = getStepPreset(stepConfig)
              return (
                <div key={idx} className="bg-gray-700 rounded p-2">
                  {/* Step header */}
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs font-medium text-blue-300">
                      {preset?.label || stepConfig.step}
                    </span>
                    {/* Step controls: move up, move down, duplicate, remove */}
                    <div className="flex items-center gap-0.5">
                      <button
                        onClick={() => moveStepUp(idx)}
                        disabled={idx === 0}
                        title="Move up"
                        className="text-gray-400 hover:text-gray-200 disabled:opacity-20 text-xs px-1 leading-none"
                      >
                        ↑
                      </button>
                      <button
                        onClick={() => moveStepDown(idx)}
                        disabled={idx === pipelineSteps.length - 1}
                        title="Move down"
                        className="text-gray-400 hover:text-gray-200 disabled:opacity-20 text-xs px-1 leading-none"
                      >
                        ↓
                      </button>
                      <button
                        onClick={() => duplicateStep(idx)}
                        title="Duplicate step"
                        className="text-gray-400 hover:text-blue-300 text-xs px-1 leading-none"
                      >
                        ⧉
                      </button>
                      <button
                        onClick={() => removeStep(idx)}
                        title="Remove step"
                        className="text-red-400 hover:text-red-300 text-xs px-1 leading-none"
                      >
                        ✕
                      </button>
                    </div>
                  </div>

                  {/* Regular fields */}
                  {preset?.fields.map(field => (
                    <div key={field.key} className="flex items-center gap-2 mb-1">
                      <label className="text-xs text-gray-400 w-24 flex-shrink-0">{field.label}</label>
                      {field.type === 'select' ? (
                        <select
                          value={stepConfig[field.key] ?? ''}
                          onChange={e => updateStepField(idx, field.key, e.target.value)}
                          className="flex-1 bg-gray-600 text-gray-200 text-xs rounded px-1 py-0.5"
                        >
                          {field.options.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="number"
                          value={stepConfig[field.key] ?? ''}
                          min={field.min}
                          max={field.max}
                          step={field.step}
                          onChange={e => updateStepField(idx, field.key, e.target.value)}
                          className="flex-1 bg-gray-600 text-gray-200 text-xs rounded px-1 py-0.5 w-0"
                        />
                      )}
                    </div>
                  ))}

                  {/* Channel Selector — visual checkbox list */}
                  {stepConfig.step === 'channels' && (
                    <div className="mt-1">
                      <p className="text-xs text-gray-400 mb-1">Channels to keep:</p>
                      <div className="flex flex-wrap gap-1 max-h-28 overflow-y-auto">
                        {channelNames.map(ch => {
                          const isSelected = (stepConfig.channels || []).includes(ch)
                          return (
                            <button
                              key={ch}
                              onClick={() => toggleChannel(idx, ch)}
                              className={`text-xs px-1.5 py-0.5 rounded border transition ${
                                isSelected
                                  ? 'bg-blue-600 border-blue-500 text-white'
                                  : 'bg-gray-600 border-gray-500 text-gray-300 hover:bg-gray-500'
                              }`}
                            >
                              {ch}
                            </button>
                          )
                        })}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {(stepConfig.channels || []).length}/{channelNames.length} selected
                      </p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Add step */}
          <div className="flex gap-1">
            <select
              value={selectedPreset}
              onChange={e => setSelectedPreset(e.target.value)}
              className="flex-1 bg-gray-600 text-gray-200 text-xs rounded px-1 py-1"
            >
              <option value="">+ Add step…</option>
              {STEP_PRESETS.map(p => (
                <option key={p.label} value={p.label}>{p.label}</option>
              ))}
            </select>
            <button
              onClick={addStep}
              disabled={!selectedPreset}
              className="bg-gray-600 hover:bg-gray-500 disabled:opacity-40 text-gray-200 text-xs rounded px-2"
            >
              Add
            </button>
          </div>
        </div>

        {/* Run name input */}
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs font-semibold text-gray-400 uppercase mb-1">Run Name</p>
          <input
            type="text"
            placeholder="e.g. Bandpass + Window 1s"
            value={runName}
            onChange={e => setPreprocessPageState({ runName: e.target.value })}
            className="w-full bg-gray-700 border border-gray-600 text-gray-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-blue-500 placeholder-gray-500"
          />
        </div>

        {/* Run pipeline button */}
        <button
          onClick={runPipeline}
          disabled={running || pipelineSteps.length === 0}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-sm rounded-lg py-2 transition"
        >
          {running ? 'Running…' : 'Run Pipeline'}
        </button>

        {runError && (
          <p className="text-red-400 text-xs bg-red-900/20 rounded p-2">{runError}</p>
        )}
        {runResult && (
          <p className="text-green-400 text-xs bg-green-900/20 rounded p-2">
            ✓ {runResult.n_windows} windows ({runResult.window_shape?.join('×')})
            {runResult.name && <span className="ml-1 text-green-300">"{runResult.name}"</span>}
          </p>
        )}

        {/* Existing runs */}
        {runs.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-3">
            <p className="text-xs font-semibold text-gray-400 uppercase mb-2">Pipeline Runs</p>
            <div className="flex flex-col gap-1">
              {runs.map(run => (
                <button
                  key={run.run_id}
                  onClick={() => setPreprocessPageState({ activeRunId: run.run_id })}
                  className={`text-left text-xs rounded p-1.5 transition ${
                    activeRunId === run.run_id
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <span className="font-medium">
                    {run.name || <span className="italic text-gray-400">Unnamed</span>}
                  </span>
                  <span className="ml-1 opacity-60">{run.run_id.slice(0, 6)}</span>
                  <br />
                  <span className="opacity-70">{run.n_windows} windows · {run.steps_applied?.join('→')}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* RIGHT PANEL — Signal Analysis Tabs                                  */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Tab bar + controls */}
        <div className="flex items-center gap-3 mb-3 flex-wrap">
          {/* Tab buttons */}
          <div className="flex bg-gray-800 rounded-lg p-0.5 gap-0.5">
            {ANALYSIS_TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setPreprocessPageState({ activeTab: tab.id })}
                className={`px-3 py-1 rounded text-sm font-medium transition ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Before/After toggle (only if there's a pipeline run) */}
          {hasRun && activeTab !== 'time' && (
            <div className="flex bg-gray-800 rounded-lg p-0.5 gap-0.5">
              <button
                onClick={() => setShowRaw(true)}
                className={`px-3 py-1 rounded text-xs transition ${
                  showRaw ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                Before
              </button>
              <button
                onClick={() => setShowRaw(false)}
                className={`px-3 py-1 rounded text-xs transition ${
                  !showRaw ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                After
              </button>
            </div>
          )}

          {/* Channel selector (for Frequency and Spectrogram) */}
          {(activeTab === 'frequency' || activeTab === 'spectrogram') && (
            <div className="flex items-center gap-1">
              <label className="text-xs text-gray-400">Channel:</label>
              <select
                value={selectedChannel}
                onChange={e => setSelectedChannel(parseInt(e.target.value))}
                className="bg-gray-700 text-gray-200 text-xs rounded px-2 py-1"
              >
                {channelNames.map((ch, i) => (
                  <option key={i} value={i}>{ch}</option>
                ))}
              </select>
            </div>
          )}

          {/* Refresh button for analysis tabs */}
          {activeTab !== 'time' && (
            <button
              onClick={() => {
                if (activeTab === 'frequency')   fetchPsd()
                if (activeTab === 'bandpower')   fetchBandPower()
                if (activeTab === 'spectrogram') fetchSpectrogram()
              }}
              disabled={analysisLoading}
              className="ml-auto text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded px-3 py-1"
            >
              {analysisLoading ? 'Loading…' : 'Analyze'}
            </button>
          )}
        </div>

        {/* Error banner */}
        {analysisError && (
          <div className="mb-2 text-red-400 text-xs bg-red-900/20 rounded p-2">
            {analysisError}
          </div>
        )}

        {/* Chart area */}
        <div className="flex-1 bg-gray-800 rounded-lg p-4 overflow-auto">
          {activeTab === 'time' && (
            <div>
              <p className="text-xs text-gray-400 mb-2">
                Raw signal — {channelNames.length} channels · {currentSession.sample_rate} Hz
              </p>
              {waveformData ? (
                <WaveformChart
                  data={waveformData.points}
                  channelNames={waveformData.channel_names}
                />
              ) : (
                <p className="text-gray-500 text-sm text-center mt-8">Loading waveform…</p>
              )}
            </div>
          )}

          {activeTab === 'frequency' && (
            <div>
              <p className="text-xs text-gray-400 mb-3">
                {showRaw ? 'Raw signal PSD' : 'Preprocessed signal PSD'} ·{' '}
                <span className="text-blue-400">{channelNames[selectedChannel]}</span> highlighted
              </p>
              {analysisLoading ? (
                <p className="text-gray-500 text-sm text-center mt-8">Computing PSD…</p>
              ) : (
                <FrequencyChart
                  data={psdData}
                  title={showRaw ? 'Raw Signal — PSD' : 'Preprocessed Signal — PSD'}
                />
              )}
            </div>
          )}

          {activeTab === 'bandpower' && (
            <div>
              <p className="text-xs text-gray-400 mb-3">
                {showRaw ? 'Raw signal band power' : 'Preprocessed signal band power'}
              </p>
              {analysisLoading ? (
                <p className="text-gray-500 text-sm text-center mt-8">Computing band power…</p>
              ) : (
                <BandPowerChart data={bandData} />
              )}
            </div>
          )}

          {activeTab === 'spectrogram' && (
            <div>
              <p className="text-xs text-gray-400 mb-3">
                {showRaw ? 'Raw signal' : 'Preprocessed signal'} ·{' '}
                channel {channelNames[selectedChannel] || selectedChannel}
              </p>
              {analysisLoading ? (
                <p className="text-gray-500 text-sm text-center mt-8">Computing spectrogram…</p>
              ) : (
                <SpectrogramChart data={spectrogramData} />
              )}
            </div>
          )}
        </div>

        {/* Status line */}
        <p className="text-xs text-gray-500 mt-2 text-right">
          {showRaw
            ? 'Showing raw signal'
            : activeRunId
              ? `Showing preprocessed — run ${runs.find(r => r.run_id === activeRunId)?.name || activeRunId.slice(0, 8)}`
              : 'No preprocessed run selected'}
        </p>
      </div>
    </div>
  )
}
