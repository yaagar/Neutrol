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
import ERPChart from '../components/ERPChart'
import ElectrodeMap from '../components/ElectrodeMap'

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
    label: 'Highpass Filter',
    config: { step: 'highpass', cutoff_hz: 1.0, filter_type: 'butterworth', order: 4 },
    fields: [
      { key: 'cutoff_hz',   label: 'Cutoff (Hz)', type: 'number', min: 0.01, step: 0.1 },
      { key: 'filter_type', label: 'Type',         type: 'select', options: ['butterworth', 'chebyshev1', 'chebyshev2', 'bessel'] },
      { key: 'order',       label: 'Order',         type: 'number', min: 1,    step: 1 },
    ],
  },
  {
    label: 'Lowpass Filter',
    config: { step: 'lowpass', cutoff_hz: 40.0, filter_type: 'butterworth', order: 4 },
    fields: [
      { key: 'cutoff_hz',   label: 'Cutoff (Hz)', type: 'number', min: 0.1, step: 1 },
      { key: 'filter_type', label: 'Type',         type: 'select', options: ['butterworth', 'chebyshev1', 'chebyshev2', 'bessel'] },
      { key: 'order',       label: 'Order',         type: 'number', min: 1,   step: 1 },
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
    config: { step: 'reref', method: 'average', ref_channel: null },
    fields: [
      { key: 'method', label: 'Method', type: 'select', options: ['average', 'linked_mastoid', 'electrode'] },
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
  {
    label: 'Resample',
    config: { step: 'resample', target_rate_hz: 256 },
    fields: [
      { key: 'target_rate_hz', label: 'Target (Hz)', type: 'number', min: 1, step: 1 },
    ],
  },
  {
    label: 'Custom Step',
    // code field handled specially in render (textarea, not a regular field)
    config: {
      step: 'custom',
      name: 'Custom',
      code: 'def process(signal, sample_rate):\n    # signal: (n_channels, n_samples)\n    # Return same shape\n    return signal\n',
    },
    fields: [
      { key: 'name', label: 'Name', type: 'text' },
    ],
  },
]

const ANALYSIS_TABS = [
  { id: 'time',        label: 'Time' },
  { id: 'frequency',   label: 'Frequency' },
  { id: 'bandpower',   label: 'Band Power' },
  { id: 'spectrogram', label: 'Spectrogram' },
  { id: 'erp',         label: 'ERP' },
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

  // Signal quality stats (from Analyze button on Time tab)
  const [statsData, setStatsData]       = useState(null)
  const [statsLoading, setStatsLoading] = useState(false)
  const [statsOpen, setStatsOpen]       = useState(false)  // collapsed by default

  // ERP data (from POST /api/analysis/erp — requires labeled windows)
  const [erpData, setErpData]       = useState(null)
  const [erpLoading, setErpLoading] = useState(false)
  const [erpError, setErpError]     = useState(null)

  // Dashboard view mode — 'tabs' shows one chart at a time, 'dashboard' shows 2×2 grid
  const [viewMode, setViewMode]     = useState('tabs')
  // In dashboard mode, clicking ⤢ on a chart sets expandedChart to show it full-width
  const [expandedChart, setExpandedChart] = useState(null)  // null | 'time' | 'frequency' | 'bandpower' | 'spectrogram'
  // Per-chart loading flags for dashboard mode so each tile shows its own spinner
  const [dashLoading, setDashLoading] = useState({ time: false, frequency: false, bandpower: false, spectrogram: false })

  // Single-channel focus mode on the Time tab (null = all channels, index = one channel)
  const [waveformFocusChannel, setWaveformFocusChannel] = useState(null)

  // Split view mode — Before (raw) and After (preprocessed) side by side for the active tab
  const [splitData, setSplitData]     = useState({ raw: null, processed: null })
  const [splitLoading, setSplitLoading] = useState(false)

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

  // Fetch signal quality stats for the Analyze button (Time tab).
  // Computes RMS, peak-to-peak, std, kurtosis, max_gradient, ZCR per channel.
  async function fetchStats() {
    if (!currentSession) return
    setStatsLoading(true)
    try {
      const body = {
        session_id: currentSession.session_id,
        run_id: showRaw ? null : (activeRunId || null),
      }
      const res  = await fetch(`${API}/api/analysis/stats`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const json = await res.json()
      if (json.success) {
        setStatsData(json.data)
        setStatsOpen(true)  // auto-expand when data arrives
      } else {
        setAnalysisError(json.error)
      }
    } catch (err) {
      setAnalysisError(`Stats error: ${err.message}`)
    } finally {
      setStatsLoading(false)
    }
  }

  // Fetch ERP data for the ERP tab.
  // Requires a pipeline run_id AND labeled windows (from the Label page).
  async function fetchERP() {
    if (!currentSession || !activeRunId) {
      setErpError('Select a pipeline run first, then label windows on the Label page.')
      return
    }
    setErpLoading(true)
    setErpError(null)
    try {
      const res  = await fetch(`${API}/api/analysis/erp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSession.session_id, run_id: activeRunId }),
      })
      const json = await res.json()
      if (json.success) setErpData(json.data)
      else setErpError(json.error || 'ERP computation failed.')
    } catch (err) {
      setErpError('Failed to compute ERP. Is the backend running?')
    } finally {
      setErpLoading(false)
    }
  }

  // -------------------------------------------------------------------------
  // Dashboard mode: fetch all 4 charts in parallel with Promise.all
  //
  // Each chart fetch is a standalone async operation with its own per-tile
  // loading flag. We fire them simultaneously so the dashboard fills in as
  // each result arrives rather than waiting for all four sequentially.
  // -------------------------------------------------------------------------
  async function fetchAllDashboard() {
    if (!currentSession) return

    const body = {
      psd:  { session_id: currentSession.session_id, run_id: showRaw ? null : (activeRunId || null) },
      band: { session_id: currentSession.session_id, run_id: showRaw ? null : (activeRunId || null) },
      spec: { session_id: currentSession.session_id, run_id: showRaw ? null : (activeRunId || null), channel_index: selectedChannel },
    }

    setDashLoading({ time: !waveformData, frequency: true, bandpower: true, spectrogram: true })
    setAnalysisError(null)

    await Promise.all([
      // Time waveform — only reload if not yet loaded (preview is raw-only)
      waveformData ? Promise.resolve() : loadWaveform().then(() => setDashLoading(p => ({ ...p, time: false }))),

      fetch(`${API}/api/analysis/psd`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body.psd) })
        .then(r => r.json())
        .then(j => { if (j.success) setPsdData(j.data); else setAnalysisError(j.error) })
        .catch(e => setAnalysisError(e.message))
        .finally(() => setDashLoading(p => ({ ...p, frequency: false }))),

      fetch(`${API}/api/analysis/bandpower`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body.band) })
        .then(r => r.json())
        .then(j => { if (j.success) setBandData(j.data); else setAnalysisError(j.error) })
        .catch(e => setAnalysisError(e.message))
        .finally(() => setDashLoading(p => ({ ...p, bandpower: false }))),

      fetch(`${API}/api/analysis/spectrogram`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body.spec) })
        .then(r => r.json())
        .then(j => { if (j.success) setSpectrogramData(j.data); else setAnalysisError(j.error) })
        .catch(e => setAnalysisError(e.message))
        .finally(() => setDashLoading(p => ({ ...p, spectrogram: false }))),
    ])
  }

  // Fetch analysis data when tab changes (auto-load on first switch)
  useEffect(() => {
    if (!currentSession) return
    if (viewMode === 'dashboard') {
      fetchAllDashboard()
      return
    }
    if (activeTab === 'frequency' && !psdData) fetchPsd()
    if (activeTab === 'bandpower' && !bandData) fetchBandPower()
    if (activeTab === 'spectrogram' && !spectrogramData) fetchSpectrogram()
    if (activeTab === 'erp' && !erpData) fetchERP()
  }, [activeTab, currentSession?.session_id, viewMode])

  // Re-fetch when before/after or channel selection changes
  useEffect(() => {
    if (!currentSession) return
    if (viewMode === 'dashboard') { fetchAllDashboard(); return }
    if (activeTab === 'frequency') fetchPsd()
    if (activeTab === 'bandpower') fetchBandPower()
    if (activeTab === 'spectrogram') fetchSpectrogram()
  }, [showRaw, activeRunId])

  useEffect(() => {
    if (!currentSession) return
    if (viewMode === 'dashboard') { fetchAllDashboard(); return }
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

  // Set the reference electrode for a reref step using the electrode mode
  function setRefChannel(idx, channelName) {
    const updated = [...pipelineSteps]
    updated[idx]  = { ...updated[idx], ref_channel: channelName }
    setPreprocessPageState({ pipelineSteps: updated })
  }

  // -------------------------------------------------------------------------
  // Split view: fetch both raw and processed data for the current tab in parallel
  // -------------------------------------------------------------------------
  async function fetchSplitView() {
    if (!currentSession || !activeRunId) return
    setSplitLoading(true)
    setSplitData({ raw: null, processed: null })

    const rawBody  = { session_id: currentSession.session_id, run_id: null }
    const procBody = { session_id: currentSession.session_id, run_id: activeRunId }

    try {
      if (activeTab === 'time') {
        // Time tab: raw waveform only (no processed waveform endpoint)
        setSplitData({ raw: waveformData, processed: null })
        setSplitLoading(false)
        return
      }

      const endpointMap = {
        frequency:   '/api/analysis/psd',
        bandpower:   '/api/analysis/bandpower',
        spectrogram: '/api/analysis/spectrogram',
      }
      const endpoint = endpointMap[activeTab]
      if (!endpoint) { setSplitLoading(false); return }

      const specBody = { ...rawBody,  channel_index: selectedChannel }
      const specProc = { ...procBody, channel_index: selectedChannel }

      const [rawRes, procRes] = await Promise.all([
        fetch(`${API}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(activeTab === 'spectrogram' ? specBody : rawBody) }),
        fetch(`${API}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(activeTab === 'spectrogram' ? specProc : procBody) }),
      ])
      const [rawJson, procJson] = await Promise.all([rawRes.json(), procRes.json()])
      setSplitData({
        raw:       rawJson.success  ? rawJson.data  : null,
        processed: procJson.success ? procJson.data : null,
      })
    } catch (err) {
      setAnalysisError(`Split view error: ${err.message}`)
    } finally {
      setSplitLoading(false)
    }
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

  // Standard 10-20 names — used to decide whether to show the ElectrodeMap SVG
  // or fall back to the simple checkbox grid when channels are non-standard.
  const STANDARD_10_20 = new Set([
    'Fp1','Fp2','Fz','F3','F4','F7','F8',
    'Cz','C3','C4','T3','T4','T7','T8',
    'Pz','P3','P4','T5','T6','P7','P8',
    'O1','O2','A1','A2','Oz','FCz','AFz',
  ])
  const matchedCount = channelNames.filter(ch => STANDARD_10_20.has(ch)).length
  const useElectrodeMap = matchedCount >= 4

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
                          type={field.type === 'text' ? 'text' : 'number'}
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

                  {/* Custom Step — Python code textarea */}
                  {stepConfig.step === 'custom' && (
                    <div className="mt-1">
                      <p className="text-xs text-gray-400 mb-1 font-mono">
                        Python code — <code className="text-yellow-300">def process(signal, sample_rate):</code>
                      </p>
                      <textarea
                        value={stepConfig.code ?? ''}
                        onChange={e => {
                          const updated = [...pipelineSteps]
                          updated[idx] = { ...updated[idx], code: e.target.value }
                          setPreprocessPageState({ pipelineSteps: updated })
                        }}
                        rows={7}
                        spellCheck={false}
                        className="w-full bg-gray-900 border border-gray-600 text-green-300 text-xs font-mono rounded px-2 py-1.5 focus:outline-none focus:border-yellow-500 resize-y leading-relaxed"
                      />
                      <p className="text-xs text-gray-600 mt-0.5">
                        Available: <code className="text-gray-400">np</code>, <code className="text-gray-400">scipy</code>. Signal shape: <code className="text-gray-400">(n_channels, n_samples)</code>.
                      </p>
                    </div>
                  )}

                  {/* Channel Selector — ElectrodeMap SVG when ≥4 standard 10-20 names, else checkbox grid */}
                  {stepConfig.step === 'channels' && (
                    <div className="mt-1">
                      <p className="text-xs text-gray-400 mb-1">Channels to keep:</p>
                      {useElectrodeMap ? (
                        <ElectrodeMap
                          channelNames={channelNames}
                          mode="select"
                          selectedChannels={stepConfig.channels || []}
                          onToggle={(ch) => toggleChannel(idx, ch)}
                        />
                      ) : (
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
                      )}
                      <p className="text-xs text-gray-500 mt-1">
                        {(stepConfig.channels || []).length}/{channelNames.length} selected
                      </p>
                    </div>
                  )}

                  {/* Re-reference electrode picker — ElectrodeMap in reref mode when method is 'electrode' */}
                  {stepConfig.step === 'reref' && stepConfig.method === 'electrode' && (
                    <div className="mt-2">
                      <p className="text-xs text-gray-400 mb-1">Click electrode to set as reference:</p>
                      {useElectrodeMap ? (
                        <ElectrodeMap
                          channelNames={channelNames}
                          mode="reref"
                          referenceChannel={stepConfig.ref_channel || null}
                          onSetRef={(ch) => setRefChannel(idx, ch)}
                        />
                      ) : (
                        <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                          {channelNames.map(ch => (
                            <button
                              key={ch}
                              onClick={() => setRefChannel(idx, ch)}
                              className={`text-xs px-1.5 py-0.5 rounded border transition ${
                                stepConfig.ref_channel === ch
                                  ? 'bg-yellow-600 border-yellow-500 text-white'
                                  : 'bg-gray-600 border-gray-500 text-gray-300 hover:bg-gray-500'
                              }`}
                            >
                              {ch}
                            </button>
                          ))}
                        </div>
                      )}
                      {stepConfig.ref_channel && (
                        <p className="text-xs text-yellow-400 mt-1">Reference: {stepConfig.ref_channel}</p>
                      )}
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

        {/* ICA warning — ICA has no progress signal, can silently take minutes */}
        {running && pipelineSteps.some(s => s.step === 'ica') && (
          <p className="text-yellow-300 text-xs bg-yellow-900/20 rounded p-2">
            ICA is running — this can take 1–5 minutes for long recordings. Please wait.
          </p>
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

          {/* View mode toggle — Tabs | Dashboard | Split */}
          <div className="flex bg-gray-800 rounded-lg p-0.5 gap-0.5">
            <button
              onClick={() => { setViewMode('tabs'); setExpandedChart(null) }}
              className={`px-3 py-1 rounded text-xs font-medium transition ${
                viewMode === 'tabs' ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-gray-200'
              }`}
              title="One chart at a time"
            >
              ☰ Tabs
            </button>
            <button
              onClick={() => { setViewMode('dashboard'); setExpandedChart(null); fetchAllDashboard() }}
              className={`px-3 py-1 rounded text-xs font-medium transition ${
                viewMode === 'dashboard' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'
              }`}
              title="All four charts simultaneously"
            >
              ⊞ Dashboard
            </button>
            {/* Split view only available when there is a processed run to compare against */}
            {hasRun && activeTab !== 'erp' && (
              <button
                onClick={() => { setViewMode('split'); setExpandedChart(null); fetchSplitView() }}
                className={`px-3 py-1 rounded text-xs font-medium transition ${
                  viewMode === 'split' ? 'bg-green-700 text-white' : 'text-gray-400 hover:text-gray-200'
                }`}
                title="Before vs After side by side"
              >
                ⇔ Split
              </button>
            )}
          </div>

          {/* Tab buttons — only shown in tabs mode */}
          {viewMode === 'tabs' && (
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
          )}

          {/* Before/After toggle — hidden in split mode (it's already implicit in both columns) */}
          {hasRun && viewMode !== 'split' && (viewMode === 'dashboard' || activeTab !== 'time') && (
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

          {/* Channel selector — in tabs mode for frequency/spectrogram; in dashboard always */}
          {(viewMode === 'dashboard' || activeTab === 'frequency' || activeTab === 'spectrogram') && (
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

          {/* Refresh/Analyze button */}
          <button
            onClick={() => {
              if (viewMode === 'dashboard') { fetchAllDashboard(); return }
              if (viewMode === 'split')     { fetchSplitView(); return }
              if (activeTab === 'frequency')   fetchPsd()
              if (activeTab === 'bandpower')   fetchBandPower()
              if (activeTab === 'spectrogram') fetchSpectrogram()
              if (activeTab === 'erp')         fetchERP()
              // Time tab: reload waveform AND compute signal quality stats
              if (activeTab === 'time') { loadWaveform(); fetchStats() }
            }}
            disabled={analysisLoading || statsLoading || splitLoading || Object.values(dashLoading).some(Boolean)}
            className="ml-auto text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded px-3 py-1"
          >
            {(analysisLoading || statsLoading || splitLoading || Object.values(dashLoading).some(Boolean)) ? 'Loading…' : 'Analyze'}
          </button>
        </div>

        {/* Error banner */}
        {analysisError && (
          <div className="mb-2 text-red-400 text-xs bg-red-900/20 rounded p-2">
            {analysisError}
          </div>
        )}

        {/* Chart area — tabs mode */}
        {viewMode === 'tabs' && (
          <div className="flex-1 bg-gray-800 rounded-lg p-4 overflow-auto">
            {activeTab === 'time' && (
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <p className="text-xs text-gray-400">
                    Raw signal — {channelNames.length} channels · {currentSession.sample_rate} Hz
                  </p>
                  {/* Single-channel focus dropdown — lets users zoom into one channel at full height */}
                  {waveformData?.channel_names?.length > 1 && (
                    <div className="flex items-center gap-1 ml-auto">
                      <label className="text-xs text-gray-500">Focus:</label>
                      <select
                        value={waveformFocusChannel ?? ''}
                        onChange={e => setWaveformFocusChannel(e.target.value === '' ? null : parseInt(e.target.value))}
                        className="bg-gray-700 text-gray-200 text-xs rounded px-2 py-0.5"
                      >
                        <option value="">All Channels</option>
                        {waveformData.channel_names.map((ch, i) => (
                          <option key={i} value={i}>{ch}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
                {waveformData ? (
                  <WaveformChart
                    data={waveformData.points}
                    channelNames={
                      waveformFocusChannel !== null
                        ? [waveformData.channel_names[waveformFocusChannel]]
                        : waveformData.channel_names
                    }
                    height={waveformFocusChannel !== null ? 300 : undefined}
                  />
                ) : (
                  <p className="text-gray-500 text-sm text-center mt-8">Loading waveform…</p>
                )}

                {/* Signal Quality Stats card — shown after clicking Analyze */}
                {(statsData || statsLoading) && (
                  <div className="mt-4 border border-gray-700 rounded-lg overflow-hidden">
                    {/* Collapsible header */}
                    <button
                      onClick={() => setStatsOpen(o => !o)}
                      className="w-full flex items-center justify-between px-4 py-2 bg-gray-700 hover:bg-gray-600 text-xs font-semibold text-gray-200 transition"
                    >
                      <span>Signal Quality Metrics {statsLoading && <span className="text-gray-400 font-normal ml-1">Computing…</span>}</span>
                      <span className="text-gray-400">{statsOpen ? '▲' : '▼'}</span>
                    </button>

                    {statsOpen && statsData && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="bg-gray-750 text-gray-400 border-b border-gray-700">
                              <th className="text-left px-3 py-2">Channel</th>
                              <th className="text-right px-3 py-2">RMS</th>
                              <th className="text-right px-3 py-2">Peak-Peak</th>
                              <th className="text-right px-3 py-2">Std Dev</th>
                              <th className="text-right px-3 py-2" title="Pearson kurtosis (normal=3.0). >5 suggests spike artifacts.">Kurtosis</th>
                              <th className="text-right px-3 py-2" title="Max sample-to-sample change — detects electrode pop-offs.">Max Grad</th>
                              <th className="text-right px-3 py-2" title="Sign changes per second — frequency proxy.">ZCR/s</th>
                              <th className="text-center px-3 py-2">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {statsData.channels.map((ch, i) => (
                              <tr
                                key={i}
                                className={`border-b border-gray-800 ${ch.artifact_warning ? 'bg-red-950/30' : ''}`}
                              >
                                <td className="px-3 py-1.5 font-mono text-gray-200">{ch.channel}</td>
                                <td className="px-3 py-1.5 text-right text-gray-300">{ch.rms.toFixed(3)}</td>
                                <td className="px-3 py-1.5 text-right text-gray-300">{ch.peak_to_peak.toFixed(2)}</td>
                                <td className="px-3 py-1.5 text-right text-gray-300">{ch.std.toFixed(3)}</td>
                                <td className={`px-3 py-1.5 text-right font-mono ${ch.kurtosis > 5 ? 'text-red-400' : 'text-gray-300'}`}>
                                  {ch.kurtosis.toFixed(2)}
                                </td>
                                <td className="px-3 py-1.5 text-right text-gray-300">{ch.max_gradient.toFixed(3)}</td>
                                <td className="px-3 py-1.5 text-right text-gray-300">{ch.zero_crossing_rate.toFixed(1)}</td>
                                <td className="px-3 py-1.5 text-center">
                                  {ch.artifact_warning
                                    ? <span className="text-red-400" title="Possible artifact — high kurtosis or large amplitude">⚠ Artifact?</span>
                                    : <span className="text-green-400">✓</span>
                                  }
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-xs text-gray-600 px-3 py-2">
                          Kurtosis &gt; 5 or peak-to-peak &gt; 200 µV flags a channel for artifact review.
                          Use Epoch Rejection or Channel Selector steps to exclude problematic channels.
                        </p>
                      </div>
                    )}
                  </div>
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
                  <FrequencyChart data={psdData} title={showRaw ? 'Raw Signal — PSD' : 'Preprocessed Signal — PSD'} />
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
            {activeTab === 'erp' && (
              <div>
                <p className="text-xs text-gray-400 mb-3">
                  Event-Related Potential — average signal per class label, time-locked to window onset.
                  Requires labeled windows (Label page). Each line is one class; noise cancels across trials.
                </p>
                <ERPChart data={erpData} loading={erpLoading} error={erpError} />
              </div>
            )}
          </div>
        )}

        {/* Chart area — dashboard mode (2×2 grid, all charts simultaneously) */}
        {viewMode === 'dashboard' && (
          <div className="flex-1 overflow-auto">
            {/* When a chart is expanded in dashboard mode, show it full-width */}
            {expandedChart ? (
              <div className="bg-gray-800 rounded-lg p-4 h-full flex flex-col">
                <div className="flex items-center gap-2 mb-3">
                  <button
                    onClick={() => setExpandedChart(null)}
                    className="text-xs text-gray-400 hover:text-gray-200 flex items-center gap-1 transition-colors"
                  >
                    ← Back to Dashboard
                  </button>
                  <span className="text-xs text-gray-500 ml-2">
                    {ANALYSIS_TABS.find(t => t.id === expandedChart)?.label}
                  </span>
                </div>
                <div className="flex-1">
                  {expandedChart === 'time' && (
                    waveformData
                      ? <WaveformChart data={waveformData.points} channelNames={waveformData.channel_names} />
                      : <p className="text-gray-500 text-sm text-center mt-8">Loading waveform…</p>
                  )}
                  {expandedChart === 'frequency' && (
                    dashLoading.frequency
                      ? <p className="text-gray-500 text-sm text-center mt-8">Computing PSD…</p>
                      : <FrequencyChart data={psdData} title={showRaw ? 'Raw — PSD' : 'Preprocessed — PSD'} />
                  )}
                  {expandedChart === 'bandpower' && (
                    dashLoading.bandpower
                      ? <p className="text-gray-500 text-sm text-center mt-8">Computing band power…</p>
                      : <BandPowerChart data={bandData} />
                  )}
                  {expandedChart === 'spectrogram' && (
                    dashLoading.spectrogram
                      ? <p className="text-gray-500 text-sm text-center mt-8">Computing spectrogram…</p>
                      : <SpectrogramChart data={spectrogramData} />
                  )}
                </div>
              </div>
            ) : (
              /* 2×2 grid — all four charts simultaneously */
              <div className="grid grid-cols-2 grid-rows-2 gap-2 h-full min-h-[480px]">
                {[
                  {
                    id: 'time', label: 'Waveform',
                    loading: dashLoading.time,
                    content: waveformData
                      ? <WaveformChart data={waveformData.points} channelNames={waveformData.channel_names} height={180} />
                      : null,
                    placeholder: 'Loading waveform…',
                  },
                  {
                    id: 'frequency', label: 'Frequency (PSD)',
                    loading: dashLoading.frequency,
                    content: psdData
                      ? <FrequencyChart data={psdData} title={showRaw ? 'Raw — PSD' : 'Preprocessed — PSD'} height={180} />
                      : null,
                    placeholder: 'Click Analyze to load…',
                  },
                  {
                    id: 'bandpower', label: 'Band Power',
                    loading: dashLoading.bandpower,
                    content: bandData
                      ? <BandPowerChart data={bandData} height={180} />
                      : null,
                    placeholder: 'Click Analyze to load…',
                  },
                  {
                    id: 'spectrogram', label: 'Spectrogram',
                    loading: dashLoading.spectrogram,
                    content: spectrogramData
                      ? <SpectrogramChart data={spectrogramData} height={180} />
                      : null,
                    placeholder: 'Click Analyze to load…',
                  },
                ].map(tile => (
                  <div key={tile.id} className="bg-gray-800 rounded-lg p-3 flex flex-col overflow-hidden">
                    {/* Tile header with expand button */}
                    <div className="flex items-center justify-between mb-1.5 flex-shrink-0">
                      <span className="text-xs font-medium text-gray-400">{tile.label}</span>
                      <button
                        onClick={() => setExpandedChart(tile.id)}
                        title="Expand to full width"
                        className="text-gray-600 hover:text-gray-300 text-xs leading-none transition-colors px-1"
                      >
                        ⤢
                      </button>
                    </div>
                    {/* Chart */}
                    <div className="flex-1 min-h-0">
                      {tile.loading ? (
                        <p className="text-gray-500 text-xs text-center mt-6">Loading…</p>
                      ) : tile.content ? (
                        tile.content
                      ) : (
                        <p className="text-gray-600 text-xs text-center mt-6">{tile.placeholder}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Chart area — split view mode (Before left, After right for the active tab) */}
        {viewMode === 'split' && (
          <div className="flex-1 overflow-auto">
            <div className="grid grid-cols-2 gap-3 h-full min-h-[300px]">
              {/* LEFT — Before (raw) */}
              <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
                <p className="text-xs font-semibold text-green-400 mb-2 flex-shrink-0">Before (Raw)</p>
                {splitLoading ? (
                  <p className="text-gray-500 text-sm text-center mt-8">Loading…</p>
                ) : (
                  <>
                    {activeTab === 'time' && (
                      waveformData
                        ? <WaveformChart data={waveformData.points} channelNames={waveformData.channel_names} />
                        : <p className="text-gray-500 text-sm text-center mt-8">Load waveform first</p>
                    )}
                    {activeTab === 'frequency' && (
                      splitData.raw
                        ? <FrequencyChart data={splitData.raw} title="Raw — PSD" />
                        : <p className="text-gray-500 text-sm text-center mt-8">Click ⇔ Split to load</p>
                    )}
                    {activeTab === 'bandpower' && (
                      splitData.raw
                        ? <BandPowerChart data={splitData.raw} />
                        : <p className="text-gray-500 text-sm text-center mt-8">Click ⇔ Split to load</p>
                    )}
                    {activeTab === 'spectrogram' && (
                      splitData.raw
                        ? <SpectrogramChart data={splitData.raw} />
                        : <p className="text-gray-500 text-sm text-center mt-8">Click ⇔ Split to load</p>
                    )}
                  </>
                )}
              </div>

              {/* RIGHT — After (processed) */}
              <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
                <p className="text-xs font-semibold text-blue-400 mb-2 flex-shrink-0">
                  After (Preprocessed{activeRunId ? ` — ${runs.find(r => r.run_id === activeRunId)?.name || activeRunId.slice(0, 8)}` : ''})
                </p>
                {splitLoading ? (
                  <p className="text-gray-500 text-sm text-center mt-8">Loading…</p>
                ) : !activeRunId ? (
                  <p className="text-yellow-400 text-xs text-center mt-8">
                    No pipeline run selected — run a pipeline first.
                  </p>
                ) : (
                  <>
                    {activeTab === 'time' && (
                      <p className="text-gray-500 text-xs text-center mt-8">
                        Processed waveform preview not available.<br />
                        Switch to Frequency or Band Power tab for Before/After comparison.
                      </p>
                    )}
                    {activeTab === 'frequency' && (
                      splitData.processed
                        ? <FrequencyChart data={splitData.processed} title="Preprocessed — PSD" />
                        : <p className="text-gray-500 text-sm text-center mt-8">No data</p>
                    )}
                    {activeTab === 'bandpower' && (
                      splitData.processed
                        ? <BandPowerChart data={splitData.processed} />
                        : <p className="text-gray-500 text-sm text-center mt-8">No data</p>
                    )}
                    {activeTab === 'spectrogram' && (
                      splitData.processed
                        ? <SpectrogramChart data={splitData.processed} />
                        : <p className="text-gray-500 text-sm text-center mt-8">No data</p>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )}

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
