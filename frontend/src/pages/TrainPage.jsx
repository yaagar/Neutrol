// TrainPage.jsx — model configuration, training trigger, live progress, results, and history
//
// Page flow:
//   1. Load label summary from the backend → show class distribution
//   2. User selects a pipeline run to train on
//   3. Choose training mode: from scratch OR fine-tune an existing model
//   4. Optionally enable data augmentation (gaussian noise, time shift, channel dropout)
//   5. User clicks "Start Training" → POST /api/train/start or /api/train/finetune
//   6. WebSocket /api/train/ws/{job_id}/progress → live loss/accuracy charts
//   7. When done → final accuracy + confusion matrix
//   8. Model History section shows all runs for a model name with accuracy chart
//   9. Model Comparison section shows side-by-side metrics for selected models
import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { useSession } from '../hooks/useSession'
import ModelLayerEditor from '../components/ModelLayerEditor'
import useAppStore from '../store/appStore'

import { API_BASE_URL, WS_BASE_URL } from '../config'
const API    = API_BASE_URL
const WS_API = WS_BASE_URL

// ---------------------------------------------------------------------------
// Compute per-class precision/recall/F1 from a confusion matrix
// ---------------------------------------------------------------------------
function computeMetrics(matrix, classNames) {
  return classNames.map((name, i) => {
    const tp  = matrix[i][i]
    const fp  = matrix.reduce((s, row, r) => r !== i ? s + row[i] : s, 0)
    const fn  = matrix[i].reduce((s, v, c) => c !== i ? s + v : s, 0)
    const prec   = tp + fp > 0 ? tp / (tp + fp) : 0
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1     = prec + recall > 0 ? 2 * prec * recall / (prec + recall) : 0
    return { name, precision: prec, recall, f1 }
  })
}

export default function TrainPage() {
  const { currentSession } = useSession()

  // --- Zustand persistent state ---
  const trainPageState    = useAppStore(s => s.trainPageState)
  const setTrainPageState = useAppStore(s => s.setTrainPageState)

  const { config, layers, jobId, status, history, finalResult, selectedRunId, modelName } = trainPageState

  // --- Non-persisted UI state ---
  const [labelSummary, setLabelSummary]     = useState(null)
  const [labelsLoading, setLabelsLoading]   = useState(false)
  const [trainingError, setTrainingError]   = useState(null)
  const [starting, setStarting]             = useState(false)
  const [runs, setRuns]                     = useState([])
  const [runsLoading, setRunsLoading]       = useState(false)
  const [renaming, setRenaming]             = useState(false)
  const [renameInput, setRenameInput]       = useState('')

  // Training mode: 'scratch' or 'finetune'
  const [trainingMode, setTrainingMode]     = useState('scratch')
  // All done jobs for base-model dropdown and comparison
  const [allJobs, setAllJobs]               = useState([])
  // Selected base job for fine-tuning
  const [baseJobId, setBaseJobId]           = useState('')
  // Compatibility message for fine-tune mode
  const [compatMsg, setCompatMsg]           = useState(null)  // {ok: bool, msg: str} | null

  // Augmentation config
  const [augOpen, setAugOpen]               = useState(false)
  const [augConfig, setAugConfig]           = useState({
    gaussian_noise:  { enabled: false, strength: 0.05 },
    time_shift:      { enabled: false, max_shift_ratio: 0.1 },
    channel_dropout: { enabled: false },
    multiplier: 3,
  })

  // Model history panel
  const [showHistory, setShowHistory]       = useState(false)
  const [historyName, setHistoryName]       = useState('')
  const [historyData, setHistoryData]       = useState(null)  // {runs: [...]}
  const [historyLoading, setHistoryLoading] = useState(false)

  // Model comparison panel
  const [showCompare, setShowCompare]       = useState(false)
  const [compareSelected, setCompareSelected] = useState(new Set())
  const [compareData, setCompareData]       = useState({})  // job_id → results

  // Manage models panel
  const [showManage, setShowManage]         = useState(false)
  const [inlineRename, setInlineRename]     = useState({})  // job_id → draft string
  const [savingRename, setSavingRename]     = useState(null)  // job_id being saved
  const [deletingJob, setDeletingJob]       = useState(null)  // job_id being deleted

  // Model import panel
  const [showImport, setShowImport]         = useState(false)
  const [importFile, setImportFile]         = useState(null)
  const [importName, setImportName]         = useState('')
  const [importing, setImporting]           = useState(false)
  const [importResult, setImportResult]     = useState(null)  // {job_id, model_name, class_names}
  const [importError, setImportError]       = useState(null)

  const wsRef = useRef(null)

  // -------------------------------------------------------------------------
  // On session change: reset run selection and reload data
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!currentSession) return
    setTrainPageState({ selectedRunId: null })
    fetchRuns()
    fetchAllJobs()
  }, [currentSession?.session_id])

  // -------------------------------------------------------------------------
  // Reconnect WebSocket when jobId is set (e.g. after page refresh)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!jobId || status !== 'running') return
    if (wsRef.current) wsRef.current.close()
    const ws = new WebSocket(`${WS_API}/api/train/ws/${jobId}/progress`)
    wsRef.current = ws
    ws.onmessage = (event) => {
      try { handleTrainingMessage(JSON.parse(event.data)) }
      catch { /* ignore non-JSON */ }
    }
    ws.onerror = () => {
      setTrainingError('WebSocket connection failed. Check the backend is running.')
      setTrainPageState({ status: 'failed' })
    }
    return () => ws.close()
  }, [jobId])

  // -------------------------------------------------------------------------
  // Validate fine-tune compatibility when base job or label summary changes
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (trainingMode !== 'finetune' || !baseJobId || !labelSummary) {
      setCompatMsg(null)
      return
    }
    const baseJob = allJobs.find(j => j.job_id === baseJobId)
    if (!baseJob) { setCompatMsg(null); return }

    const baseClasses    = [...(baseJob.classes || [])].sort()
    const currentClasses = Object.keys(labelSummary.counts).filter(l => l !== 'unknown').sort()

    if (JSON.stringify(baseClasses) === JSON.stringify(currentClasses)) {
      setCompatMsg({ ok: true, msg: `Compatible — classes match: ${baseClasses.join(', ')}` })
    } else {
      setCompatMsg({
        ok: false,
        msg: `Class mismatch: base has [${baseClasses.join(', ')}] but labels have [${currentClasses.join(', ')}]`
      })
    }
  }, [trainingMode, baseJobId, labelSummary, allJobs])

  // -------------------------------------------------------------------------
  // Message handler — processes each WebSocket message by type
  // -------------------------------------------------------------------------
  function handleTrainingMessage(msg) {
    if (msg.type === 'progress') {
      setTrainPageState(prev => ({
        history: [...prev.history, {
          epoch:      msg.epoch,
          train_loss: msg.train_loss,
          val_loss:   msg.val_loss,
          train_acc:  parseFloat((msg.train_acc * 100).toFixed(1)),
          val_acc:    parseFloat((msg.val_acc   * 100).toFixed(1)),
        }],
      }))
    } else if (msg.type === 'done') {
      setTrainPageState({ finalResult: msg, status: 'done' })
      fetchAllJobs()  // refresh so new model appears in dropdown/history/compare
    } else if (msg.type === 'error') {
      setTrainingError(msg.message)
      setTrainPageState({ status: 'failed' })
    }
  }

  // -------------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------------
  async function fetchRuns() {
    if (!currentSession) return
    setRunsLoading(true)
    try {
      const res  = await fetch(`${API}/api/pipeline/runs`)
      const json = await res.json()
      if (json.success) {
        const sessionRuns = json.data.filter(r => r.session_id === currentSession.session_id)
        setRuns(sessionRuns)
        if (sessionRuns.length > 0) {
          setTrainPageState({ selectedRunId: sessionRuns[sessionRuns.length - 1].run_id })
        }
      }
    } catch { /* ignore */ }
    finally { setRunsLoading(false) }
  }

  async function fetchAllJobs() {
    try {
      const res  = await fetch(`${API}/api/train/jobs`)
      const json = await res.json()
      if (json.success) {
        setAllJobs(json.data.jobs.filter(j => j.status === 'done'))
      }
    } catch { /* ignore */ }
  }

  // Rename any model by job_id (used in Manage Models list).
  async function renameModelInline(jobId, newName) {
    if (!newName.trim()) return
    setSavingRename(jobId)
    try {
      const res  = await fetch(`${API}/api/train/${jobId}/name`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ model_name: newName.trim() }),
      })
      const json = await res.json()
      if (json.success) {
        await fetchAllJobs()
        // Clear the inline draft for this job
        setInlineRename(prev => { const n = { ...prev }; delete n[jobId]; return n })
      }
    } catch { /* ignore */ } finally {
      setSavingRename(null)
    }
  }

  // Delete a model by job_id — removes from store + deletes .pt file.
  async function deleteModel(jobId) {
    setDeletingJob(jobId)
    try {
      const res  = await fetch(`${API}/api/train/${jobId}`, { method: 'DELETE' })
      const json = await res.json()
      if (json.success) {
        await fetchAllJobs()
        // Clear any state that referenced this job
        if (baseJobId === jobId) setBaseJobId('')
        setCompareSelected(prev => { const n = new Set(prev); n.delete(jobId); return n })
      }
    } catch { /* ignore */ } finally {
      setDeletingJob(null)
    }
  }

  // Import an external .pt model file into Neutrol's training job store.
  // Sends a multipart/form-data upload to POST /api/train/import.
  async function importModel() {
    if (!importFile) return
    setImporting(true)
    setImportResult(null)
    setImportError(null)
    try {
      const form = new FormData()
      form.append('file', importFile)
      form.append('model_name', importName || importFile.name.replace('.pt', ''))
      const res  = await fetch(`${API}/api/train/import`, { method: 'POST', body: form })
      const json = await res.json()
      if (json.success) {
        setImportResult(json.data)
        // Refresh job list so the imported model appears in dropdowns immediately
        await fetchAllJobs()
      } else {
        setImportError(json.error || 'Import failed.')
      }
    } catch (err) {
      setImportError(`Network error: ${err.message}`)
    } finally {
      setImporting(false)
    }
  }

  async function fetchLabelSummary(runId) {
    if (!currentSession) return
    setLabelsLoading(true)
    try {
      const url  = runId
        ? `${API}/api/labels/by-run/${runId}`
        : `${API}/api/labels/${currentSession.session_id}`
      const res  = await fetch(url)
      const json = await res.json()
      if (json.success) {
        const counts = {}
        json.data.labels.forEach(l => { counts[l.label] = (counts[l.label] || 0) + 1 })
        setLabelSummary({ total: json.data.total_labeled, counts })
      }
    } catch {
      setLabelSummary(null)
    } finally {
      setLabelsLoading(false)
    }
  }

  async function startTraining() {
    if (!currentSession || !labelSummary || labelSummary.total === 0) return
    if (trainingMode === 'finetune' && (!baseJobId || !compatMsg?.ok)) return

    setStarting(true)
    setTrainingError(null)
    setTrainPageState({ history: [], finalResult: null, status: 'idle' })

    const detectedClasses = Object.keys(labelSummary.counts).filter(l => l !== 'unknown').sort()
    const inputSamples    = currentSession.sample_rate * 1

    // Build augmentation payload — only include if at least one option is enabled
    const anyAugEnabled = augConfig.gaussian_noise.enabled ||
                          augConfig.time_shift.enabled ||
                          augConfig.channel_dropout.enabled
    const augPayload = anyAugEnabled ? augConfig : null

    try {
      let url, body

      if (trainingMode === 'finetune') {
        // Fine-tune: backend uses base model's architecture — we don't send layers
        url  = `${API}/api/train/finetune`
        body = {
          base_job_id:         baseJobId,
          session_id:          currentSession.session_id,
          run_id:              selectedRunId || null,
          model_name:          modelName,
          augmentation_config: augPayload,
        }
      } else {
        // Train from scratch: send full model config
        const modelConfig = {
          ...config,
          classes:        detectedClasses,
          input_channels: currentSession.n_channels,
          input_samples:  inputSamples,
          layers,
        }
        url  = `${API}/api/train/start`
        body = {
          session_id:          currentSession.session_id,
          run_id:              selectedRunId || null,
          model_name:          modelName,
          training_config:     modelConfig,
          augmentation_config: augPayload,
        }
      }

      const res  = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      })
      const json = await res.json()
      if (json.success) {
        setTrainPageState({ jobId: json.data.job_id, status: 'running' })
      } else {
        setTrainingError(json.error || 'Failed to start training.')
        setTrainPageState({ status: 'failed' })
      }
    } catch {
      setTrainingError('Could not reach the backend. Is it running on port 8000?')
      setTrainPageState({ status: 'failed' })
    } finally {
      setStarting(false)
    }
  }

  async function saveModelName() {
    if (!jobId) return
    setRenaming(true)
    try {
      const res  = await fetch(`${API}/api/train/${jobId}/name`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ model_name: renameInput }),
      })
      const json = await res.json()
      if (json.success) {
        setTrainPageState({ modelName: json.data.model_name })
        setRenameInput('')
        fetchAllJobs()
      }
    } catch { /* ignore */ }
    finally { setRenaming(false) }
  }

  async function fetchHistory() {
    if (!historyName.trim()) return
    setHistoryLoading(true)
    try {
      const res  = await fetch(`${API}/api/train/history/${encodeURIComponent(historyName.trim())}`)
      const json = await res.json()
      if (json.success) setHistoryData(json.data)
    } catch { /* ignore */ }
    finally { setHistoryLoading(false) }
  }

  async function fetchCompareResults(jobId) {
    if (compareData[jobId]) return   // already loaded
    try {
      const res  = await fetch(`${API}/api/train/${jobId}/results`)
      const json = await res.json()
      if (json.success) {
        setCompareData(prev => ({ ...prev, [jobId]: json.data }))
      }
    } catch { /* ignore */ }
  }

  function toggleCompareJob(jId) {
    setCompareSelected(prev => {
      const next = new Set(prev)
      if (next.has(jId)) {
        next.delete(jId)
      } else {
        next.add(jId)
        fetchCompareResults(jId)
      }
      return next
    })
  }

  // Update augmentation config helper — patch a nested key
  function setAugField(section, key, value) {
    setAugConfig(prev => ({
      ...prev,
      [section]: { ...prev[section], [key]: value },
    }))
  }

  // Refresh label summary whenever selected run changes
  useEffect(() => {
    if (currentSession) fetchLabelSummary(selectedRunId)
  }, [selectedRunId])

  // -------------------------------------------------------------------------
  // Guard: no session
  // -------------------------------------------------------------------------
  if (!currentSession) {
    return (
      <div className="max-w-xl">
        <h1 className="text-2xl font-bold text-gray-100 mb-2">Train</h1>
        <p className="text-gray-400 text-sm">
          No session loaded. Go to{' '}
          <a href="/import" className="text-purple-400 hover:underline">Import</a> first.
        </p>
      </div>
    )
  }

  const detectedClasses = labelSummary
    ? Object.keys(labelSummary.counts).filter(l => l !== 'unknown').sort()
    : []
  const totalEpochs  = config.epochs
  const currentEpoch = history.length

  const startDisabled = starting
    || !labelSummary
    || labelSummary.total === 0
    || (trainingMode === 'scratch' && detectedClasses.length < 2)
    || (trainingMode === 'finetune' && (!baseJobId || !compatMsg?.ok))

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold text-gray-100 mb-1">Train</h1>
      <p className="text-gray-400 text-sm mb-8">
        {currentSession.file_path?.split(/[\\/]/).pop()} —{' '}
        {currentSession.n_channels} ch, {currentSession.sample_rate} Hz
      </p>

      {/* ------------------------------------------------------------------ */}
      {/* Section 1: Pipeline Run selector                                    */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Pipeline Run
        </h2>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          {runsLoading ? (
            <p className="text-gray-500 text-sm">Loading runs…</p>
          ) : runs.length === 0 ? (
            <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg px-4 py-3 text-sm text-yellow-300">
              No pipeline runs found.{' '}
              <a href="/preprocess" className="underline hover:text-yellow-200">Go to Preprocess</a>{' '}
              and run a pipeline first.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-gray-500 mb-1">Select which pipeline run to train on:</p>
              <div className="flex flex-col gap-1.5">
                {runs.map(run => (
                  <label
                    key={run.run_id}
                    className={`flex items-center gap-3 cursor-pointer rounded-lg px-3 py-2 border transition ${
                      selectedRunId === run.run_id
                        ? 'border-purple-500 bg-purple-900/20'
                        : 'border-gray-700 bg-gray-800 hover:border-gray-600'
                    }`}
                  >
                    <input
                      type="radio"
                      name="selectedRun"
                      value={run.run_id}
                      checked={selectedRunId === run.run_id}
                      onChange={() => setTrainPageState({ selectedRunId: run.run_id })}
                      className="accent-purple-500"
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-gray-200 font-medium">
                        {run.name || <span className="text-gray-500 italic">Unnamed run</span>}
                        <span className="ml-2 text-xs text-gray-500 font-normal">{run.run_id.slice(0, 8)}</span>
                      </p>
                      <p className="text-xs text-gray-400">
                        {run.n_windows} windows · {run.steps_applied?.join(' → ')}
                      </p>
                    </div>
                  </label>
                ))}
              </div>
              <button
                onClick={fetchRuns}
                className="text-xs text-gray-600 hover:text-gray-400 transition-colors self-start mt-1"
              >
                Refresh
              </button>
            </div>
          )}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 2: Labeled data summary                                     */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Labeled Data
        </h2>
        {labelsLoading ? (
          <p className="text-gray-500 text-sm">Loading labels…</p>
        ) : !labelSummary || labelSummary.total === 0 ? (
          <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg px-4 py-3 text-sm text-yellow-300">
            No labeled windows found for this run.{' '}
            <a href="/label" className="underline hover:text-yellow-200">Go to Label</a>{' '}
            and label some windows first.
          </div>
        ) : (
          <div className="flex flex-wrap gap-3 items-center">
            <span className="text-gray-400 text-sm">{labelSummary.total} total windows</span>
            {Object.entries(labelSummary.counts).map(([label, count]) => (
              <span key={label} className={`border text-xs px-3 py-1 rounded-full ${
                label === 'unknown'
                  ? 'bg-gray-800 border-gray-600 text-gray-500'
                  : 'bg-gray-800 border-gray-700 text-gray-300'
              }`}>
                {count}× <span className={label === 'unknown' ? 'text-gray-500' : 'text-purple-300 font-medium'}>{label}</span>
              </span>
            ))}
            <button
              onClick={() => fetchLabelSummary(selectedRunId)}
              className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
            >
              Refresh
            </button>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 3: Training Mode toggle                                     */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Training Mode
        </h2>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          {/* Mode toggle buttons */}
          <div className="flex gap-2 mb-4">
            {[
              { value: 'scratch',  label: 'Train from scratch' },
              { value: 'finetune', label: 'Fine-tune existing model' },
            ].map(opt => (
              <button
                key={opt.value}
                onClick={() => setTrainingMode(opt.value)}
                disabled={status === 'running'}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  trainingMode === opt.value
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                } disabled:opacity-50`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Fine-tune: base model selector */}
          {trainingMode === 'finetune' && (
            <div className="flex flex-col gap-3">
              <div>
                <p className="text-xs text-gray-500 mb-1">Base model to fine-tune from:</p>
                {allJobs.length === 0 ? (
                  <p className="text-yellow-400 text-sm">No trained models found. Train a model first.</p>
                ) : (
                  <select
                    value={baseJobId}
                    onChange={e => setBaseJobId(e.target.value)}
                    disabled={status === 'running'}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                  >
                    <option value="">— select a model —</option>
                    {allJobs.map(j => (
                      <option key={j.job_id} value={j.job_id}>
                        {j.model_name || `Model ${j.job_id.slice(0, 8)}`}
                        {j.final_val_acc != null ? ` — ${(j.final_val_acc * 100).toFixed(1)}% val acc` : ''}
                        {' '}[{(j.classes || []).join(', ')}]
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Compatibility validation message */}
              {compatMsg && (
                <div className={`rounded-lg px-3 py-2 text-sm ${
                  compatMsg.ok
                    ? 'bg-green-900/20 border border-green-700 text-green-300'
                    : 'bg-red-900/20 border border-red-700 text-red-300'
                }`}>
                  {compatMsg.ok ? '✓ ' : '✗ '}{compatMsg.msg}
                </div>
              )}

              <p className="text-xs text-gray-600">
                Fine-tuning loads the base model's weights as a starting point. Classes must match exactly.
                The new model is saved separately — the base model is not overwritten.
              </p>
            </div>
          )}

          {/* Scratch mode: show layer editor and hyperparameters */}
          {trainingMode === 'scratch' && (
            <>
              {/* Model name */}
              <div className="mb-4">
                <p className="text-xs text-gray-500 mb-1">Model Name (optional)</p>
                <input
                  type="text"
                  placeholder="e.g. Motor Imagery v1"
                  value={modelName}
                  onChange={e => setTrainPageState({ modelName: e.target.value })}
                  disabled={status === 'running'}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                />
              </div>

              {/* Auto-detected classes */}
              <div className="mb-4">
                <p className="text-xs text-gray-500 mb-1">Classes (auto-detected from labels — excludes 'unknown')</p>
                {detectedClasses.length > 0 ? (
                  <div className="flex gap-2 flex-wrap">
                    {detectedClasses.map(c => (
                      <span key={c} className="bg-purple-900/40 border border-purple-700 text-purple-300 text-xs px-2 py-0.5 rounded">
                        {c}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-gray-600 text-xs">No labels detected</span>
                )}
              </div>

              {/* Layer architecture editor */}
              <div className="mb-5">
                <p className="text-xs text-gray-500 mb-3">
                  Layer Architecture
                  <span className="ml-2 text-gray-600">— click × to delete, use ▲▼ to reorder</span>
                </p>
                <ModelLayerEditor
                  layers={layers}
                  onChange={(newLayers) => setTrainPageState({ layers: newLayers })}
                  inputChannels={currentSession.n_channels}
                  inputSamples={currentSession.sample_rate * 1}
                  classes={detectedClasses}
                />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-400">Learning rate</span>
                  <input
                    type="number" step="0.0001" min="0.00001" max="0.1"
                    value={config.learning_rate}
                    onChange={e => setTrainPageState({ config: { ...config, learning_rate: parseFloat(e.target.value) } })}
                    disabled={status === 'running'}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-400">Epochs</span>
                  <input
                    type="number" step="1" min="1" max="500"
                    value={config.epochs}
                    onChange={e => setTrainPageState({ config: { ...config, epochs: parseInt(e.target.value) } })}
                    disabled={status === 'running'}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-400">Batch size</span>
                  <input
                    type="number" step="1" min="1" max="256"
                    value={config.batch_size}
                    onChange={e => setTrainPageState({ config: { ...config, batch_size: parseInt(e.target.value) } })}
                    disabled={status === 'running'}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                  />
                </label>
              </div>
            </>
          )}

          {/* Fine-tune mode: also show model name + hyperparameters */}
          {trainingMode === 'finetune' && baseJobId && (
            <div className="mt-4 pt-4 border-t border-gray-800">
              <p className="text-xs text-gray-500 mb-3">Fine-tune settings:</p>
              <div className="mb-3">
                <p className="text-xs text-gray-500 mb-1">Model Name (optional)</p>
                <input
                  type="text"
                  placeholder="e.g. Motor Imagery v2 (fine-tuned)"
                  value={modelName}
                  onChange={e => setTrainPageState({ modelName: e.target.value })}
                  disabled={status === 'running'}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-400">Epochs</span>
                  <input
                    type="number" step="1" min="1" max="200"
                    value={config.epochs}
                    onChange={e => setTrainPageState({ config: { ...config, epochs: parseInt(e.target.value) } })}
                    disabled={status === 'running'}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-400">Learning rate</span>
                  <input
                    type="number" step="0.0001" min="0.00001" max="0.1"
                    value={config.learning_rate}
                    onChange={e => setTrainPageState({ config: { ...config, learning_rate: parseFloat(e.target.value) } })}
                    disabled={status === 'running'}
                    className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                  />
                </label>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 4: Data Augmentation (collapsible)                          */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <button
          onClick={() => setAugOpen(o => !o)}
          className="flex items-center gap-2 text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 hover:text-gray-200 transition-colors"
        >
          <span>{augOpen ? '▾' : '▸'}</span>
          Data Augmentation
          {(augConfig.gaussian_noise.enabled || augConfig.time_shift.enabled || augConfig.channel_dropout.enabled) && (
            <span className="text-xs font-normal normal-case tracking-normal bg-purple-900/40 border border-purple-700 text-purple-300 px-2 py-0.5 rounded-full ml-1">
              active
            </span>
          )}
        </button>

        {augOpen && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-4">
              Augmentation creates extra copies of each window with synthetic variations.
              Helps the model generalize when you have limited data. Off by default.
            </p>

            <div className="flex flex-col gap-4">
              {/* Gaussian Noise */}
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="aug-noise"
                  checked={augConfig.gaussian_noise.enabled}
                  onChange={e => setAugField('gaussian_noise', 'enabled', e.target.checked)}
                  className="mt-1 accent-purple-500"
                />
                <div className="flex-1">
                  <label htmlFor="aug-noise" className="text-sm text-gray-200 font-medium cursor-pointer">
                    Gaussian Noise
                  </label>
                  <p className="text-xs text-gray-500">Adds small random noise — makes model robust to signal variation</p>
                  {augConfig.gaussian_noise.enabled && (
                    <label className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-gray-400">Strength (% of signal std):</span>
                      <input
                        type="number" step="0.01" min="0.001" max="0.5"
                        value={augConfig.gaussian_noise.strength}
                        onChange={e => setAugField('gaussian_noise', 'strength', parseFloat(e.target.value))}
                        className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500"
                      />
                    </label>
                  )}
                </div>
              </div>

              {/* Time Shift */}
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="aug-shift"
                  checked={augConfig.time_shift.enabled}
                  onChange={e => setAugField('time_shift', 'enabled', e.target.checked)}
                  className="mt-1 accent-purple-500"
                />
                <div className="flex-1">
                  <label htmlFor="aug-shift" className="text-sm text-gray-200 font-medium cursor-pointer">
                    Time Shift
                  </label>
                  <p className="text-xs text-gray-500">Shifts each window ±shift% in time — robust to temporal misalignment</p>
                  {augConfig.time_shift.enabled && (
                    <label className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-gray-400">Max shift (% of window):</span>
                      <input
                        type="number" step="0.01" min="0.01" max="0.3"
                        value={augConfig.time_shift.max_shift_ratio}
                        onChange={e => setAugField('time_shift', 'max_shift_ratio', parseFloat(e.target.value))}
                        className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500"
                      />
                    </label>
                  )}
                </div>
              </div>

              {/* Channel Dropout */}
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="aug-dropout"
                  checked={augConfig.channel_dropout.enabled}
                  onChange={e => setAugField('channel_dropout', 'enabled', e.target.checked)}
                  className="mt-1 accent-purple-500"
                />
                <div className="flex-1">
                  <label htmlFor="aug-dropout" className="text-sm text-gray-200 font-medium cursor-pointer">
                    Channel Dropout
                  </label>
                  <p className="text-xs text-gray-500">Randomly zeros one channel per copy — robust to electrode artifacts</p>
                </div>
              </div>

              {/* Multiplier */}
              {(augConfig.gaussian_noise.enabled || augConfig.time_shift.enabled || augConfig.channel_dropout.enabled) && (
                <div className="flex items-center gap-3 pt-2 border-t border-gray-800">
                  <span className="text-xs text-gray-400">Copies per original window:</span>
                  <input
                    type="number" step="1" min="2" max="10"
                    value={augConfig.multiplier}
                    onChange={e => setAugConfig(prev => ({ ...prev, multiplier: parseInt(e.target.value) }))}
                    className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500"
                  />
                  <span className="text-xs text-gray-500">
                    (dataset size × {augConfig.multiplier})
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Start training button + error                                        */}
      {/* ------------------------------------------------------------------ */}
      {status !== 'running' && (
        <div className="mb-8">
          <button
            onClick={startTraining}
            disabled={startDisabled}
            className="bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white font-medium px-6 py-2.5 rounded-lg transition-colors text-sm"
          >
            {starting
              ? 'Starting…'
              : status === 'done'
                ? (trainingMode === 'finetune' ? 'Fine-tune Again' : 'Train Again')
                : (trainingMode === 'finetune' ? 'Fine-tune' : 'Start Training')
            }
          </button>
          {trainingMode === 'scratch' && detectedClasses.length < 2 && labelSummary?.total > 0 && (
            <p className="text-yellow-400 text-xs mt-2">
              Need at least 2 different classes to train. Currently: {detectedClasses.join(', ') || 'none'}.
            </p>
          )}
          {trainingError && (
            <p className="text-red-400 text-sm mt-2">{trainingError}</p>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Section 5: Live training progress                                   */}
      {/* ------------------------------------------------------------------ */}
      {(status === 'running' || history.length > 0) && (
        <section className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
              Training Progress
            </h2>
            <div className="flex items-center gap-2">
              {status === 'running' && (
                <>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
                  </span>
                  <span className="text-xs text-gray-400">
                    Epoch {currentEpoch} / {totalEpochs}
                  </span>
                </>
              )}
              {status === 'done'   && <span className="text-xs text-green-400 font-medium">Complete</span>}
              {status === 'failed' && <span className="text-xs text-red-400 font-medium">Failed</span>}
            </div>
          </div>

          {status === 'running' && (
            <div className="mb-4">
              <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500 rounded-full transition-all duration-300"
                  style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }}
                />
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-4">
              <p className="text-xs text-gray-500 mb-3">Loss</p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#f87171" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val_loss"   name="Val Loss"   stroke="#fb923c" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {history.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs text-gray-500 mb-3">Accuracy (%)</p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', fontSize: 12 }} formatter={(v) => `${v}%`} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="train_acc" name="Train Acc" stroke="#4ade80" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val_acc"   name="Val Acc"   stroke="#60a5fa" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Section 6: Final results + confusion matrix                         */}
      {/* ------------------------------------------------------------------ */}
      {status === 'done' && finalResult && (
        <section className="mb-8">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Results
          </h2>

          {/* Rename model after training */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-5 flex items-center gap-3">
            <span className="text-xs text-gray-400 flex-shrink-0">Model name:</span>
            <input
              type="text"
              placeholder="Give this model a name…"
              value={renameInput}
              onChange={e => setRenameInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && saveModelName()}
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-purple-500"
            />
            <button
              onClick={saveModelName}
              disabled={renaming || !renameInput.trim()}
              className="bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-gray-200 text-xs px-3 py-1 rounded transition"
            >
              {renaming ? 'Saving…' : 'Save'}
            </button>
            {modelName && (
              <span className="text-xs text-purple-400 flex-shrink-0">✓ "{modelName}"</span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-500 mb-1">Final Val Accuracy</p>
              <p className="text-3xl font-bold text-green-400">
                {(finalResult.final_val_acc * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-500 mb-1">Final Train Accuracy</p>
              <p className="text-3xl font-bold text-purple-400">
                {(finalResult.final_train_acc * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {finalResult.finetuned_from && (
            <div className="bg-blue-900/20 border border-blue-800 rounded-lg px-3 py-2 text-xs text-blue-300 mb-4">
              Fine-tuned from model {finalResult.finetuned_from.slice(0, 8)}…
            </div>
          )}

          {finalResult.confusion_matrix && finalResult.class_names && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <p className="text-xs text-gray-500 mb-1">Confusion Matrix</p>
              <p className="text-xs text-gray-600 mb-4">Rows = true class, Columns = predicted class</p>
              <ConfusionMatrix
                matrix={finalResult.confusion_matrix}
                classNames={finalResult.class_names}
              />
            </div>
          )}

          {finalResult.model_path && (
            <p className="text-xs text-gray-600 mt-4">
              Model saved to: <span className="font-mono">{finalResult.model_path}</span>
            </p>
          )}
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Section 7: Model History                                            */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <button
          onClick={() => setShowHistory(h => !h)}
          className="flex items-center gap-2 text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 hover:text-gray-200 transition-colors"
        >
          <span>{showHistory ? '▾' : '▸'}</span>
          Model History
        </button>

        {showHistory && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-3">
              View all training runs for a model name to track improvement over time.
            </p>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                placeholder="Enter model name…"
                value={historyName}
                onChange={e => setHistoryName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && fetchHistory()}
                className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-purple-500"
              />
              <button
                onClick={fetchHistory}
                disabled={historyLoading || !historyName.trim()}
                className="bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-gray-200 text-sm px-4 py-1.5 rounded transition"
              >
                {historyLoading ? 'Loading…' : 'Load'}
              </button>
            </div>

            {historyData && (
              historyData.runs.length === 0 ? (
                <p className="text-gray-500 text-sm">No completed runs found for "{historyData.model_name}".</p>
              ) : (
                <>
                  {/* Accuracy chart across runs */}
                  <div className="mb-4">
                    <p className="text-xs text-gray-500 mb-2">Val Accuracy over training runs</p>
                    <ResponsiveContainer width="100%" height={160}>
                      <LineChart data={historyData.runs.map((r, i) => ({
                        run: `Run ${i + 1}`,
                        val_acc: r.final_val_acc != null ? parseFloat((r.final_val_acc * 100).toFixed(1)) : null,
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="run" stroke="#6b7280" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} domain={[0, 100]} />
                        <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', fontSize: 12 }} formatter={v => `${v}%`} />
                        <Line type="monotone" dataKey="val_acc" name="Val Acc" stroke="#60a5fa" strokeWidth={2} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* History table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="text-gray-500 border-b border-gray-800">
                          <th className="text-left py-2 pr-4 font-medium">Run</th>
                          <th className="text-left py-2 pr-4 font-medium">Date</th>
                          <th className="text-left py-2 pr-4 font-medium">Epochs</th>
                          <th className="text-left py-2 pr-4 font-medium">Val Accuracy</th>
                          <th className="text-left py-2 font-medium">Fine-tuned from</th>
                        </tr>
                      </thead>
                      <tbody>
                        {historyData.runs.map((run, i) => (
                          <tr key={run.job_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                            <td className="py-2 pr-4 text-gray-300 font-medium">Run {i + 1}</td>
                            <td className="py-2 pr-4 text-gray-400">
                              {run.started_at ? new Date(run.started_at).toLocaleDateString() : '—'}
                            </td>
                            <td className="py-2 pr-4 text-gray-400">{run.epochs_run}</td>
                            <td className="py-2 pr-4">
                              {run.final_val_acc != null
                                ? <span className="text-green-400 font-medium">{(run.final_val_acc * 100).toFixed(1)}%</span>
                                : <span className="text-gray-600">—</span>
                              }
                            </td>
                            <td className="py-2 text-gray-500">
                              {run.finetuned_from
                                ? <span className="font-mono">{run.finetuned_from.slice(0, 8)}…</span>
                                : '—'
                              }
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )
            )}
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 8: Model Comparison                                         */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <button
          onClick={() => setShowCompare(c => !c)}
          className="flex items-center gap-2 text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 hover:text-gray-200 transition-colors"
        >
          <span>{showCompare ? '▾' : '▸'}</span>
          Compare Models
          {compareSelected.size > 0 && (
            <span className="text-xs font-normal normal-case tracking-normal bg-blue-900/40 border border-blue-700 text-blue-300 px-2 py-0.5 rounded-full ml-1">
              {compareSelected.size} selected
            </span>
          )}
        </button>

        {showCompare && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            {allJobs.length === 0 ? (
              <p className="text-gray-500 text-sm">No trained models found. Train at least one model first.</p>
            ) : (
              <>
                <p className="text-xs text-gray-500 mb-3">Select models to compare side-by-side:</p>

                {/* Model checkboxes */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {allJobs.map(j => (
                    <label
                      key={j.job_id}
                      className={`flex items-center gap-2 cursor-pointer rounded-lg px-3 py-1.5 border text-xs transition ${
                        compareSelected.has(j.job_id)
                          ? 'border-blue-500 bg-blue-900/20 text-blue-300'
                          : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={compareSelected.has(j.job_id)}
                        onChange={() => toggleCompareJob(j.job_id)}
                        className="accent-blue-500"
                      />
                      {j.model_name || `Model ${j.job_id.slice(0, 8)}`}
                      {j.final_val_acc != null && (
                        <span className="text-gray-500">{(j.final_val_acc * 100).toFixed(1)}%</span>
                      )}
                    </label>
                  ))}
                </div>

                {/* Comparison table */}
                {compareSelected.size > 0 && (
                  <CompareTable
                    selectedIds={[...compareSelected]}
                    allJobs={allJobs}
                    compareData={compareData}
                  />
                )}
              </>
            )}
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 9: Manage Models — rename / delete                          */}
      {/* ------------------------------------------------------------------ */}
      <section className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <button
          onClick={() => { setShowManage(v => !v); if (!showManage) fetchAllJobs() }}
          className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-gray-300 hover:bg-gray-800 transition"
        >
          <span>{showManage ? '▾' : '▸'} Manage Models</span>
          <span className="text-xs font-normal text-gray-500">Rename or delete trained models</span>
        </button>

        {showManage && (
          <div className="px-5 py-4 border-t border-gray-800">
            {allJobs.length === 0 ? (
              <p className="text-xs text-gray-500">No trained models yet.</p>
            ) : (
              <div className="space-y-2">
                {allJobs.map(j => {
                  const displayName = j.model_name || `Model ${j.job_id.slice(0, 8)}`
                  const draft       = inlineRename[j.job_id]  // undefined = not editing
                  const isDeleting  = deletingJob === j.job_id
                  const isSaving    = savingRename === j.job_id
                  return (
                    <div key={j.job_id} className="flex items-center gap-2 bg-gray-800 rounded-lg px-3 py-2">
                      {/* Name — inline editable */}
                      {draft !== undefined ? (
                        <input
                          type="text"
                          value={draft}
                          autoFocus
                          onChange={e => setInlineRename(prev => ({ ...prev, [j.job_id]: e.target.value }))}
                          onKeyDown={e => {
                            if (e.key === 'Enter') renameModelInline(j.job_id, draft)
                            if (e.key === 'Escape') setInlineRename(prev => { const n = { ...prev }; delete n[j.job_id]; return n })
                          }}
                          className="flex-1 bg-gray-700 border border-gray-600 text-gray-100 text-xs rounded px-2 py-1 focus:outline-none focus:border-purple-500"
                        />
                      ) : (
                        <span className="flex-1 text-xs text-gray-200 truncate">{displayName}</span>
                      )}

                      {/* Acc badge */}
                      {j.final_val_acc != null && (
                        <span className="text-xs text-green-400 flex-shrink-0">{(j.final_val_acc * 100).toFixed(1)}%</span>
                      )}
                      {j.imported && (
                        <span className="text-xs text-purple-400 flex-shrink-0">imported</span>
                      )}

                      {/* Rename / Save button */}
                      {draft !== undefined ? (
                        <button
                          onClick={() => renameModelInline(j.job_id, draft)}
                          disabled={isSaving}
                          className="text-xs text-blue-400 hover:text-blue-300 disabled:opacity-40 px-1"
                        >
                          {isSaving ? '…' : 'Save'}
                        </button>
                      ) : (
                        <button
                          onClick={() => setInlineRename(prev => ({ ...prev, [j.job_id]: j.model_name || '' }))}
                          title="Rename"
                          className="text-xs text-gray-500 hover:text-gray-300 px-1"
                        >
                          ✎
                        </button>
                      )}

                      {/* Delete button */}
                      <button
                        onClick={() => { if (window.confirm(`Delete "${displayName}"? This cannot be undone.`)) deleteModel(j.job_id) }}
                        disabled={isDeleting}
                        title="Delete model"
                        className="text-xs text-gray-600 hover:text-red-400 disabled:opacity-40 px-1 transition-colors"
                      >
                        {isDeleting ? '…' : '✕'}
                      </button>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 10: Import External Model (.pt)                             */}
      {/* ------------------------------------------------------------------ */}
      <section className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowImport(v => !v)}
          className="w-full flex items-center justify-between px-5 py-3 text-sm font-semibold text-gray-300 hover:bg-gray-800 transition"
        >
          <span>{showImport ? '▾' : '▸'} Import External Model (.pt)</span>
          <span className="text-xs font-normal text-gray-500">Load a previously trained model</span>
        </button>

        {showImport && (
          <div className="px-5 py-4 border-t border-gray-800 space-y-3">
            <p className="text-xs text-gray-400">
              Import a <code className="text-purple-300">.pt</code> checkpoint file exported from Neutrol (or any PyTorch model
              saved with <code className="text-gray-400">torch.save()</code> that includes <code className="text-gray-400">model_state_dict</code> and <code className="text-gray-400">class_names</code> keys).
              The model will appear in the Inference and Export pages immediately.
            </p>

            <div className="flex gap-3 flex-wrap items-end">
              {/* File picker */}
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Model file (.pt)</label>
                <input
                  type="file"
                  accept=".pt"
                  onChange={e => { setImportFile(e.target.files[0] || null); setImportResult(null); setImportError(null) }}
                  className="text-xs text-gray-300 file:mr-2 file:py-1 file:px-3 file:rounded file:border-0 file:bg-gray-700 file:text-gray-200 file:text-xs hover:file:bg-gray-600 cursor-pointer"
                />
              </div>

              {/* Name input */}
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400">Model name (optional)</label>
                <input
                  type="text"
                  placeholder="e.g. MyMotorModel"
                  value={importName}
                  onChange={e => setImportName(e.target.value)}
                  className="bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-3 py-1.5 focus:outline-none focus:border-purple-500 w-48"
                />
              </div>

              {/* Import button */}
              <button
                onClick={importModel}
                disabled={!importFile || importing}
                className="bg-purple-700 hover:bg-purple-600 disabled:opacity-40 text-white text-xs font-medium rounded px-4 py-1.5 transition"
              >
                {importing ? 'Importing…' : 'Import Model'}
              </button>
            </div>

            {/* Result */}
            {importResult && (
              <div className="bg-green-900/30 border border-green-700 rounded p-3 text-xs text-green-300">
                <p className="font-semibold mb-1">Model imported successfully</p>
                <p>Name: <span className="text-white">{importResult.model_name}</span></p>
                <p>Classes: <span className="text-white">{importResult.class_names?.join(', ')}</span></p>
                <p>Job ID: <code className="text-gray-400">{importResult.job_id?.slice(0, 8)}…</code></p>
                <p className="mt-1 text-green-400">The model is now available in Inference and Export pages.</p>
              </div>
            )}
            {importError && (
              <p className="text-red-400 text-xs bg-red-900/20 rounded p-2">{importError}</p>
            )}
          </div>
        )}
      </section>
    </div>
  )
}


// ---------------------------------------------------------------------------
// ConfusionMatrix — renders an n×n table with heat-coloured cells
// ---------------------------------------------------------------------------
function ConfusionMatrix({ matrix, classNames }) {
  const maxVal = Math.max(...matrix.flat(), 1)

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="px-2 py-1 text-gray-600 font-normal text-right">True ↓ / Pred →</th>
            {classNames.map(name => (
              <th key={name} className="px-3 py-1 text-gray-400 font-medium text-center">{name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIdx) => (
            <tr key={rowIdx}>
              <td className="px-2 py-1 text-gray-400 font-medium text-right pr-3">
                {classNames[rowIdx]}
              </td>
              {row.map((count, colIdx) => {
                const intensity = count / maxVal
                const isCorrect = rowIdx === colIdx
                const bg = isCorrect
                  ? `rgba(74, 222, 128, ${0.1 + intensity * 0.7})`
                  : count > 0
                    ? `rgba(248, 113, 113, ${0.1 + intensity * 0.7})`
                    : 'transparent'
                return (
                  <td
                    key={colIdx}
                    className="px-4 py-2 text-center font-mono font-medium border border-gray-800"
                    style={{ background: bg, color: count > 0 ? '#f3f4f6' : '#4b5563' }}
                  >
                    {count}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}


// ---------------------------------------------------------------------------
// CompareTable — side-by-side metrics for selected models
// ---------------------------------------------------------------------------
function CompareTable({ selectedIds, allJobs, compareData }) {
  // Only show models whose results have been fetched
  const ready = selectedIds.filter(id => compareData[id]?.final_metrics)
  if (ready.length === 0) {
    return <p className="text-gray-500 text-xs">Loading results…</p>
  }

  // Collect all class names across selected models (union)
  const classSet = new Set()
  ready.forEach(id => {
    const cm = compareData[id]?.final_metrics
    if (cm?.class_names) cm.class_names.forEach(c => classSet.add(c))
  })
  const classNames = [...classSet].sort()

  // Build per-model metrics
  const modelMetrics = ready.map(id => {
    const job     = allJobs.find(j => j.job_id === id) || {}
    const fm      = compareData[id]?.final_metrics || {}
    const matrix  = fm.confusion_matrix
    const classes = fm.class_names || []
    const perClass = matrix ? computeMetrics(matrix, classes) : []

    return {
      id,
      label:    job.model_name || `Model ${id.slice(0, 8)}`,
      overall:  fm.final_val_acc != null ? (fm.final_val_acc * 100).toFixed(1) + '%' : '—',
      perClass, // list of {name, precision, recall, f1}
    }
  })

  // Build rows: Overall Accuracy, then per-class precision/recall/F1
  const rows = [
    {
      metric: 'Overall Val Accuracy',
      values: modelMetrics.map(m => ({ label: m.overall, highlight: true })),
    },
    ...classNames.flatMap(cls => [
      {
        metric: `${cls} — precision`,
        values: modelMetrics.map(m => {
          const entry = m.perClass.find(p => p.name === cls)
          return { label: entry ? entry.precision.toFixed(3) : '—', highlight: false }
        }),
      },
      {
        metric: `${cls} — recall`,
        values: modelMetrics.map(m => {
          const entry = m.perClass.find(p => p.name === cls)
          return { label: entry ? entry.recall.toFixed(3) : '—', highlight: false }
        }),
      },
      {
        metric: `${cls} — F1`,
        values: modelMetrics.map(m => {
          const entry = m.perClass.find(p => p.name === cls)
          return { label: entry ? entry.f1.toFixed(3) : '—', highlight: false }
        }),
      },
    ]),
  ]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="text-gray-500 border-b border-gray-800">
            <th className="text-left py-2 pr-6 font-medium min-w-32">Metric</th>
            {modelMetrics.map(m => (
              <th key={m.id} className="text-center py-2 px-4 font-medium text-gray-300">
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            // Separate row for per-class grouping visual clarity
            const isAccRow = row.metric === 'Overall Val Accuracy'
            return (
              <tr
                key={i}
                className={`border-b border-gray-800/50 ${isAccRow ? 'bg-gray-800/30' : 'hover:bg-gray-800/20'}`}
              >
                <td className={`py-2 pr-6 ${isAccRow ? 'text-gray-200 font-medium' : 'text-gray-500'}`}>
                  {row.metric}
                </td>
                {row.values.map((v, j) => (
                  <td
                    key={j}
                    className={`text-center py-2 px-4 font-mono ${
                      v.highlight ? 'text-green-400 font-bold text-sm' : 'text-gray-300'
                    }`}
                  >
                    {v.label}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
