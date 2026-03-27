// TrainPage.jsx — model configuration, training trigger, live progress, and results
//
// Page flow:
//   1. Load label summary from the backend → show class distribution
//   2. User selects a pipeline run to train on
//   3. User configures the model (learning rate, epochs, batch size, layers)
//   4. User clicks "Start Training" → POST /api/train/start → get job_id
//   5. Connect WebSocket to /api/train/ws/{job_id}/progress
//   6. Receive progress messages → update loss/accuracy charts in real time
//   7. When training finishes → show final accuracy + confusion matrix
//
// State is persisted in Zustand (trainPageState) so navigating away and back
// does not reset the form or wipe the in-progress training results.
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

export default function TrainPage() {
  const { currentSession } = useSession()

  // --- Zustand persistent state ---
  const trainPageState     = useAppStore(s => s.trainPageState)
  const setTrainPageState  = useAppStore(s => s.setTrainPageState)

  // Destructure for local convenience — writes go back through setTrainPageState
  const { config, layers, jobId, status, history, finalResult, selectedRunId, modelName } = trainPageState

  // --- Non-persisted UI state (reset on mount is fine) ---
  const [labelSummary, setLabelSummary]     = useState(null)
  const [labelsLoading, setLabelsLoading]   = useState(false)
  const [trainingError, setTrainingError]   = useState(null)
  const [starting, setStarting]             = useState(false)
  const [runs, setRuns]                     = useState([])
  const [runsLoading, setRunsLoading]       = useState(false)
  const [renaming, setRenaming]             = useState(false)
  const [renameInput, setRenameInput]       = useState('')

  // WebSocket ref — managed manually for fine-grained connect/disconnect control
  const wsRef = useRef(null)

  // --- On session change: reset run selection, reload runs + labels ---
  useEffect(() => {
    if (!currentSession) return
    // Reset the run selection when switching sessions so we don't carry over a
    // stale run_id from the previous session — that would make the label filter
    // return 0 results for the new session's labels.
    setTrainPageState({ selectedRunId: null })
    fetchRuns()
    fetchLabelSummary()
  }, [currentSession?.session_id])

  // --- Reconnect WebSocket when jobId is set (e.g. after page refresh restores state) ---
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
        // Always select the most recent run for this session.
        // Not using !selectedRunId check because the previous session's run_id
        // could still be in Zustand state after a session switch.
        if (sessionRuns.length > 0) {
          setTrainPageState({ selectedRunId: sessionRuns[sessionRuns.length - 1].run_id })
        }
      }
    } catch { /* ignore */ }
    finally { setRunsLoading(false) }
  }

  async function fetchLabelSummary() {
    if (!currentSession) return
    setLabelsLoading(true)
    try {
      const res  = await fetch(`${API}/api/labels/${currentSession.session_id}`)
      const json = await res.json()
      if (json.success) {
        // Build counts for labels that belong to the selected run
        const relevantLabels = selectedRunId
          ? json.data.labels.filter(l => l.run_id === selectedRunId)
          : json.data.labels
        const counts = {}
        relevantLabels.forEach(l => { counts[l.label] = (counts[l.label] || 0) + 1 })
        setLabelSummary({ total: relevantLabels.length, counts })
      }
    } catch {
      setLabelSummary(null)
    } finally {
      setLabelsLoading(false)
    }
  }

  async function startTraining() {
    if (!currentSession || !labelSummary || labelSummary.total === 0) return

    setStarting(true)
    setTrainingError(null)
    setTrainPageState({ history: [], finalResult: null, status: 'idle' })

    const detectedClasses = Object.keys(labelSummary.counts).sort()
    const inputSamples    = currentSession.sample_rate * 1

    const modelConfig = {
      ...config,
      classes:        detectedClasses,
      input_channels: currentSession.n_channels,
      input_samples:  inputSamples,
      layers,
    }

    try {
      const res  = await fetch(`${API}/api/train/start`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:      currentSession.session_id,
          run_id:          selectedRunId || null,
          model_name:      modelName,
          training_config: modelConfig,
        }),
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
      const res = await fetch(`${API}/api/train/${jobId}/name`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: renameInput }),
      })
      const json = await res.json()
      if (json.success) {
        setTrainPageState({ modelName: json.data.model_name })
        setRenameInput('')
      }
    } catch { /* ignore */ }
    finally { setRenaming(false) }
  }

  // Refresh label summary when selected run changes
  useEffect(() => {
    if (currentSession) fetchLabelSummary()
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

  const detectedClasses = labelSummary ? Object.keys(labelSummary.counts).filter(l => l !== 'unknown').sort() : []
  const totalEpochs     = config.epochs
  const currentEpoch    = history.length

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
      {/* Section 1: Run selector                                              */}
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
      {/* Section 2: Label summary                                             */}
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
              onClick={fetchLabelSummary}
              className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
            >
              Refresh
            </button>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 3: Model configuration                                       */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Model Configuration
        </h2>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
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
              <span className="ml-2 text-gray-600">
                — click × to delete, use ▲▼ to reorder
              </span>
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
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Start training button + error                                        */}
      {/* ------------------------------------------------------------------ */}
      {status !== 'running' && (
        <div className="mb-8">
          <button
            onClick={startTraining}
            disabled={starting || !labelSummary || labelSummary.total === 0 || detectedClasses.length < 2}
            className="bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white font-medium px-6 py-2.5 rounded-lg transition-colors text-sm"
          >
            {starting ? 'Starting…' : status === 'done' ? 'Train Again' : 'Start Training'}
          </button>
          {detectedClasses.length < 2 && labelSummary?.total > 0 && (
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
      {/* Section 4: Live training progress                                    */}
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
              {status === 'done' && (
                <span className="text-xs text-green-400 font-medium">Complete</span>
              )}
              {status === 'failed' && (
                <span className="text-xs text-red-400 font-medium">Failed</span>
              )}
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
      {/* Section 5: Final results + confusion matrix                          */}
      {/* ------------------------------------------------------------------ */}
      {status === 'done' && finalResult && (
        <section>
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
