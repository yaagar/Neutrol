// TrainPage.jsx — model configuration, training trigger, live progress, and results
//
// Page flow:
//   1. Load label summary from the backend → show class distribution
//   2. User configures the model (learning rate, epochs, batch size)
//      Classes are auto-detected from the labels
//   3. User clicks "Start Training" → POST /api/train/start → get job_id
//   4. Connect WebSocket to /api/train/ws/{job_id}/progress
//   5. Receive progress messages → update loss/accuracy charts in real time
//   6. When training finishes → show final accuracy + confusion matrix
import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { useSession } from '../hooks/useSession'
import ModelLayerEditor from '../components/ModelLayerEditor'

import { API_BASE_URL, WS_BASE_URL } from '../config'
const API    = API_BASE_URL
const WS_API = WS_BASE_URL

// Default model configuration — matches CLAUDE.md ModelConfig spec
const DEFAULT_CONFIG = {
  learning_rate: 0.001,
  epochs:        50,
  batch_size:    32,
}

// Default EEGNet-inspired layer architecture from CLAUDE.md
const DEFAULT_LAYERS = [
  { type: 'Conv2d',          out_channels: 16, kernel_size: [1, 32], padding: 'same', activation: 'ELU', batch_norm: true },
  { type: 'DepthwiseConv2d', depth_multiplier: 2, kernel_size: ['n_channels', 1], activation: 'ELU', batch_norm: true },
  { type: 'AvgPool2d',       kernel_size: [1, 4] },
  { type: 'Dropout',         rate: 0.5 },
  { type: 'Conv2d',          out_channels: 32, kernel_size: [1, 16], padding: 'same', activation: 'ELU', batch_norm: true },
  { type: 'AvgPool2d',       kernel_size: [1, 8] },
  { type: 'Dropout',         rate: 0.5 },
  { type: 'Flatten' },
  { type: 'Linear',          out_features: 'n_classes' },
]

export default function TrainPage() {
  const { currentSession } = useSession()

  // --- Label summary ---
  const [labelSummary, setLabelSummary]   = useState(null)   // {total_labeled, label_counts}
  const [labelsLoading, setLabelsLoading] = useState(false)

  // --- Model config form ---
  const [config, setConfig]   = useState(DEFAULT_CONFIG)
  const [layers, setLayers]   = useState(DEFAULT_LAYERS)

  // --- Training state ---
  const [jobId, setJobId]           = useState(null)
  const [status, setStatus]         = useState('idle')     // idle | running | done | failed
  const [history, setHistory]       = useState([])         // [{epoch, train_loss, val_loss, train_acc, val_acc}]
  const [finalResult, setFinalResult] = useState(null)     // the "done" message from WS
  const [trainingError, setTrainingError] = useState(null)
  const [starting, setStarting]     = useState(false)

  // WebSocket ref — we manage it manually here rather than via useWebSocket
  // because we need fine-grained control over when to connect (after job starts)
  const wsRef = useRef(null)

  // --- Load label summary when session changes ---
  useEffect(() => {
    if (!currentSession) return
    fetchLabelSummary()
    // Reset training state when switching sessions
    setJobId(null)
    setStatus('idle')
    setHistory([])
    setFinalResult(null)
    setTrainingError(null)
  }, [currentSession?.session_id])

  // --- Connect WebSocket when a job starts ---
  useEffect(() => {
    if (!jobId) return

    // Close any existing connection first
    if (wsRef.current) wsRef.current.close()

    const ws = new WebSocket(`${WS_API}/api/train/ws/${jobId}/progress`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        handleTrainingMessage(msg)
      } catch {
        // Ignore non-JSON messages
      }
    }

    ws.onerror = () => {
      setTrainingError('WebSocket connection failed. Check the backend is running.')
      setStatus('failed')
    }

    // Cleanup: close socket when component unmounts or jobId changes
    return () => ws.close()
  }, [jobId])

  // -------------------------------------------------------------------------
  // Message handler — processes each WebSocket message by type
  // -------------------------------------------------------------------------

  function handleTrainingMessage(msg) {
    if (msg.type === 'progress') {
      // Append one data point to the chart history
      setHistory(prev => [...prev, {
        epoch:      msg.epoch,
        train_loss: msg.train_loss,
        val_loss:   msg.val_loss,
        train_acc:  parseFloat((msg.train_acc * 100).toFixed(1)),
        val_acc:    parseFloat((msg.val_acc   * 100).toFixed(1)),
      }])
    } else if (msg.type === 'done') {
      setFinalResult(msg)
      setStatus('done')
    } else if (msg.type === 'error') {
      setTrainingError(msg.message)
      setStatus('failed')
    }
  }

  // -------------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------------

  async function fetchLabelSummary() {
    if (!currentSession) return
    setLabelsLoading(true)
    try {
      const res  = await fetch(`${API}/api/labels/${currentSession.session_id}`)
      const json = await res.json()
      if (json.success) {
        // Build a counts map: {label: count}
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

    setStarting(true)
    setTrainingError(null)
    setHistory([])
    setFinalResult(null)
    setStatus('idle')

    // Auto-detect classes from labels
    const detectedClasses = Object.keys(labelSummary.counts).sort()

    // input_samples = sample_rate × 1s window (default window size)
    const inputSamples = currentSession.sample_rate * 1

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
        body:    JSON.stringify({
          session_id:      currentSession.session_id,
          training_config: modelConfig,
        }),
      })
      const json = await res.json()
      if (json.success) {
        setJobId(json.data.job_id)
        setStatus('running')
      } else {
        setTrainingError(json.error || 'Failed to start training.')
        setStatus('failed')
      }
    } catch {
      setTrainingError('Could not reach the backend. Is it running on port 8000?')
      setStatus('failed')
    } finally {
      setStarting(false)
    }
  }

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

  // Detect the classes auto from labeled data
  const detectedClasses = labelSummary ? Object.keys(labelSummary.counts).sort() : []
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
      {/* Section 1: Label summary                                             */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Labeled Data
        </h2>

        {labelsLoading ? (
          <p className="text-gray-500 text-sm">Loading labels…</p>
        ) : !labelSummary || labelSummary.total === 0 ? (
          <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg px-4 py-3 text-sm text-yellow-300">
            No labeled windows found for this session.{' '}
            <a href="/label" className="underline hover:text-yellow-200">Go to Label</a>{' '}
            and label some windows first.
          </div>
        ) : (
          <div className="flex flex-wrap gap-3 items-center">
            <span className="text-gray-400 text-sm">{labelSummary.total} total windows</span>
            {Object.entries(labelSummary.counts).map(([label, count]) => (
              <span key={label} className="bg-gray-800 border border-gray-700 text-gray-300 text-xs px-3 py-1 rounded-full">
                {count}× <span className="text-purple-300 font-medium">{label}</span>
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
      {/* Section 2: Model configuration                                       */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Model Configuration
        </h2>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          {/* Auto-detected classes */}
          <div className="mb-4">
            <p className="text-xs text-gray-500 mb-1">Classes (auto-detected from labels)</p>
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
                — drag cards up/down to reorder, click × to delete
              </span>
            </p>
            <ModelLayerEditor
              layers={layers}
              onChange={setLayers}
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
                onChange={e => setConfig(c => ({ ...c, learning_rate: parseFloat(e.target.value) }))}
                disabled={status === 'running'}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Epochs</span>
              <input
                type="number" step="1" min="1" max="500"
                value={config.epochs}
                onChange={e => setConfig(c => ({ ...c, epochs: parseInt(e.target.value) }))}
                disabled={status === 'running'}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-purple-500 disabled:opacity-50"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-400">Batch size</span>
              <input
                type="number" step="1" min="1" max="256"
                value={config.batch_size}
                onChange={e => setConfig(c => ({ ...c, batch_size: parseInt(e.target.value) }))}
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
      {/* Section 3: Live training progress                                    */}
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
                  {/* Animated pulse dot to show it's live */}
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

          {/* Epoch progress bar */}
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

          {/* Loss chart */}
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
                  {/* Train loss in red, val loss in orange — conventional colour scheme */}
                  <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#f87171" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val_loss"   name="Val Loss"   stroke="#fb923c" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Accuracy chart */}
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
                  {/* Train acc in green, val acc in blue */}
                  <Line type="monotone" dataKey="train_acc" name="Train Acc" stroke="#4ade80" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val_acc"   name="Val Acc"   stroke="#60a5fa" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Section 4: Final results + confusion matrix                          */}
      {/* ------------------------------------------------------------------ */}
      {status === 'done' && finalResult && (
        <section>
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Results
          </h2>

          {/* Summary cards */}
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

          {/* Confusion matrix */}
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
// Props:
//   matrix     {number[][]} — n_classes × n_classes count matrix
//   classNames {string[]}   — class name for each row/column
// ---------------------------------------------------------------------------
function ConfusionMatrix({ matrix, classNames }) {
  // Find the maximum value to normalise cell colour intensity
  const maxVal = Math.max(...matrix.flat(), 1)

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            {/* Top-left empty cell */}
            <th className="px-2 py-1 text-gray-600 font-normal text-right">True ↓ / Pred →</th>
            {classNames.map(name => (
              <th key={name} className="px-3 py-1 text-gray-400 font-medium text-center">
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {/* Row label */}
              <td className="px-2 py-1 text-gray-400 font-medium text-right pr-3">
                {classNames[rowIdx]}
              </td>
              {row.map((count, colIdx) => {
                // Diagonal cells (correct predictions) are green-tinted;
                // off-diagonal (errors) are red-tinted.
                // Opacity scales with the cell value relative to the max.
                const intensity = count / maxVal
                const isCorrect = rowIdx === colIdx
                const bg = isCorrect
                  ? `rgba(74, 222, 128, ${0.1 + intensity * 0.7})`   // green for correct
                  : count > 0
                    ? `rgba(248, 113, 113, ${0.1 + intensity * 0.7})` // red for errors
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
