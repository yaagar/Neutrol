// ExportPage.jsx — download trained models, scripts, and pipeline configs
//
// Layout per trained model — four export cards:
//   1. Python Script    — standalone .py + .pt, only needs numpy + torch
//   2. ONNX Package     — .onnx + onnxruntime .py, no PyTorch needed
//   3. Microcontroller  — ZIP with TFLite .h, ESP32 .cpp, README
//   4. Pipeline Config  — preprocessing steps as .json
//
// Downloads are triggered via a hidden <a> element. FastAPI responds with
// Content-Disposition: attachment, so the browser saves the file.
import { useState, useEffect } from 'react'

import { API_BASE_URL } from '../config'
const API = API_BASE_URL

export default function ExportPage() {
  const [jobs, setJobs]           = useState([])
  const [runs, setRuns]           = useState([])
  const [jobsLoading, setJobsLoading] = useState(false)
  const [runsLoading, setRunsLoading] = useState(false)
  // downloading tracks which button is active: e.g. 'model-{id}', 'onnx-{id}', etc.
  const [downloading, setDownloading] = useState(null)

  useEffect(() => {
    fetchJobs()
    fetchRuns()
  }, [])

  async function fetchJobs() {
    setJobsLoading(true)
    try {
      const res  = await fetch(`${API}/api/train/jobs`)
      const json = await res.json()
      if (json.success) {
        setJobs(json.data.jobs.filter(j => j.status === 'done'))
      }
    } catch { /* ignore — backend not running */ }
    finally { setJobsLoading(false) }
  }

  async function fetchRuns() {
    setRunsLoading(true)
    try {
      const res  = await fetch(`${API}/api/pipeline/runs`)
      const json = await res.json()
      if (json.success) setRuns(json.data)
    } catch { /* ignore */ }
    finally { setRunsLoading(false) }
  }

  // Trigger a file download by navigating to the export endpoint.
  // A hidden <a> with target="_blank" lets the browser handle the
  // Content-Disposition: attachment header and save the file.
  function triggerDownload(url, key) {
    setDownloading(key)
    const a = document.createElement('a')
    a.href   = url
    a.target = '_blank'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setTimeout(() => setDownloading(null), 1500)
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold text-gray-100 mb-1">Export</h1>
      <p className="text-gray-400 text-sm mb-8">
        Download everything needed to use your model outside Neutrol.
        Keep the script and model file in the same folder.
      </p>

      {/* ------------------------------------------------------------------ */}
      {/* Section 1: Trained models                                            */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-10">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
            Trained Models
          </h2>
          <button
            onClick={fetchJobs}
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            Refresh
          </button>
        </div>

        {jobsLoading ? (
          <p className="text-gray-500 text-sm">Loading…</p>
        ) : jobs.length === 0 ? (
          <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-6 text-center">
            <p className="text-gray-500 text-sm">No completed training jobs yet.</p>
            <p className="text-gray-600 text-xs mt-1">
              Train a model on the{' '}
              <a href="/train" className="text-purple-400 hover:underline">Train page</a>{' '}
              first.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {jobs.map(job => (
              <ModelCard
                key={job.job_id}
                job={job}
                downloading={downloading}
                onDownloadModel={() =>
                  triggerDownload(`${API}/api/export/model/${job.job_id}`, `model-${job.job_id}`)
                }
                onDownloadScript={() =>
                  triggerDownload(`${API}/api/export/script/${job.job_id}`, `script-${job.job_id}`)
                }
                onDownloadOnnx={() =>
                  triggerDownload(`${API}/api/export/onnx/${job.job_id}`, `onnx-${job.job_id}`)
                }
                onDownloadOnnxScript={() =>
                  triggerDownload(`${API}/api/export/onnx-script/${job.job_id}`, `onnx-script-${job.job_id}`)
                }
                onDownloadMicro={() =>
                  triggerDownload(`${API}/api/export/microcontroller/${job.job_id}`, `micro-${job.job_id}`)
                }
              />
            ))}
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 2: Pipeline configs                                          */}
      {/* ------------------------------------------------------------------ */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
            Pipeline Configurations
          </h2>
          <button
            onClick={fetchRuns}
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            Refresh
          </button>
        </div>

        {runsLoading ? (
          <p className="text-gray-500 text-sm">Loading…</p>
        ) : runs.length === 0 ? (
          <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-6 text-center">
            <p className="text-gray-500 text-sm">No pipeline runs yet.</p>
            <p className="text-gray-600 text-xs mt-1">
              Run a preprocessing pipeline on the{' '}
              <a href="/preprocess" className="text-purple-400 hover:underline">Preprocess page</a>{' '}
              first.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {runs.map(run => (
              <PipelineCard
                key={run.run_id}
                run={run}
                downloading={downloading}
                onDownload={() =>
                  triggerDownload(`${API}/api/export/pipeline/${run.run_id}`, `pipeline-${run.run_id}`)
                }
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}


// ---------------------------------------------------------------------------
// ModelCard — one trained model with four export format cards
// ---------------------------------------------------------------------------

function ModelCard({
  job,
  downloading,
  onDownloadModel,
  onDownloadScript,
  onDownloadOnnx,
  onDownloadOnnxScript,
  onDownloadMicro,
}) {
  const accuracy = job.final_val_acc != null
    ? `${(job.final_val_acc * 100).toFixed(1)}% val acc`
    : null

  const date = job.started_at
    ? new Date(job.started_at).toLocaleDateString(undefined, {
        month: 'short', day: 'numeric', year: 'numeric',
      })
    : null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      {/* Model header — name, classes, stats */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" />
            <p className="text-sm font-medium text-gray-100 truncate">
              {job.model_name || <span className="text-gray-500 italic">Unnamed model</span>}
            </p>
            <span className="text-xs text-gray-600 font-mono flex-shrink-0">
              {job.job_id.slice(0, 8)}
            </span>
          </div>

          {job.classes?.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-1">
              {job.classes.map(c => (
                <span
                  key={c}
                  className="bg-purple-900/40 border border-purple-700 text-purple-300 text-xs px-2 py-0.5 rounded"
                >
                  {c}
                </span>
              ))}
            </div>
          )}

          <p className="text-xs text-gray-500">
            {[accuracy, date, `${job.epochs_run} epoch${job.epochs_run !== 1 ? 's' : ''}`]
              .filter(Boolean)
              .join(' · ')}
          </p>
        </div>
      </div>

      {/* Four export format cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

        {/* Card 1 — Python Script */}
        <ExportFormatCard
          title="Python Script"
          badge=".py + .pt"
          description="Standalone inference. Only needs numpy + torch. Call predict(window) on any raw window — preprocessing is embedded."
          badgeColor="purple"
          buttons={[
            {
              label: 'Model weights',
              sublabel: '.pt',
              key: `model-${job.job_id}`,
              onClick: onDownloadModel,
              color: 'gray',
            },
            {
              label: 'Inference script',
              sublabel: '.py',
              key: `script-${job.job_id}`,
              onClick: onDownloadScript,
              color: 'purple',
            },
          ]}
          downloading={downloading}
        />

        {/* Card 2 — ONNX */}
        <ExportFormatCard
          title="ONNX Package"
          badge=".onnx + .py"
          description="No PyTorch needed at runtime. Use onnxruntime — faster on CPU, deployable on edge devices, mobile, and web."
          badgeColor="blue"
          buttons={[
            {
              label: 'ONNX model',
              sublabel: '.onnx',
              key: `onnx-${job.job_id}`,
              onClick: onDownloadOnnx,
              color: 'gray',
            },
            {
              label: 'ONNX script',
              sublabel: '.py',
              key: `onnx-script-${job.job_id}`,
              onClick: onDownloadOnnxScript,
              color: 'blue',
            },
          ]}
          downloading={downloading}
        />

        {/* Card 3 — Microcontroller */}
        <ExportFormatCard
          title="Microcontroller"
          badge=".zip"
          description="TFLite C array + ESP32 Arduino sketch. Flash directly to a microcontroller. Requires tensorflow + onnx2tf."
          badgeColor="green"
          buttons={[
            {
              label: 'Download ZIP',
              sublabel: '.zip',
              key: `micro-${job.job_id}`,
              onClick: onDownloadMicro,
              color: 'green',
            },
          ]}
          downloading={downloading}
        />

        {/* Card 4 — Pipeline JSON (links to pipeline runs section) */}
        <ExportFormatCard
          title="Pipeline Config"
          badge=".json"
          description="Preprocessing steps with all parameters. Load back into Neutrol or replay manually. See the Pipeline Configs section below."
          badgeColor="yellow"
          buttons={[]}
          downloading={downloading}
          footer={
            <p className="text-xs text-gray-600 mt-1">
              Download from the{' '}
              <button
                onClick={() => document.querySelector('section:last-of-type')?.scrollIntoView({ behavior: 'smooth' })}
                className="text-yellow-500 hover:underline"
              >
                Pipeline Configurations
              </button>{' '}
              section below.
            </p>
          }
        />
      </div>
    </div>
  )
}


// ---------------------------------------------------------------------------
// ExportFormatCard — one format tile inside a ModelCard
// ---------------------------------------------------------------------------

function ExportFormatCard({ title, badge, description, badgeColor, buttons, downloading, footer }) {
  const badgeColors = {
    purple: 'bg-purple-900/40 border-purple-700 text-purple-300',
    blue:   'bg-blue-900/40   border-blue-700   text-blue-300',
    green:  'bg-green-900/40  border-green-700  text-green-300',
    yellow: 'bg-yellow-900/30 border-yellow-700 text-yellow-300',
    gray:   'bg-gray-800      border-gray-700   text-gray-400',
  }

  return (
    <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 flex flex-col gap-2">
      {/* Title + badge */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-200">{title}</span>
        <span className={`text-xs px-1.5 py-0.5 rounded border font-mono ${badgeColors[badgeColor]}`}>
          {badge}
        </span>
      </div>

      {/* Description */}
      <p className="text-xs text-gray-500 leading-relaxed">{description}</p>

      {/* Download buttons */}
      {buttons.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-auto">
          {buttons.map(btn => (
            <DownloadButton
              key={btn.key}
              label={btn.label}
              sublabel={btn.sublabel}
              active={downloading === btn.key}
              onClick={btn.onClick}
              color={btn.color}
            />
          ))}
        </div>
      )}

      {footer}
    </div>
  )
}


// ---------------------------------------------------------------------------
// PipelineCard — one pipeline run with a download button
// ---------------------------------------------------------------------------

function PipelineCard({ run, downloading, onDownload }) {
  const key  = `pipeline-${run.run_id}`
  const date = run.created_at
    ? new Date(run.created_at).toLocaleDateString(undefined, {
        month: 'short', day: 'numeric', year: 'numeric',
      })
    : null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-between gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <p className="text-sm font-medium text-gray-100 truncate">
            {run.name || <span className="text-gray-500 italic">Unnamed run</span>}
          </p>
          <span className="text-xs text-gray-600 font-mono flex-shrink-0">
            {run.run_id.slice(0, 8)}
          </span>
        </div>

        {run.steps_applied?.length > 0 && (
          <p className="text-xs text-gray-500 mb-1">
            {run.steps_applied.join(' → ')}
          </p>
        )}

        <p className="text-xs text-gray-600">
          {[
            `${run.n_windows} windows`,
            run.window_shape ? `${run.window_shape[0]}ch × ${run.window_shape[1]}samp` : null,
            date,
          ].filter(Boolean).join(' · ')}
        </p>
      </div>

      <DownloadButton
        label="Pipeline config"
        sublabel=".json"
        active={downloading === key}
        onClick={onDownload}
        color="green"
      />
    </div>
  )
}


// ---------------------------------------------------------------------------
// DownloadButton — reusable styled download button
// ---------------------------------------------------------------------------

const COLOR_MAP = {
  purple: 'border-purple-700 bg-purple-900/20 hover:bg-purple-900/40 text-purple-300',
  blue:   'border-blue-700   bg-blue-900/20   hover:bg-blue-900/40   text-blue-300',
  green:  'border-green-700  bg-green-900/20  hover:bg-green-900/40  text-green-300',
  yellow: 'border-yellow-700 bg-yellow-900/20 hover:bg-yellow-900/40 text-yellow-300',
  gray:   'border-gray-700   bg-gray-800/60   hover:bg-gray-700/60   text-gray-300',
}

function DownloadButton({ label, sublabel, active, onClick, color = 'purple' }) {
  return (
    <button
      onClick={onClick}
      disabled={active}
      className={`
        flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium
        transition-colors disabled:opacity-60 ${COLOR_MAP[color]}
      `}
    >
      {active ? (
        <>
          <span className="animate-spin">↻</span>
          <span>Opening…</span>
        </>
      ) : (
        <>
          <span>↓</span>
          <span>{label}</span>
          <span className="font-mono opacity-60">{sublabel}</span>
        </>
      )}
    </button>
  )
}
