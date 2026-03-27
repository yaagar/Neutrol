// ExportPage.jsx — download trained models, pipeline configs, and inference scripts
//
// Two sections:
//   1. Trained Models — list of completed training jobs with download buttons for:
//        - Model weights (.pt file)
//        - Inference script (.py file, standalone, no Neutrol required)
//   2. Pipeline Configs — list of pipeline runs with download buttons for:
//        - Pipeline config (.json file, describes preprocessing steps)
//
// Downloads are triggered by setting window.location.href to the API URL.
// FastAPI responds with Content-Disposition: attachment which tells the browser
// to save the file rather than display it.
import { useState, useEffect } from 'react'

import { API_BASE_URL } from '../config'
const API = API_BASE_URL

export default function ExportPage() {
  const [jobs, setJobs]           = useState([])
  const [runs, setRuns]           = useState([])
  const [jobsLoading, setJobsLoading] = useState(false)
  const [runsLoading, setRunsLoading] = useState(false)
  const [downloading, setDownloading] = useState(null)  // 'model-{id}' | 'script-{id}' | 'pipeline-{id}'

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
        // Only show completed jobs — failed / running have no model file to download
        setJobs(json.data.jobs.filter(j => j.status === 'done'))
      }
    } catch { /* ignore */ }
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
  // The backend sets Content-Disposition: attachment so the browser saves it.
  function triggerDownload(url, key) {
    setDownloading(key)
    // Use a temporary hidden <a> element so we get the filename from
    // Content-Disposition (which window.open does not honour on all browsers).
    const a = document.createElement('a')
    a.href   = url
    a.target = '_blank'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    // Reset the download state after a short delay
    setTimeout(() => setDownloading(null), 1500)
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold text-gray-100 mb-1">Export</h1>
      <p className="text-gray-400 text-sm mb-8">
        Download your trained models, inference scripts, and pipeline configurations.
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
          <div className="flex flex-col gap-3">
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
// ModelCard — one trained model with two download buttons
// ---------------------------------------------------------------------------

function ModelCard({ job, downloading, onDownloadModel, onDownloadScript }) {
  const modelKey  = `model-${job.job_id}`
  const scriptKey = `script-${job.job_id}`

  const accuracy = job.final_val_acc != null
    ? `${(job.final_val_acc * 100).toFixed(1)}% val acc`
    : null

  const date = job.started_at
    ? new Date(job.started_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    : null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-start justify-between gap-4">
        {/* Left: model info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {/* Green dot = trained, ready to download */}
            <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" />
            <p className="text-sm font-medium text-gray-100 truncate">
              {job.model_name || <span className="text-gray-500 italic">Unnamed model</span>}
            </p>
            <span className="text-xs text-gray-600 font-mono flex-shrink-0">
              {job.job_id.slice(0, 8)}
            </span>
          </div>

          {/* Classes */}
          {job.classes?.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
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

          {/* Stats row */}
          <p className="text-xs text-gray-500">
            {[accuracy, date, `${job.epochs_run} epoch${job.epochs_run !== 1 ? 's' : ''}`]
              .filter(Boolean)
              .join(' · ')}
          </p>
        </div>

        {/* Right: download buttons */}
        <div className="flex flex-col gap-2 flex-shrink-0">
          <DownloadButton
            label="Model weights"
            sublabel=".pt"
            active={downloading === modelKey}
            onClick={onDownloadModel}
            color="purple"
          />
          <DownloadButton
            label="Inference script"
            sublabel=".py"
            active={downloading === scriptKey}
            onClick={onDownloadScript}
            color="blue"
          />
        </div>
      </div>

      {/* Inference script explanation */}
      <div className="mt-3 pt-3 border-t border-gray-800">
        <p className="text-xs text-gray-600">
          The inference script is standalone — it only requires{' '}
          <span className="font-mono text-gray-500">numpy</span> and{' '}
          <span className="font-mono text-gray-500">torch</span>. Keep it in the same
          folder as the .pt file and call{' '}
          <span className="font-mono text-gray-500">predict(window)</span> with a preprocessed
          numpy array of shape{' '}
          <span className="font-mono text-gray-500">(n_channels, n_samples)</span>.
        </p>
      </div>
    </div>
  )
}


// ---------------------------------------------------------------------------
// PipelineCard — one pipeline run with a download button
// ---------------------------------------------------------------------------

function PipelineCard({ run, downloading, onDownload }) {
  const key  = `pipeline-${run.run_id}`
  const date = run.created_at
    ? new Date(run.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    : null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-between gap-4">
      {/* Left: run info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <p className="text-sm font-medium text-gray-100 truncate">
            {run.name || <span className="text-gray-500 italic">Unnamed run</span>}
          </p>
          <span className="text-xs text-gray-600 font-mono flex-shrink-0">
            {run.run_id.slice(0, 8)}
          </span>
        </div>

        {/* Steps */}
        {run.steps_applied?.length > 0 && (
          <p className="text-xs text-gray-500 mb-1">
            {run.steps_applied.join(' → ')}
          </p>
        )}

        {/* Stats */}
        <p className="text-xs text-gray-600">
          {[
            `${run.n_windows} windows`,
            run.window_shape ? `${run.window_shape[0]}ch × ${run.window_shape[1]}samp` : null,
            date,
          ].filter(Boolean).join(' · ')}
        </p>
      </div>

      {/* Right: download button */}
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
