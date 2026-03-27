// ImportPage.jsx — EEG file upload, session manager, and waveform preview
//
// This page is the entry point of the entire workflow.
// Three sections stacked vertically:
//   1. Upload zone — drop or click to select a .csv or .edf file
//   2. Session list — all loaded sessions; click one to make it active
//   3. Waveform preview — chart of the active session's signals
import { useState, useEffect, useRef } from 'react'
import { useSession } from '../hooks/useSession'
import WaveformChart from '../components/WaveformChart'

import { API_BASE_URL } from '../config'
const API = API_BASE_URL

export default function ImportPage() {
  // Read session state and write actions from the Zustand store
  const { currentSession, sessions, setCurrentSession, setSessions, addSession, removeSession } =
    useSession()

  // Local state for this page only (not needed by other pages, so not in Zustand)
  const [uploading, setUploading]       = useState(false)   // is a file upload in progress?
  const [uploadError, setUploadError]   = useState(null)    // error string from last upload
  const [uploadSuccess, setUploadSuccess] = useState(false) // show green confirmation briefly
  const [isDragOver, setIsDragOver]     = useState(false)   // drag-and-drop hover state
  const [previewData, setPreviewData]   = useState(null)    // waveform chart data for current session
  const [previewLoading, setPreviewLoading] = useState(false)
  const [visibleChannels, setVisibleChannels] = useState([]) // which channels to show in chart
  const fileInputRef = useRef(null)                         // hidden <input type="file"> ref

  // On mount: load all existing sessions from the backend.
  // This handles the case where the user navigates away and comes back —
  // sessions stay in the backend memory even if Zustand is reset on hot reload.
  useEffect(() => {
    fetchSessions()
  }, [])

  // When the active session changes, fetch its waveform preview data.
  useEffect(() => {
    if (currentSession) {
      fetchPreview(currentSession.session_id)
      // Default: show all channels
      setVisibleChannels(currentSession.channel_names)
    } else {
      setPreviewData(null)
      setVisibleChannels([])
    }
  }, [currentSession?.session_id])

  // -------------------------------------------------------------------------
  // API helpers — all use async/await as required by CLAUDE.md
  // -------------------------------------------------------------------------

  async function fetchSessions() {
    try {
      const res = await fetch(`${API}/api/data/sessions`)
      const json = await res.json()
      if (json.success) {
        setSessions(json.data)
        // If there's no current session but sessions exist, auto-select the first one
        if (!currentSession && json.data.length > 0) {
          setCurrentSession(json.data[0])
        }
      }
    } catch {
      // Network error — backend likely not running. Don't crash, just show nothing.
    }
  }

  async function fetchPreview(sessionId) {
    setPreviewLoading(true)
    try {
      // max_samples=512 keeps chart rendering fast even for long recordings
      const res = await fetch(`${API}/api/data/${sessionId}/preview?max_samples=512`)
      const json = await res.json()
      if (json.success) {
        setPreviewData(json.data)
      }
    } catch {
      setPreviewData(null)
    } finally {
      setPreviewLoading(false)
    }
  }

  async function uploadFile(file) {
    // Validate extension before sending — gives instant feedback, avoids wasted round-trip
    const ext = file.name.split('.').pop().toLowerCase()
    if (ext !== 'csv' && ext !== 'edf') {
      setUploadError(`Unsupported file type ".${ext}". Please upload a .csv or .edf file.`)
      return
    }

    setUploading(true)
    setUploadError(null)
    setUploadSuccess(false)

    // FormData is required for multipart file uploads
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/api/data/upload`, { method: 'POST', body: form })
      const json = await res.json()

      if (json.success) {
        // Add the new session to the Zustand store and make it the active session
        addSession(json.data)
        setCurrentSession(json.data)
        setUploadSuccess(true)
        // Clear the success message after 3 seconds
        setTimeout(() => setUploadSuccess(false), 3000)
      } else {
        setUploadError(json.error || 'Upload failed.')
      }
    } catch {
      setUploadError('Could not reach the backend. Is it running on port 8000?')
    } finally {
      setUploading(false)
    }
  }

  async function deleteSession(sessionId) {
    try {
      const res = await fetch(`${API}/api/data/${sessionId}`, { method: 'DELETE' })
      const json = await res.json()
      if (json.success) {
        // Update Zustand — removeSession also clears currentSession if it was the deleted one
        removeSession(sessionId)
      }
    } catch {
      // Silently ignore delete errors — worst case the session stays listed
    }
  }

  // -------------------------------------------------------------------------
  // Drag-and-drop handlers
  // -------------------------------------------------------------------------

  function onDragOver(e) {
    e.preventDefault()   // required to allow drop
    setIsDragOver(true)
  }

  function onDragLeave() {
    setIsDragOver(false)
  }

  function onDrop(e) {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }

  function onFileInputChange(e) {
    const file = e.target.files[0]
    if (file) uploadFile(file)
    // Reset the input so the same file can be re-uploaded if needed
    e.target.value = ''
  }

  // Toggle a channel on/off in the waveform chart
  function toggleChannel(name) {
    setVisibleChannels(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    )
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold text-gray-100 mb-1">Import</h1>
      <p className="text-gray-400 text-sm mb-8">
        Upload a .csv or .edf EEG file to begin. The file is loaded into memory on the backend.
      </p>

      {/* ------------------------------------------------------------------ */}
      {/* Section 1: Upload zone                                               */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-10">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Upload File
        </h2>

        {/* The visible drop zone — clicking it triggers the hidden file input */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={`
            border-2 border-dashed rounded-xl p-10 text-center cursor-pointer
            transition-colors duration-150 select-none
            ${isDragOver
              ? 'border-purple-400 bg-purple-900/20'
              : 'border-gray-700 hover:border-gray-500 bg-gray-900/40'
            }
          `}
        >
          {uploading ? (
            <p className="text-gray-300">Uploading and parsing file…</p>
          ) : (
            <>
              <p className="text-gray-300 font-medium">Drop a file here, or click to browse</p>
              <p className="text-gray-600 text-sm mt-1">Supported: .csv, .edf</p>
            </>
          )}
        </div>

        {/* Hidden file input — triggered by clicking the drop zone above */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.edf"
          className="hidden"
          onChange={onFileInputChange}
        />

        {/* Upload success message */}
        {uploadSuccess && (
          <p className="mt-3 text-green-400 text-sm">
            File uploaded successfully. Session created.
          </p>
        )}

        {/* Upload error message */}
        {uploadError && (
          <p className="mt-3 text-red-400 text-sm">{uploadError}</p>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 2: Session list                                              */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-10">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Loaded Sessions ({sessions.length})
        </h2>

        {sessions.length === 0 ? (
          <p className="text-gray-600 text-sm">
            No sessions loaded. Upload a file above to get started.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {sessions.map(session => {
              const isActive = currentSession?.session_id === session.session_id
              return (
                <li
                  key={session.session_id}
                  className={`
                    rounded-lg border px-4 py-3 flex items-center justify-between gap-4
                    transition-colors
                    ${isActive
                      ? 'border-purple-500 bg-purple-900/20'
                      : 'border-gray-700 bg-gray-900/40 hover:border-gray-600'
                    }
                  `}
                >
                  {/* Session metadata */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {isActive && (
                        <span className="text-xs bg-purple-600 text-white px-2 py-0.5 rounded-full font-medium">
                          Active
                        </span>
                      )}
                      {/* Show filename from file_path, falling back to truncated session ID */}
                      <span className="text-gray-200 text-sm font-mono truncate">
                        {session.file_path
                          ? session.file_path.split(/[\\/]/).pop()
                          : session.session_id.slice(0, 8) + '…'}
                      </span>
                    </div>
                    <div className="flex gap-4 text-xs text-gray-500">
                      <span>{session.n_channels} channels</span>
                      <span>{session.sample_rate} Hz</span>
                      <span>{session.duration_seconds?.toFixed(1)}s</span>
                      <span className="text-gray-600 font-mono">{session.session_id.slice(0, 8)}…</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 shrink-0">
                    {!isActive && (
                      <button
                        onClick={() => setCurrentSession(session)}
                        className="text-xs text-purple-400 hover:text-purple-300 transition-colors px-2 py-1 border border-purple-700 rounded hover:border-purple-500"
                      >
                        Set Active
                      </button>
                    )}
                    <button
                      onClick={() => deleteSession(session.session_id)}
                      className="text-xs text-red-500 hover:text-red-400 transition-colors px-2 py-1 border border-red-900 rounded hover:border-red-700"
                    >
                      Delete
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 3: Waveform preview (only shown when a session is active)   */}
      {/* ------------------------------------------------------------------ */}
      {currentSession && (
        <section>
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Waveform Preview — {currentSession.file_path?.split(/[\\/]/).pop()}
          </h2>

          {/* Channel toggles — let the user show/hide individual channels */}
          {currentSession.channel_names && (
            <div className="flex flex-wrap gap-2 mb-4">
              {currentSession.channel_names.map(name => (
                <button
                  key={name}
                  onClick={() => toggleChannel(name)}
                  className={`
                    text-xs px-2 py-1 rounded border transition-colors font-mono
                    ${visibleChannels.includes(name)
                      ? 'bg-purple-700 border-purple-500 text-white'
                      : 'bg-transparent border-gray-700 text-gray-500 hover:border-gray-500'
                    }
                  `}
                >
                  {name}
                </button>
              ))}
            </div>
          )}

          {/* Chart area */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            {previewLoading ? (
              <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
                Loading waveform…
              </div>
            ) : previewData ? (
              <WaveformChart
                data={previewData.points}
                channelNames={visibleChannels}
                height={280}
              />
            ) : (
              <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
                Could not load waveform data.
              </div>
            )}
          </div>

          {/* Preview metadata */}
          {previewData && (
            <p className="text-xs text-gray-600 mt-2">
              Showing {previewData.n_samples_preview} of {previewData.n_samples_original} samples
              (downsampled for display)
            </p>
          )}
        </section>
      )}
    </div>
  )
}
