// ImportPage.jsx — EEG file upload, session manager, and waveform preview
//
// Three sections:
//   1. Upload zone — drop or click to select a .csv or .edf file
//   2. Session list — all loaded sessions; click to make active, pencil to rename
//   3. Waveform preview — chart of the active session's signals with channel toggles
import { useState, useEffect, useRef } from 'react'
import { useSession } from '../hooks/useSession'
import useAppStore from '../store/appStore'
import WaveformChart from '../components/WaveformChart'

import { API_BASE_URL } from '../config'
const API = API_BASE_URL

export default function ImportPage() {
  const { currentSession, sessions, setCurrentSession, setSessions, addSession, removeSession } =
    useSession()
  const sessionNicknames   = useAppStore(s => s.sessionNicknames)
  const setSessionNickname = useAppStore(s => s.setSessionNickname)

  const [uploading, setUploading]           = useState(false)
  const [uploadError, setUploadError]       = useState(null)
  const [uploadSuccess, setUploadSuccess]   = useState(false)
  const [isDragOver, setIsDragOver]         = useState(false)
  const [previewData, setPreviewData]       = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [visibleChannels, setVisibleChannels] = useState([])

  // Which session is currently being renamed (its session_id), and the draft text
  const [renamingId, setRenamingId]     = useState(null)
  const [renameDraft, setRenameDraft]   = useState('')
  const renameInputRef                  = useRef(null)

  const fileInputRef = useRef(null)

  useEffect(() => { fetchSessions() }, [])

  useEffect(() => {
    if (currentSession) {
      fetchPreview(currentSession.session_id)
      setVisibleChannels(currentSession.channel_names)
    } else {
      setPreviewData(null)
      setVisibleChannels([])
    }
  }, [currentSession?.session_id])

  // Focus the rename input as soon as it appears
  useEffect(() => {
    if (renamingId) renameInputRef.current?.focus()
  }, [renamingId])

  // -------------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------------

  async function fetchSessions() {
    try {
      const res  = await fetch(`${API}/api/data/sessions`)
      const json = await res.json()
      if (json.success) {
        setSessions(json.data)
        if (!currentSession && json.data.length > 0) setCurrentSession(json.data[0])
      }
    } catch { /* backend not running — show nothing */ }
  }

  async function fetchPreview(sessionId) {
    setPreviewLoading(true)
    try {
      const res  = await fetch(`${API}/api/data/${sessionId}/preview?max_samples=512`)
      const json = await res.json()
      if (json.success) setPreviewData(json.data)
    } catch { setPreviewData(null) }
    finally { setPreviewLoading(false) }
  }

  async function uploadFile(file) {
    const ext = file.name.split('.').pop().toLowerCase()
    if (ext !== 'csv' && ext !== 'edf') {
      setUploadError(`Unsupported file type ".${ext}". Please upload a .csv or .edf file.`)
      return
    }
    setUploading(true)
    setUploadError(null)
    setUploadSuccess(false)

    const form = new FormData()
    form.append('file', file)

    try {
      const res  = await fetch(`${API}/api/data/upload`, { method: 'POST', body: form })
      const json = await res.json()
      if (json.success) {
        addSession(json.data)
        setCurrentSession(json.data)
        setUploadSuccess(true)
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
      const res  = await fetch(`${API}/api/data/${sessionId}`, { method: 'DELETE' })
      const json = await res.json()
      if (json.success) removeSession(sessionId)
    } catch { /* ignore */ }
  }

  // -------------------------------------------------------------------------
  // Rename helpers
  // -------------------------------------------------------------------------

  function startRename(session, e) {
    e.stopPropagation()
    const current = sessionNicknames[session.session_id] ||
      session.file_path?.split(/[\\/]/).pop() || ''
    setRenamingId(session.session_id)
    setRenameDraft(current)
  }

  function commitRename() {
    if (renamingId) setSessionNickname(renamingId, renameDraft.trim())
    setRenamingId(null)
  }

  function onRenameKeyDown(e) {
    if (e.key === 'Enter') commitRename()
    if (e.key === 'Escape') setRenamingId(null)
  }

  // -------------------------------------------------------------------------
  // Drag-and-drop
  // -------------------------------------------------------------------------

  function onDragOver(e)  { e.preventDefault(); setIsDragOver(true) }
  function onDragLeave()  { setIsDragOver(false) }
  function onDrop(e)      { e.preventDefault(); setIsDragOver(false); const f = e.dataTransfer.files[0]; if (f) uploadFile(f) }
  function onFileInputChange(e) { const f = e.target.files[0]; if (f) uploadFile(f); e.target.value = '' }
  function toggleChannel(name) {
    setVisibleChannels(prev => prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name])
  }

  // Helper: display name for a session (nickname → filename → id)
  function displayName(session) {
    return sessionNicknames[session.session_id] ||
      session.file_path?.split(/[\\/]/).pop() ||
      session.session_id.slice(0, 8) + '…'
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
            <div>
              <div className="flex justify-center mb-2">
                <span className="animate-spin text-purple-400 text-2xl">↻</span>
              </div>
              <p className="text-gray-300">Uploading and parsing file…</p>
            </div>
          ) : (
            <>
              <p className="text-gray-300 font-medium">Drop a file here, or click to browse</p>
              <p className="text-gray-600 text-sm mt-1">Supported: .csv, .edf</p>
            </>
          )}
        </div>

        <input ref={fileInputRef} type="file" accept=".csv,.edf" className="hidden" onChange={onFileInputChange} />

        {uploadSuccess && (
          <p className="mt-3 text-green-400 text-sm">File uploaded successfully. Session created.</p>
        )}
        {uploadError && (
          <div className="mt-3 bg-red-900/20 border border-red-800 rounded-lg p-3">
            <p className="text-red-400 text-sm">{uploadError}</p>
            <button onClick={() => setUploadError(null)} className="text-xs text-red-500 hover:text-red-400 mt-1">Dismiss</button>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Section 2: Session list                                              */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-10">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
            Loaded Sessions ({sessions.length})
          </h2>
          {sessions.length > 0 && (
            <button onClick={fetchSessions} className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
              Refresh
            </button>
          )}
        </div>

        {sessions.length === 0 ? (
          <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-8 text-center">
            <p className="text-gray-500 text-sm">No sessions loaded yet.</p>
            <p className="text-gray-600 text-xs mt-1">Upload a .csv or .edf file above to get started.</p>
          </div>
        ) : (
          <ul className="flex flex-col gap-2">
            {sessions.map(session => {
              const isActive  = currentSession?.session_id === session.session_id
              const isRenaming = renamingId === session.session_id

              return (
                <li
                  key={session.session_id}
                  onClick={() => !isRenaming && setCurrentSession(session)}
                  className={`
                    rounded-lg border px-4 py-3 flex items-center justify-between gap-4
                    transition-colors cursor-pointer
                    ${isActive
                      ? 'border-purple-500 bg-purple-900/20'
                      : 'border-gray-700 bg-gray-900/40 hover:border-gray-600'
                    }
                  `}
                >
                  {/* Left: name + stats */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {isActive && (
                        <span className="text-xs bg-purple-600 text-white px-2 py-0.5 rounded-full font-medium shrink-0">
                          Active
                        </span>
                      )}

                      {/* Inline rename input or display name */}
                      {isRenaming ? (
                        <input
                          ref={renameInputRef}
                          value={renameDraft}
                          onChange={e => setRenameDraft(e.target.value)}
                          onBlur={commitRename}
                          onKeyDown={onRenameKeyDown}
                          onClick={e => e.stopPropagation()}
                          className="flex-1 bg-gray-700 border border-purple-500 text-gray-100 text-sm rounded px-2 py-0.5 focus:outline-none"
                        />
                      ) : (
                        <span className="text-gray-200 text-sm font-medium truncate">
                          {displayName(session)}
                        </span>
                      )}

                      {/* Pencil icon to start renaming */}
                      {!isRenaming && (
                        <button
                          onClick={e => startRename(session, e)}
                          title="Rename session"
                          className="text-gray-600 hover:text-gray-400 transition-colors shrink-0 text-xs"
                        >
                          ✎
                        </button>
                      )}
                    </div>

                    <div className="flex gap-4 text-xs text-gray-500">
                      <span>{session.n_channels} channels</span>
                      <span>{session.sample_rate} Hz</span>
                      <span>{session.duration_seconds?.toFixed(1)}s</span>
                      <span className="text-gray-700 font-mono">{session.session_id.slice(0, 8)}…</span>
                    </div>
                  </div>

                  {/* Right: actions */}
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={e => { e.stopPropagation(); deleteSession(session.session_id) }}
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
      {/* Section 3: Waveform preview                                          */}
      {/* ------------------------------------------------------------------ */}
      {currentSession && (
        <section>
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Waveform Preview — {displayName(currentSession)}
          </h2>

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

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            {previewLoading ? (
              <div className="flex items-center justify-center h-48 text-gray-500 text-sm gap-2">
                <span className="animate-spin">↻</span>
                <span>Loading waveform…</span>
              </div>
            ) : previewData ? (
              <WaveformChart data={previewData.points} channelNames={visibleChannels} height={280} />
            ) : (
              <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
                Could not load waveform data.
              </div>
            )}
          </div>

          {previewData && (
            <p className="text-xs text-gray-600 mt-2">
              Showing {previewData.n_samples_preview} of {previewData.n_samples_original} samples (downsampled for display)
            </p>
          )}
        </section>
      )}
    </div>
  )
}
