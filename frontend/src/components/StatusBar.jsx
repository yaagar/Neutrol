// StatusBar.jsx — top-of-page session info bar
// Reads the currentSession from the Zustand store and displays key metadata.
// Stub for Session 1 — will be wired to real data in Session 2.
import { useSession } from '../hooks/useSession'

export default function StatusBar() {
  // Pull the active session from global Zustand store
  const { currentSession } = useSession()

  return (
    <div className="bg-gray-800 border-b border-gray-700 px-6 py-2 flex items-center gap-6 text-sm text-gray-400">
      <span className="font-medium text-gray-200">Session:</span>
      {currentSession ? (
        // When a session is loaded, show its key properties
        <>
          <span>{currentSession.session_id?.slice(0, 8)}…</span>
          <span>{currentSession.channel_names?.length} channels</span>
          <span>{currentSession.sample_rate} Hz</span>
          <span>{currentSession.duration_seconds?.toFixed(1)}s</span>
        </>
      ) : (
        // No session yet — prompt the user to go to Import
        <span className="text-gray-600">No session loaded — go to Import</span>
      )}
    </div>
  )
}
