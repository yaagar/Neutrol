// useSession.js — convenience hook for reading/writing the current session
//
// Why a custom hook instead of calling useAppStore directly in every component?
//   - Encapsulates the selector logic in one place. If the store shape changes,
//     we update this hook, not every component that uses session data.
//   - Makes component code read naturally: const { currentSession } = useSession()
//     instead of const currentSession = useAppStore(s => s.currentSession)
import useAppStore from '../store/appStore'

export function useSession() {
  // Each selector is kept separate so components only re-render when the
  // specific slice they care about changes — not when unrelated state updates.
  const currentSession   = useAppStore((s) => s.currentSession)
  const sessions         = useAppStore((s) => s.sessions)
  const setCurrentSession = useAppStore((s) => s.setCurrentSession)
  const setSessions      = useAppStore((s) => s.setSessions)
  const addSession       = useAppStore((s) => s.addSession)
  const removeSession    = useAppStore((s) => s.removeSession)

  return { currentSession, sessions, setCurrentSession, setSessions, addSession, removeSession }
}
