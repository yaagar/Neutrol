// appStore.js — Zustand global state store
//
// Why Zustand over React Context or Redux?
//   - Redux: too much boilerplate for a 6-page app with simple shared state
//   - Context: re-renders every consumer on every state change; causes performance issues
//     when the EEG signal data updates frequently (live chart, labeling)
//   - Zustand: minimal API, components subscribe to only the slices they need,
//     no provider wrapping required
//
// This store is the single source of truth for all cross-page state.
// Pages read from it via selectors, and write to it via actions defined below.
import { create } from 'zustand'

const useAppStore = create((set) => ({
  // --- State fields (exactly as specified in CLAUDE.md) ---

  // The currently active session metadata (a DataSession summary from the API).
  // null means no file has been loaded yet.
  currentSession: null,

  // All loaded sessions this run — allows switching between files without re-uploading.
  // Each entry is a DataSession summary dict (no signal arrays — those stay on the backend).
  sessions: [],

  // The current pipeline configuration dict (PipelineConfig JSON format from CLAUDE.md).
  // null means the user hasn't built or loaded a pipeline yet.
  pipelineConfig: null,

  // Labeled windows for the current session — list of LabeledWindow objects.
  // Populated after the user runs the labeling workflow in Session 3.
  labeledData: [],

  // The active training job — contains job_id, status, epoch progress, metrics.
  // null when no training has been started.
  trainingJob: null,

  // --- Actions (functions that update the state) ---

  // Set (or clear) the current session. Called after a successful file upload.
  setCurrentSession: (session) => set({ currentSession: session }),

  // Replace the full sessions list. Called when /api/data/sessions is fetched.
  setSessions: (sessions) => set({ sessions }),

  // Add a single session to the list without replacing all others.
  addSession: (session) =>
    set((state) => ({ sessions: [...state.sessions, session] })),

  // Remove a session by its ID. Called after DELETE /api/data/{session_id}.
  removeSession: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.session_id !== sessionId),
      // If the deleted session was the current one, clear it
      currentSession:
        state.currentSession?.session_id === sessionId ? null : state.currentSession,
    })),

  // Save the pipeline config the user has built.
  setPipelineConfig: (config) => set({ pipelineConfig: config }),

  // Replace all labeled data for the current session.
  setLabeledData: (data) => set({ labeledData: data }),

  // Update the active training job (called repeatedly as WebSocket progress comes in).
  setTrainingJob: (job) => set({ trainingJob: job }),
}))

export default useAppStore
