// Sidebar.jsx — left navigation panel
//
// Shows the app name, a numbered list of workflow steps, and a small
// status dot on each step so the user can see at a glance how far through
// the workflow they are.
//
// Status dot colours:
//   green  — this step has been completed (derived from Zustand state)
//   purple — this is the currently active page
//   gray   — not yet reached
import { NavLink } from 'react-router-dom'
import useAppStore from '../store/appStore'
import PipelineProgress from './PipelineProgress'

const NAV_ITEMS = [
  { label: 'Import',     path: '/import',     step: 1 },
  { label: 'Preprocess', path: '/preprocess', step: 2 },
  { label: 'Label',      path: '/label',      step: 3 },
  { label: 'Train',      path: '/train',      step: 4 },
  { label: 'Inference',  path: '/inference',  step: 5 },
  { label: 'Export',     path: '/export',     step: 6 },
]

export default function Sidebar() {
  const currentSession     = useAppStore(s => s.currentSession)
  const labeledData        = useAppStore(s => s.labeledData)
  const trainingJob        = useAppStore(s => s.trainingJob)
  const preprocessState    = useAppStore(s => s.preprocessPageState)

  // Derive which steps are "done" based on what exists in the store
  const done = {
    '/import':     currentSession !== null,
    '/preprocess': preprocessState.activeRunId !== null,
    '/label':      labeledData.length > 0,
    '/train':      trainingJob?.status === 'done',
    '/inference':  false,
    '/export':     false,
  }

  return (
    <nav className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col py-6 px-4 shrink-0">

      {/* App title */}
      <div className="mb-8">
        <h1 className="text-xl font-bold text-purple-400">Neutrol</h1>
        <p className="text-xs text-gray-500 mt-1">EEG Development Platform</p>
      </div>

      {/* Pipeline progress indicator */}
      <PipelineProgress />

      {/* Navigation links */}
      <ul className="flex flex-col gap-1 mt-2">
        {NAV_ITEMS.map(({ label, path, step }) => (
          <li key={path}>
            <NavLink
              to={path}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {/* Step number */}
                  <span className={`text-xs w-4 shrink-0 ${isActive ? 'text-purple-200' : 'text-gray-600'}`}>
                    {step}
                  </span>

                  {/* Label */}
                  <span className="flex-1">{label}</span>

                  {/* Status dot — green if done, hidden if active (purple bg says it all) */}
                  {!isActive && done[path] && (
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
                  )}
                </>
              )}
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Session info at the bottom */}
      {currentSession && (
        <div className="mt-auto pt-4 border-t border-gray-800">
          <p className="text-xs text-gray-600 truncate">
            {currentSession.file_path?.split(/[\\/]/).pop() || 'Session loaded'}
          </p>
          <p className="text-xs text-gray-700">
            {currentSession.n_channels}ch · {currentSession.sample_rate}Hz
          </p>
        </div>
      )}
    </nav>
  )
}
