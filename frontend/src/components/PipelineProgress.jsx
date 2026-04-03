// PipelineProgress.jsx — Visual 5-stage pipeline progress indicator
//
// Shows the EEG development workflow stages as connected circles.
// Each stage lights up as the user completes the corresponding step.
// Reads completion state from Zustand (same source as Sidebar.jsx).
//
// Stages:
//   1. Import     — a session is loaded
//   2. Preprocess — a pipeline run exists (activeRunId)
//   3. Label      — any labeled windows exist
//   4. Train      — any training job is done
//   5. Export     — always available (gray — no completion signal)

import useAppStore from '../store/appStore'

const STAGES = [
  { label: 'Import',     short: '1' },
  { label: 'Preprocess', short: '2' },
  { label: 'Label',      short: '3' },
  { label: 'Train',      short: '4' },
  { label: 'Export',     short: '5' },
]

export default function PipelineProgress() {
  const currentSession  = useAppStore(s => s.currentSession)
  const labeledData     = useAppStore(s => s.labeledData)
  const trainingJob     = useAppStore(s => s.trainingJob)
  const preprocessState = useAppStore(s => s.preprocessPageState)

  // Which stages are complete
  const done = [
    currentSession !== null,
    preprocessState.activeRunId !== null,
    labeledData.length > 0,
    trainingJob?.status === 'done',
    false,  // Export has no completion signal
  ]

  return (
    <div className="px-1 py-3 border-t border-gray-800">
      <p className="text-xs text-gray-600 mb-2 font-medium uppercase tracking-wide">Progress</p>
      <div className="flex items-center">
        {STAGES.map((stage, i) => (
          <div key={stage.label} className="flex items-center">
            {/* Stage circle */}
            <div className="flex flex-col items-center">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
                  done[i]
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-800 border border-gray-600 text-gray-500'
                }`}
                title={stage.label}
              >
                {done[i] ? '✓' : stage.short}
              </div>
              <span className={`text-xs mt-0.5 ${done[i] ? 'text-green-500' : 'text-gray-600'}`}
                style={{ fontSize: '9px' }}>
                {stage.label}
              </span>
            </div>

            {/* Connector line between stages */}
            {i < STAGES.length - 1 && (
              <div
                className={`h-0.5 w-3 mb-3 transition-colors ${
                  done[i] ? 'bg-green-600' : 'bg-gray-700'
                }`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
