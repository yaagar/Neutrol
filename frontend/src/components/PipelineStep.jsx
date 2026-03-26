// PipelineStep.jsx — a single step card in the pipeline builder UI
// Props:
//   step      {object}   — the step config object, e.g. { step: "bandpass", low_hz: 1, high_hz: 40 }
//   onRemove  {function} — called when the user clicks Remove on this card
// Stub for Session 1 — the pipeline builder UI is built in Session 2/8.

// Props documented above per CLAUDE.md code standards
export default function PipelineStep({ step, onRemove }) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 flex items-center justify-between">
      {/* Step name and parameters */}
      <div>
        <span className="font-mono text-purple-300 text-sm">{step.step}</span>
        <div className="text-xs text-gray-500 mt-1">
          {/* Render each parameter as key=value, filtering out the "step" key itself */}
          {Object.entries(step)
            .filter(([k]) => k !== 'step')
            .map(([k, v]) => `${k}=${v}`)
            .join('  ')}
        </div>
      </div>

      {/* Remove button — calls the parent's onRemove handler */}
      <button
        onClick={onRemove}
        className="text-xs text-red-400 hover:text-red-300 transition-colors"
      >
        Remove
      </button>
    </div>
  )
}
