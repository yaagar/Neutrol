// WaveformChart.jsx — EEG waveform display with MATLAB-like zoom/pan
//
// Props:
//   data         {Array}  — [{time: 0.0, Fp1: 0.12, ...}]
//   channelNames {Array}  — channel names to plot
//   height       {number} — chart height in pixels (default 300)
//   showBrush    {bool}   — show Recharts Brush scroll bar (default true for long signals)
//   onRangeChange {fn}    — called with {startSec, endSec} when user applies a time range
//   totalDuration {number} — total recording duration in seconds (for range inputs)

import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Brush,
} from 'recharts'

const CHANNEL_COLORS = [
  '#a78bfa', '#34d399', '#fb923c', '#60a5fa',
  '#f472b6', '#facc15', '#4ade80', '#f87171',
  '#38bdf8', '#fb7185', '#a3e635', '#c084fc',
]

// Amplitude multipliers for Y-axis zoom
const AMP_STEPS = [1, 2, 5, 10]

export default function WaveformChart({
  data = [],
  channelNames = [],
  height = 300,
  showBrush = false,
  onRangeChange = null,
  totalDuration = null,
}) {
  const [ampIdx, setAmpIdx]       = useState(0)   // index into AMP_STEPS
  const [startSec, setStartSec]   = useState('')
  const [endSec, setEndSec]       = useState('')

  const ampMult = AMP_STEPS[ampIdx]

  // Apply time range — call parent callback or no-op if not provided
  function applyRange() {
    if (!onRangeChange) return
    const s = parseFloat(startSec)
    const e = parseFloat(endSec)
    if (!isNaN(s) && !isNaN(e) && e > s) {
      onRangeChange({ startSec: s, endSec: e })
    }
  }

  function resetRange() {
    setStartSec('')
    setEndSec('')
    if (onRangeChange) onRangeChange({ startSec: 0, endSec: totalDuration ?? 9999 })
  }

  return (
    <div>
      {/* Controls row: time range + amplitude zoom */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        {/* Time range inputs — only shown if parent supports range changes */}
        {onRangeChange && (
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <span>From</span>
            <input
              type="number"
              value={startSec}
              onChange={e => setStartSec(e.target.value)}
              placeholder="0"
              className="w-14 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-gray-200 text-xs"
            />
            <span>s  To</span>
            <input
              type="number"
              value={endSec}
              onChange={e => setEndSec(e.target.value)}
              placeholder={totalDuration ?? '—'}
              className="w-14 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-gray-200 text-xs"
            />
            <span>s</span>
            <button
              onClick={applyRange}
              className="px-2 py-0.5 bg-purple-700 hover:bg-purple-600 text-white rounded text-xs"
            >Apply</button>
            <button
              onClick={resetRange}
              className="px-2 py-0.5 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded text-xs"
            >Reset</button>
          </div>
        )}

        {/* Amplitude zoom */}
        <div className="flex items-center gap-1 text-xs text-gray-500 ml-auto">
          <span>Amp</span>
          {AMP_STEPS.map((step, i) => (
            <button
              key={step}
              onClick={() => setAmpIdx(i)}
              className={`px-1.5 py-0.5 rounded text-xs ${
                ampIdx === i
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              ×{step}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: showBrush ? 20 : 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 11 }} />
          <YAxis
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
            // Apply amplitude zoom by scaling the domain
            domain={([min, max]) => [min / ampMult, max / ampMult]}
          />
          <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />

          {channelNames.map((name, idx) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={CHANNEL_COLORS[idx % CHANNEL_COLORS.length]}
              dot={false}
              strokeWidth={1.5}
            />
          ))}

          {/* Brush scroll bar — lets user pan through long recordings */}
          {showBrush && data.length > 0 && (
            <Brush
              dataKey="time"
              height={16}
              stroke="#4b5563"
              fill="#1f2937"
              travellerWidth={6}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
