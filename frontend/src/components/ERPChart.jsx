// ERPChart.jsx — Event-Related Potential visualization
//
// Displays the average EEG waveform per class label, time-locked to window onset.
// Each class gets a distinct colored line. This lets users visually identify
// ERP components like P300 (~300ms positive peak) and N200 (~200ms negative deflection).
//
// Data shape expected from POST /api/analysis/erp:
//   {
//     classes:   ["left", "right"],
//     time_ms:   [0, 3.9, 7.8, ...],
//     channels:  ["Fp1", "Fp2", ...],
//     averages:  { "left": { "Fp1": [...], ... }, "right": { ... } },
//     n_windows: { "left": 45, "right": 42 }
//   }

import { useMemo, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

// Distinct colors for up to 8 classes — contrasted against the dark background
const CLASS_COLORS = [
  '#60a5fa', // blue
  '#f97316', // orange
  '#34d399', // green
  '#f472b6', // pink
  '#a78bfa', // purple
  '#fbbf24', // amber
  '#22d3ee', // cyan
  '#fb7185', // rose
]

export default function ERPChart({ data, loading, error }) {
  const [selectedChannel, setSelectedChannel] = useState(null)

  // Pick the channel to display (defaults to first channel)
  const channel = selectedChannel || (data?.channels?.[0] ?? null)

  // Build Recharts data array: [{time_ms, "left": val, "right": val, ...}, ...]
  const chartData = useMemo(() => {
    if (!data || !channel) return []
    const { time_ms, classes, averages } = data
    return time_ms.map((t, i) => {
      const point = { time_ms: Math.round(t) }
      for (const cls of classes) {
        const vals = averages?.[cls]?.[channel]
        point[cls] = vals ? parseFloat(vals[i].toFixed(4)) : null
      }
      return point
    })
  }, [data, channel])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400">
        Computing ERPs…
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-48 text-red-400 text-sm px-4 text-center">
        {error}
      </div>
    )
  }

  if (!data || data.classes.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm text-center px-4">
        No ERP data. Label windows on the Label page first, then click Analyze.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Channel selector */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-gray-400">Channel:</span>
        <div className="flex flex-wrap gap-1">
          {data.channels.map(ch => (
            <button
              key={ch}
              onClick={() => setSelectedChannel(ch)}
              className={`px-2 py-0.5 rounded text-xs border transition-colors ${
                ch === channel
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-500'
              }`}
            >
              {ch}
            </button>
          ))}
        </div>

        {/* Window count badges */}
        <div className="flex flex-wrap gap-2 ml-auto">
          {data.classes.map((cls, i) => (
            <span
              key={cls}
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                backgroundColor: CLASS_COLORS[i % CLASS_COLORS.length] + '30',
                color: CLASS_COLORS[i % CLASS_COLORS.length],
                border: `1px solid ${CLASS_COLORS[i % CLASS_COLORS.length]}60`,
              }}
            >
              {cls} (n={data.n_windows[cls]})
            </span>
          ))}
        </div>
      </div>

      {/* ERP line chart */}
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData} margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="time_ms"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={v => `${v}ms`}
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            label={{ value: 'Time (ms)', position: 'insideBottomRight', offset: -4, fill: '#6b7280', fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            label={{ value: 'Amplitude (µV)', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }}
          />
          {/* Vertical zero-time reference line */}
          <ReferenceLine x={0} stroke="#4b5563" strokeDasharray="4 2" />
          {/* Horizontal zero-amplitude reference line */}
          <ReferenceLine y={0} stroke="#374151" />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 6 }}
            labelStyle={{ color: '#9ca3af', fontSize: 11 }}
            itemStyle={{ fontSize: 11 }}
            labelFormatter={v => `${v} ms`}
            formatter={(val, name) => [typeof val === 'number' ? val.toFixed(3) : val, name]}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          {data.classes.map((cls, i) => (
            <Line
              key={cls}
              type="monotone"
              dataKey={cls}
              stroke={CLASS_COLORS[i % CLASS_COLORS.length]}
              strokeWidth={1.8}
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <p className="text-xs text-gray-500">
        Averaged across {Object.values(data.n_windows).reduce((a, b) => a + b, 0)} total windows.
        Each line is the mean signal for that class — noise cancels out across trials.
      </p>
    </div>
  )
}
