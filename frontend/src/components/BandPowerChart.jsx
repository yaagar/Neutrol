// BandPowerChart.jsx — Delta/Theta/Alpha/Beta/Gamma power bar chart
//
// Shows band power per channel as a grouped bar chart.
// X axis = EEG frequency bands, grouped bars = channels.
// This is the standard way to compare brain states and channels at a glance.
//
// Props:
//   data           - API response from /api/analysis/bandpower
//                    Shape: { channels: string[], bands: { [bandName]: number[] } }
//   activeChannels - Set of channel names to show (all shown if null)

import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'

// Color palette for channels
const CHANNEL_COLORS = [
  '#60a5fa', '#34d399', '#f87171', '#fbbf24', '#a78bfa',
  '#fb923c', '#38bdf8', '#4ade80', '#f472b6', '#facc15',
  '#c084fc', '#2dd4bf', '#fb7185', '#86efac', '#93c5fd',
  '#fcd34d',
]

// Human-readable band labels with their Hz ranges for the tooltip
const BAND_LABELS = {
  delta: 'Delta (0.5–4 Hz)',
  theta: 'Theta (4–8 Hz)',
  alpha: 'Alpha (8–13 Hz)',
  beta:  'Beta (13–30 Hz)',
  gamma: 'Gamma (30–100 Hz)',
}

export default function BandPowerChart({ data, activeChannels = null }) {
  // Build Recharts format: [{ band: 'Delta', Fp1: 0.12, Fp2: 0.09, ... }, ...]
  const { chartData, channelNames } = useMemo(() => {
    if (!data || !data.bands || !data.channels) {
      return { chartData: [], channelNames: [] }
    }

    const names = activeChannels
      ? data.channels.filter(ch => activeChannels.has(ch))
      : data.channels

    const points = Object.entries(data.bands).map(([bandName, values]) => {
      const point = { band: bandName.charAt(0).toUpperCase() + bandName.slice(1) }
      names.forEach((ch, i) => {
        const chIdx = data.channels.indexOf(ch)
        const rawPower = chIdx >= 0 ? (values[chIdx] ?? 0) : 0
        // Convert to log scale (dB) for better visualization — band powers span orders of magnitude
        point[ch] = rawPower > 0
          ? parseFloat((10 * Math.log10(rawPower)).toFixed(2))
          : -120
      })
      return point
    })

    return { chartData: points, channelNames: names }
  }, [data, activeChannels])

  if (!data || chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        No band power data. Click &quot;Analyze&quot; to compute.
      </div>
    )
  }

  return (
    <div className="w-full">
      <p className="text-xs text-gray-400 mb-2 text-center">Band Power by Channel (dB)</p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ top: 4, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="band"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            label={{ value: 'dB', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }}
            width={36}
          />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 6 }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(val, name) => [`${val} dB`, name]}
            labelFormatter={(v) => {
              const key = v.toLowerCase()
              return BAND_LABELS[key] || v
            }}
          />
          {channelNames.length <= 8 && (
            <Legend wrapperStyle={{ fontSize: 11 }} />
          )}
          {channelNames.map((ch, i) => (
            <Bar
              key={ch}
              dataKey={ch}
              fill={CHANNEL_COLORS[i % CHANNEL_COLORS.length]}
              maxBarSize={20}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
