// FrequencyChart.jsx — FFT or PSD frequency-domain line chart
//
// Displays the power spectrum for each EEG channel as overlapping lines.
// X axis = frequency in Hz, Y axis = power (linear or dB scale).
//
// Props:
//   data         - API response from /api/analysis/psd or /api/analysis/fft
//                  Shape: { frequencies: number[], power: { [channelName]: number[] } }
//                  OR:    { frequencies: number[], magnitudes: { [channelName]: number[] } }
//   title        - chart title string (e.g. "Power Spectral Density")
//   maxFreqHz    - clip x axis at this frequency (default 60 Hz — EEG range)
//   activeChannels - Set of channel names to show (all shown if null)

import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'

// Color palette for up to 16 channels
const CHANNEL_COLORS = [
  '#60a5fa', '#34d399', '#f87171', '#fbbf24', '#a78bfa',
  '#fb923c', '#38bdf8', '#4ade80', '#f472b6', '#facc15',
  '#c084fc', '#2dd4bf', '#fb7185', '#86efac', '#93c5fd',
  '#fcd34d',
]

export default function FrequencyChart({
  data,
  title = 'Frequency Spectrum',
  maxFreqHz = 60,
  activeChannels = null,
}) {
  // Build Recharts-compatible data: array of { freq, Fp1, Fp2, ... }
  const { chartData, channelNames } = useMemo(() => {
    if (!data || !data.frequencies) return { chartData: [], channelNames: [] }

    // Support both 'power' (PSD) and 'magnitudes' (FFT) keys
    const valueMap = data.power || data.magnitudes || {}
    const names = Object.keys(valueMap)

    // Filter to maxFreqHz — anything above is usually noise and clutters the chart
    const points = []
    for (let i = 0; i < data.frequencies.length; i++) {
      const freq = data.frequencies[i]
      if (freq > maxFreqHz) break

      const point = { freq: parseFloat(freq.toFixed(2)) }
      for (const ch of names) {
        if (activeChannels === null || activeChannels.has(ch)) {
          const val = valueMap[ch]?.[i] ?? 0
          // Log scale for PSD is much more readable for EEG
          point[ch] = val > 0 ? parseFloat((10 * Math.log10(val)).toFixed(3)) : -120
        }
      }
      points.push(point)
    }

    return { chartData: points, channelNames: names }
  }, [data, maxFreqHz, activeChannels])

  if (!data || chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        No frequency data. Click &quot;Analyze&quot; to compute.
      </div>
    )
  }

  const visibleChannels = activeChannels
    ? channelNames.filter(ch => activeChannels.has(ch))
    : channelNames

  return (
    <div className="w-full">
      {title && (
        <p className="text-xs text-gray-400 mb-2 text-center">{title}</p>
      )}
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="freq"
            type="number"
            domain={[0, maxFreqHz]}
            tickCount={7}
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -2, fill: '#9ca3af', fontSize: 11 }}
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
            labelFormatter={(v) => `${v} Hz`}
          />
          {visibleChannels.length <= 8 && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {visibleChannels.map((ch, i) => (
            <Line
              key={ch}
              type="monotone"
              dataKey={ch}
              stroke={CHANNEL_COLORS[i % CHANNEL_COLORS.length]}
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
