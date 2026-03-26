// WaveformChart.jsx — EEG waveform display using Recharts
// Props:
//   data         {Array}  — array of data points: [{ time: 0.0, Fp1: 0.12, Fp2: -0.03, ... }]
//   channelNames {Array}  — list of channel name strings to plot, e.g. ["Fp1", "C3"]
//   height       {number} — chart height in pixels (default 300)
// Stub for Session 1 — the actual waveform display is built in Session 2.
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// A small fixed palette of colors — one per channel.
// In practice channels rarely exceed 64 so this is fine for MVP.
const CHANNEL_COLORS = [
  '#a78bfa', '#34d399', '#fb923c', '#60a5fa',
  '#f472b6', '#facc15', '#4ade80', '#f87171',
]

// Props documented above per CLAUDE.md code standards
export default function WaveformChart({ data = [], channelNames = [], height = 300 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 11 }} />
        <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
        <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />

        {/* Render one line per channel, each with a distinct color from the palette */}
        {channelNames.map((name, idx) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={CHANNEL_COLORS[idx % CHANNEL_COLORS.length]}
            dot={false}         // no dots — EEG data has thousands of points, dots would be invisible noise
            strokeWidth={1.5}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
