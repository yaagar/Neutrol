// SpectrogramChart.jsx — Time-frequency spectrogram heatmap
//
// Renders a STFT spectrogram as a canvas-based heatmap.
// X axis = time, Y axis = frequency, color = power (blue=low, red=high).
//
// Uses HTML5 Canvas for rendering because Recharts doesn't have a native
// 2D color matrix chart type. Canvas draws hundreds of colored cells efficiently.
//
// Props:
//   data  - API response from /api/analysis/spectrogram
//           Shape: { times: number[], frequencies: number[], power_db: number[][], channel_name: string }

import { useEffect, useRef, useMemo } from 'react'

// Jet-like colormap: maps a 0–1 value to an RGB color
// Blue (cold/low power) → Cyan → Green → Yellow → Red (hot/high power)
function jetColor(t) {
  // Clamp
  t = Math.max(0, Math.min(1, t))

  let r, g, b
  if (t < 0.125) {
    r = 0; g = 0; b = 0.5 + t * 4
  } else if (t < 0.375) {
    r = 0; g = (t - 0.125) * 4; b = 1
  } else if (t < 0.625) {
    r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4
  } else if (t < 0.875) {
    r = 1; g = 1 - (t - 0.625) * 4; b = 0
  } else {
    r = 1 - (t - 0.875) * 4; g = 0; b = 0
  }

  return [
    Math.round(Math.max(0, Math.min(1, r)) * 255),
    Math.round(Math.max(0, Math.min(1, g)) * 255),
    Math.round(Math.max(0, Math.min(1, b)) * 255),
  ]
}

export default function SpectrogramChart({ data }) {
  const canvasRef = useRef(null)

  // Clip frequencies to 60 Hz for EEG range — higher is mostly noise
  const MAX_FREQ_HZ = 60

  const { filteredPower, filteredFreqs, times } = useMemo(() => {
    if (!data || !data.power_db || !data.frequencies || !data.times) {
      return { filteredPower: null, filteredFreqs: [], times: [] }
    }

    // Find the index where frequency exceeds MAX_FREQ_HZ
    const freqCutoff = data.frequencies.findIndex(f => f > MAX_FREQ_HZ)
    const endIdx = freqCutoff === -1 ? data.frequencies.length : freqCutoff

    const filteredFreqs = data.frequencies.slice(0, endIdx)
    const filteredPower = data.power_db.slice(0, endIdx)

    return { filteredPower, filteredFreqs, times: data.times }
  }, [data])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !filteredPower || filteredPower.length === 0) return

    const ctx = canvas.getContext('2d')
    const nFreqs = filteredPower.length
    const nTimes = filteredPower[0]?.length ?? 0

    if (nFreqs === 0 || nTimes === 0) return

    // Find global min/max for color normalization
    let minDb = Infinity, maxDb = -Infinity
    for (const row of filteredPower) {
      for (const v of row) {
        if (v > -100 && v < minDb) minDb = v
        if (v > maxDb) maxDb = v
      }
    }
    // Avoid div-by-zero on flat signals
    const range = maxDb - minDb || 1

    const W = canvas.width
    const H = canvas.height
    const cellW = W / nTimes
    const cellH = H / nFreqs

    ctx.clearRect(0, 0, W, H)

    // Draw each time-frequency cell as a colored rectangle.
    // The spectrogram matrix is [freq_idx][time_idx].
    // Frequency axis is flipped: index 0 = lowest freq, should appear at bottom.
    for (let fi = 0; fi < nFreqs; fi++) {
      for (let ti = 0; ti < nTimes; ti++) {
        const db = filteredPower[fi][ti]
        const t = (db - minDb) / range
        const [r, g, b] = jetColor(t)

        // Flip frequency axis so low frequencies are at bottom
        const yIdx = nFreqs - 1 - fi
        ctx.fillStyle = `rgb(${r},${g},${b})`
        ctx.fillRect(
          Math.floor(ti * cellW),
          Math.floor(yIdx * cellH),
          Math.ceil(cellW) + 1,   // +1 to avoid gaps between cells
          Math.ceil(cellH) + 1,
        )
      }
    }
  }, [filteredPower])

  if (!data) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        No spectrogram data. Click &quot;Analyze&quot; to compute.
      </div>
    )
  }

  const channelLabel = data.channel_name || 'Channel 0'
  const duration = times.length > 0 ? times[times.length - 1].toFixed(1) : '?'
  const maxFreq = filteredFreqs.length > 0
    ? Math.min(MAX_FREQ_HZ, filteredFreqs[filteredFreqs.length - 1]).toFixed(0)
    : '?'

  return (
    <div className="w-full">
      <p className="text-xs text-gray-400 mb-2 text-center">
        Spectrogram — {channelLabel} (0–{maxFreq} Hz, {duration}s)
      </p>
      {/* Canvas container with axis labels */}
      <div className="relative">
        {/* Y-axis label */}
        <div className="absolute left-0 top-0 bottom-0 flex items-center">
          <span
            className="text-xs text-gray-500"
            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', marginLeft: 2 }}
          >
            Frequency (Hz)
          </span>
        </div>
        {/* Frequency labels on Y axis */}
        <div className="absolute left-5 top-0 bottom-0 flex flex-col justify-between" style={{ width: 28 }}>
          <span className="text-xs text-gray-500">{maxFreq}</span>
          <span className="text-xs text-gray-500">0</span>
        </div>
        {/* Main canvas */}
        <div className="ml-12">
          <canvas
            ref={canvasRef}
            width={680}
            height={200}
            className="w-full rounded"
            style={{ imageRendering: 'pixelated', background: '#111827' }}
          />
          {/* X-axis labels */}
          <div className="flex justify-between mt-1">
            <span className="text-xs text-gray-500">0s</span>
            <span className="text-xs text-gray-500 text-center">Time</span>
            <span className="text-xs text-gray-500">{duration}s</span>
          </div>
        </div>
        {/* Colorbar legend */}
        <div className="mt-2 ml-12 flex items-center gap-2">
          <span className="text-xs text-gray-500">Low</span>
          <div
            className="flex-1 h-2 rounded"
            style={{
              background: 'linear-gradient(to right, #00008b, #00bfff, #00ff00, #ffff00, #ff0000)',
            }}
          />
          <span className="text-xs text-gray-500">High (dB)</span>
        </div>
      </div>
    </div>
  )
}
