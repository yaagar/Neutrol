// ElectrodeMap.jsx — Interactive 10-20 electrode placement SVG map
//
// Renders a standard EEG head diagram with 21 electrode positions.
// Supports two modes:
//   "select"  — click electrodes to include/exclude channels (for ChannelSelector step)
//   "reref"   — click one electrode to set as the reference (for Rereferencing step)
//
// Props:
//   channelNames      {string[]}  — all channel names in the session
//   mode              {string}    — "select" | "reref"
//   selectedChannels  {string[]}  — channels currently selected (select mode)
//   referenceChannel  {string}    — the current reference channel (reref mode)
//   onToggle          {fn(ch)}    — called when a channel is toggled (select mode)
//   onSetRef          {fn(ch)}    — called when a channel is set as reference (reref mode)

import { useState } from 'react'

// Standard 10-20 positions as % of viewBox (300×320)
// Values represent center coordinates (cx%, cy%) of each electrode circle
const POSITIONS_10_20 = {
  Fp1:  [35,  9], Fp2:  [65,  9],
  F7:   [14, 29], F3:   [29, 24], Fz:   [50, 21], F4:   [71, 24], F8:   [86, 29],
  T3:   [ 9, 51], C3:   [22, 50], Cz:   [50, 50], C4:   [78, 50], T4:   [91, 51],
  T5:   [14, 71], P3:   [29, 76], Pz:   [50, 79], P4:   [71, 76], T6:   [86, 71],
  O1:   [35, 91], O2:   [65, 91],
  A1:   [ 4, 60], A2:   [96, 60],
}

const VB_W = 300
const VB_H = 320

export default function ElectrodeMap({
  channelNames = [],
  mode = 'select',
  selectedChannels = [],
  referenceChannel = null,
  onToggle = () => {},
  onSetRef = () => {},
}) {
  const [hovered, setHovered] = useState(null)

  // Channels in the session that have known 10-20 positions
  const mappedChannels   = channelNames.filter(ch => POSITIONS_10_20[ch])
  // Channels not in the 10-20 lookup — shown as checkboxes below the SVG
  const unmappedChannels = channelNames.filter(ch => !POSITIONS_10_20[ch])

  // Don't show 10-20 positions that don't correspond to session channels
  // (avoids confusing ghost electrodes)
  const activePositions  = Object.entries(POSITIONS_10_20).filter(([ch]) =>
    channelNames.includes(ch)
  )

  function getElectrodeColor(ch) {
    if (mode === 'reref') {
      if (ch === referenceChannel) return '#f59e0b'  // gold = reference
      if (selectedChannels.includes(ch)) return '#7c3aed'  // purple = active
      return '#374151'
    }
    // select mode
    return selectedChannels.includes(ch) ? '#7c3aed' : '#374151'
  }

  function getElectrodeStroke(ch) {
    if (hovered === ch) return '#a78bfa'
    if (ch === referenceChannel) return '#fbbf24'
    if (selectedChannels.includes(ch)) return '#a78bfa'
    return '#6b7280'
  }

  function handleClick(ch) {
    if (mode === 'select') onToggle(ch)
    else onSetRef(ch)
  }

  return (
    <div className="flex flex-col items-start gap-4">
      {/* SVG head diagram */}
      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="220"
          height="235"
          className="block"
        >
          {/* Head outline — filled ellipse */}
          <ellipse
            cx="150" cy="158"
            rx="132" ry="140"
            fill="#111827"
            stroke="#4b5563"
            strokeWidth="2"
          />

          {/* Nose — small triangle at top center */}
          <polygon
            points="135,16 165,16 150,3"
            fill="#4b5563"
          />

          {/* Left ear */}
          <ellipse cx="16" cy="158" rx="10" ry="18" fill="#111827" stroke="#4b5563" strokeWidth="1.5" />
          {/* Right ear */}
          <ellipse cx="284" cy="158" rx="10" ry="18" fill="#111827" stroke="#4b5563" strokeWidth="1.5" />

          {/* Nasion mark (top) */}
          <line x1="150" y1="18" x2="150" y2="25" stroke="#6b7280" strokeWidth="1.5" />
          {/* Inion mark (bottom) */}
          <line x1="150" y1="291" x2="150" y2="298" stroke="#6b7280" strokeWidth="1.5" />

          {/* Electrode circles for all session channels with known positions */}
          {activePositions.map(([ch, [px, py]]) => {
            const cx = (px / 100) * VB_W
            const cy = (py / 100) * VB_H
            return (
              <g
                key={ch}
                onClick={() => handleClick(ch)}
                onMouseEnter={() => setHovered(ch)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: 'pointer' }}
              >
                <circle
                  cx={cx} cy={cy} r="14"
                  fill={getElectrodeColor(ch)}
                  stroke={getElectrodeStroke(ch)}
                  strokeWidth="2"
                />
                <text
                  x={cx} y={cy + 1}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="7"
                  fontFamily="monospace"
                  fill={ch === referenceChannel ? '#fef3c7' : selectedChannels.includes(ch) ? '#e9d5ff' : '#9ca3af'}
                >
                  {ch}
                </text>
                {/* Star badge on reference channel */}
                {ch === referenceChannel && (
                  <text x={cx + 10} y={cy - 10} fontSize="10" fill="#f59e0b">★</text>
                )}
              </g>
            )
          })}
        </svg>

        {/* Hover tooltip */}
        {hovered && (
          <div className="absolute top-0 left-full ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 whitespace-nowrap pointer-events-none z-10">
            {hovered}
            {mode === 'reref' && hovered === referenceChannel && ' (reference)'}
            {mode === 'select' && selectedChannels.includes(hovered) && ' (selected)'}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-3 text-xs text-gray-500 flex-wrap">
        {mode === 'select' ? (
          <>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-purple-700 inline-block" /> Selected
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-gray-700 border border-gray-600 inline-block" /> Unselected
            </span>
          </>
        ) : (
          <>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-amber-500 inline-block" /> Reference
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-purple-700 inline-block" /> Active
            </span>
          </>
        )}
        {mappedChannels.length === 0 && (
          <span className="text-gray-600">No 10-20 channels detected in session</span>
        )}
      </div>

      {/* Unmapped channels — fallback checkbox list */}
      {unmappedChannels.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Other channels (not in 10-20 map):</p>
          <div className="flex flex-wrap gap-1.5">
            {unmappedChannels.map(ch => (
              <button
                key={ch}
                onClick={() => handleClick(ch)}
                className={`px-2 py-0.5 rounded text-xs border transition-colors ${
                  mode === 'reref' && ch === referenceChannel
                    ? 'bg-amber-700 border-amber-500 text-amber-100'
                    : selectedChannels.includes(ch)
                    ? 'bg-purple-800 border-purple-600 text-purple-200'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {ch}
                {mode === 'reref' && ch === referenceChannel && ' ★'}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
