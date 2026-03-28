// ModelLayerEditor.jsx — Visual CNN layer builder
//
// Renders the layers array of a ModelConfig as editable cards.
// Each card shows the layer type, its parameters, an optional activation
// selector, move-up/down buttons, and a delete button.
//
// Props:
//   layers         - array of layer config objects (the current architecture)
//   onChange       - called with (newLayers) whenever the user edits anything
//   inputChannels  - number of EEG channels (for validate-config call)
//   inputSamples   - number of samples per window (for validate-config call)
//   classes        - array of class name strings (for validate-config call)

import { useState, useEffect } from 'react'

import { API_BASE_URL } from '../config'
const API = API_BASE_URL

// Color per layer type for the left border of each card
const LAYER_COLORS = {
  Conv2d:          'border-blue-500',
  DepthwiseConv2d: 'border-cyan-500',
  AvgPool2d:       'border-green-500',
  MaxPool2d:       'border-emerald-500',
  Dropout:         'border-yellow-500',
  BatchNorm2d:     'border-orange-500',
  Flatten:         'border-gray-400',
  Linear:          'border-purple-500',
  LSTM:            'border-pink-500',
  CustomLayer:     'border-amber-500',
}

// Default parameters to fill in when adding a new layer of a given type
function defaultLayer(type, schema) {
  const typeDef = schema.find(s => s.type === type)
  if (!typeDef) return { type }
  const layer = { type }
  typeDef.parameters.forEach(p => {
    layer[p.name] = p.default
  })
  if (typeDef.supports_activation) {
    layer.activation = 'ELU'
  }
  return layer
}

export default function ModelLayerEditor({
  layers,
  onChange,
  inputChannels,
  inputSamples,
  classes,
}) {
  const [layerSchema, setLayerSchema]       = useState([])      // from GET /layer-types
  const [activationOptions, setActivationOptions] = useState([])
  const [schemaLoaded, setSchemaLoaded]     = useState(false)
  const [showAddMenu, setShowAddMenu]       = useState(false)    // dropdown for Add Layer

  const [validating, setValidating]         = useState(false)
  const [validationResult, setValidationResult] = useState(null) // {valid, errors, layer_shapes}

  // Load layer type schema on mount
  useEffect(() => {
    fetch(`${API}/api/train/layer-types`)
      .then(r => r.json())
      .then(json => {
        if (json.success) {
          setLayerSchema(json.data.layer_types)
          setActivationOptions(json.data.activation_options)
          setSchemaLoaded(true)
        }
      })
      .catch(() => {})
  }, [])

  // Clear validation result whenever layers change
  useEffect(() => {
    setValidationResult(null)
  }, [layers])

  // -------------------------------------------------------------------------
  // Mutation helpers
  // -------------------------------------------------------------------------

  function updateLayer(index, patch) {
    const updated = layers.map((l, i) => i === index ? { ...l, ...patch } : l)
    onChange(updated)
  }

  function deleteLayer(index) {
    onChange(layers.filter((_, i) => i !== index))
  }

  function moveLayer(index, direction) {
    const newLayers = [...layers]
    const target    = index + direction
    if (target < 0 || target >= newLayers.length) return
    ;[newLayers[index], newLayers[target]] = [newLayers[target], newLayers[index]]
    onChange(newLayers)
  }

  function addLayer(type) {
    const newLayer = defaultLayer(type, layerSchema)
    onChange([...layers, newLayer])
    setShowAddMenu(false)
  }

  // -------------------------------------------------------------------------
  // Validation
  // -------------------------------------------------------------------------

  async function validateArchitecture() {
    setValidating(true)
    setValidationResult(null)
    try {
      const res  = await fetch(`${API}/api/train/validate-config`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          layers,
          input_channels: inputChannels,
          input_samples:  inputSamples,
          classes:        classes.length > 0 ? classes : ['A', 'B'], // need >=1 class to run
        }),
      })
      const json = await res.json()
      if (json.success) {
        setValidationResult(json.data)
      }
    } catch {
      setValidationResult({ valid: false, errors: ['Could not reach backend.'], layer_shapes: [] })
    } finally {
      setValidating(false)
    }
  }

  // -------------------------------------------------------------------------
  // Parameter input renderer — switches on the parameter type
  // -------------------------------------------------------------------------

  function renderParamInput(layerIndex, param, currentValue) {
    const { name, type } = param

    if (type === 'bool') {
      return (
        <label key={name} className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={!!currentValue}
            onChange={e => updateLayer(layerIndex, { [name]: e.target.checked })}
            className="accent-purple-500"
          />
          <span className="text-xs text-gray-400">{name}</span>
        </label>
      )
    }

    if (type === 'list[int]') {
      // Show each element of the array as a separate small input
      const arr = Array.isArray(currentValue) ? currentValue : [currentValue]
      return (
        <div key={name} className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">{name}</span>
          <div className="flex gap-1 items-center">
            {arr.map((val, idx) => (
              <input
                key={idx}
                type="text"
                value={val}
                onChange={e => {
                  const newArr = [...arr]
                  const parsed = e.target.value
                  // Keep as string if it's the special keyword, else convert to number
                  newArr[idx] = isNaN(parsed) || parsed === '' ? parsed : Number(parsed)
                  updateLayer(layerIndex, { [name]: newArr })
                }}
                className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500 text-center"
              />
            ))}
            <span className="text-xs text-gray-600">[h, w]</span>
          </div>
        </div>
      )
    }

    if (type === 'str_or_int') {
      return (
        <div key={name} className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">{name}</span>
          <input
            type="text"
            value={currentValue ?? ''}
            onChange={e => {
              const v = e.target.value
              updateLayer(layerIndex, { [name]: isNaN(v) || v === '' ? v : Number(v) })
            }}
            className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500"
          />
        </div>
      )
    }

    if (type === 'int_or_string') {
      return (
        <div key={name} className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">{name}</span>
          <input
            type="text"
            value={currentValue ?? ''}
            onChange={e => {
              const v = e.target.value
              updateLayer(layerIndex, { [name]: isNaN(v) || v === '' ? v : Number(v) })
            }}
            className="w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500"
          />
        </div>
      )
    }

    // code — multi-line textarea for user-pasted Python code (CustomLayer, CustomStep)
    if (type === 'code') {
      return (
        <div key={name} className="flex flex-col gap-1 w-full">
          <span className="text-xs text-gray-500 font-mono">Python code</span>
          <textarea
            value={currentValue ?? ''}
            onChange={e => updateLayer(layerIndex, { [name]: e.target.value })}
            rows={10}
            spellCheck={false}
            className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-xs text-green-300 font-mono focus:outline-none focus:border-amber-500 resize-y leading-relaxed"
            placeholder="class CustomLayer(nn.Module):&#10;    def __init__(self):&#10;        super().__init__()&#10;&#10;    def forward(self, x):&#10;        return x"
          />
          <span className="text-xs text-gray-600">
            Class must be named <code className="text-amber-400">CustomLayer</code> and extend <code className="text-amber-400">nn.Module</code>. Available: <code className="text-gray-400">nn</code>, <code className="text-gray-400">torch</code>.
          </span>
        </div>
      )
    }

    // float or int — number input
    const step = type === 'float' ? 0.05 : 1
    return (
      <div key={name} className="flex flex-col gap-1">
        <span className="text-xs text-gray-500">{name}</span>
        <input
          type="number"
          step={step}
          value={currentValue ?? ''}
          onChange={e => {
            const v = type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value)
            updateLayer(layerIndex, { [name]: isNaN(v) ? currentValue : v })
          }}
          className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 focus:outline-none focus:border-purple-500"
        />
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (!schemaLoaded) {
    return (
      <div className="text-gray-500 text-sm py-4 text-center">
        Loading layer types…
      </div>
    )
  }

  return (
    <div className="w-full">
      {/* Layer cards */}
      <div className="space-y-2 mb-4">
        {layers.map((layer, index) => {
          const typeDef    = layerSchema.find(s => s.type === layer.type)
          const colorClass = LAYER_COLORS[layer.type] || 'border-gray-600'
          const shape      = validationResult?.layer_shapes?.[index]

          return (
            <div
              key={index}
              className={`bg-gray-900 border-l-4 ${colorClass} border border-gray-800 rounded-lg px-4 py-3`}
            >
              {/* Card header: type + move/delete buttons */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-200">{layer.type}</span>
                <div className="flex items-center gap-1">
                  {/* Move up */}
                  <button
                    onClick={() => moveLayer(index, -1)}
                    disabled={index === 0}
                    title="Move up"
                    className="text-gray-600 hover:text-gray-300 disabled:opacity-20 px-1 text-xs transition-colors"
                  >
                    ▲
                  </button>
                  {/* Move down */}
                  <button
                    onClick={() => moveLayer(index, 1)}
                    disabled={index === layers.length - 1}
                    title="Move down"
                    className="text-gray-600 hover:text-gray-300 disabled:opacity-20 px-1 text-xs transition-colors"
                  >
                    ▼
                  </button>
                  {/* Delete */}
                  <button
                    onClick={() => deleteLayer(index)}
                    title="Delete layer"
                    className="text-gray-600 hover:text-red-400 px-1.5 text-sm transition-colors ml-1"
                  >
                    ×
                  </button>
                </div>
              </div>

              {/* Parameters */}
              {typeDef && typeDef.parameters.length > 0 && (
                <div className="flex flex-wrap gap-3 mb-2">
                  {typeDef.parameters.map(param =>
                    renderParamInput(index, param, layer[param.name])
                  )}
                </div>
              )}

              {/* Activation selector (only for layers that support it) */}
              {typeDef?.supports_activation && (
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-gray-500">activation</span>
                  <select
                    value={layer.activation || ''}
                    onChange={e => updateLayer(index, { activation: e.target.value || undefined })}
                    className="bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-200 focus:outline-none focus:border-purple-500"
                  >
                    <option value="">None</option>
                    {activationOptions.map(a => (
                      <option key={a.name} value={a.name}>{a.name}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Shape display under each card */}
              {shape && (
                <div className="mt-2 text-xs text-gray-600 font-mono">
                  → {shape.join(' × ')}
                </div>
              )}
              {validationResult && shape === null && (
                <div className="mt-2 text-xs text-red-500">
                  shape error — check layers above
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Add Layer button */}
      <div className="relative mb-4">
        <button
          onClick={() => setShowAddMenu(v => !v)}
          className="text-sm px-4 py-1.5 border border-dashed border-gray-600 rounded-lg text-gray-400 hover:border-purple-500 hover:text-purple-300 transition-colors w-full"
        >
          + Add Layer
        </button>
        {showAddMenu && (
          <div className="absolute left-0 right-0 mt-1 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-10 overflow-hidden">
            {layerSchema.map(lt => (
              <button
                key={lt.type}
                onClick={() => addLayer(lt.type)}
                className="w-full text-left px-4 py-2.5 text-sm text-gray-300 hover:bg-gray-800 transition-colors border-b border-gray-800 last:border-0"
              >
                <span className="font-medium text-gray-200">{lt.type}</span>
                <span className="ml-2 text-xs text-gray-500">{lt.description}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Validate button + results */}
      <div>
        <button
          onClick={validateArchitecture}
          disabled={validating || layers.length === 0}
          className="text-sm px-4 py-1.5 bg-gray-800 border border-gray-600 hover:border-purple-500 disabled:opacity-40 text-gray-300 rounded-lg transition-colors"
        >
          {validating ? 'Validating…' : 'Validate Architecture'}
        </button>

        {validationResult && (
          <div className="mt-3">
            {validationResult.valid ? (
              <p className="text-xs text-green-400 font-medium">
                Architecture is valid — shapes computed above.
              </p>
            ) : (
              <div>
                <p className="text-xs text-red-400 font-medium mb-1">
                  {validationResult.errors.length} error{validationResult.errors.length !== 1 ? 's' : ''}:
                </p>
                <ul className="space-y-1">
                  {validationResult.errors.map((err, i) => (
                    <li key={i} className="text-xs text-red-400">• {err}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
