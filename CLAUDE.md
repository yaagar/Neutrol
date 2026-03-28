# CLAUDE.md — Neutrol Project Memory

> Single source of truth. Read completely before writing any code.
> Do not invent alternatives. Do not introduce unlisted libraries. Follow this file exactly.

---

## Mandatory Behavior Rules

### Rule 1 — Explain Before You Build
Before any code, write this block:
```
=== BEFORE I BUILD ===
What I'm building: [description]
Why this approach: [reasoning]
Files I'll create/modify:
  - path/to/file → [purpose]
What this connects to: [dependencies]
======================
```

### Rule 2 — Explain Code Inline
Every non-trivial block gets a plain-English comment explaining what and why.

### Rule 3 — Explain Connections
After finishing a task, write `=== HOW THIS CONNECTS ===` explaining what calls this and what comes next.

### Rule 4 — Flag Decisions
Any choice not specified here: `=== DECISION === I chose X over Y because [reason] ================`

### Rule 5 — Never Silently Deviate
If a prompt contradicts this file, stop and ask. Do not guess.

---

## What This Project Is

**Neutrol** — open-source, beginner-friendly EEG development platform.

User journey: Import EEG data → Preprocess + Analyze → Label segments → Train model → Test live → Export

Design philosophy: Like Arduino IDE. Simple for beginners, fully editable for experts.

---

## Tech Stack

### Frontend
| Tool | Why |
|---|---|
| React + Vite | Complex dynamic state — live charts, pipeline editing, real-time predictions |
| TailwindCSS | Consistent styling, Claude Code generates it reliably |
| Zustand | Simple global store for session state across pages |
| Recharts | EEG waveforms, charts, FFT plots — pure React |

### Backend
| Tool | Why |
|---|---|
| Python 3.11 | Non-negotiable for EEG/ML ecosystem |
| FastAPI + Uvicorn | HTTP + WebSocket support, auto-docs at /docs |
| NumPy | All signals as `(n_channels, n_samples)` arrays |
| SciPy | Filtering, FFT, PSD, spectrogram |
| MNE-Python | EDF/BDF reading, event parsing, ICA, re-referencing |
| PyTorch | Dynamic CNN/LSTM builder from layer config |
| scikit-learn | Confusion matrix, classification metrics |
| python-multipart | File uploads |
| websockets | Real-time streaming and inference |
| **onnx** | NEW (Part 2) — Export PyTorch model to ONNX format |
| **onnxruntime** | NEW (Part 2) — Run ONNX models on edge devices (Raspberry Pi etc.) |
| **brainflow** | NEW (Part 2) — Unified API for 30+ live EEG hardware devices |

---

## Directory Structure

```
Neutrol/
├── CLAUDE.md
├── README.md
├── frontend/
│   ├── src/
│   │   ├── config.js                     ← API_BASE_URL = 'http://localhost:8000'
│   │   ├── App.jsx                       ← Root component, routing, sidebar
│   │   ├── components/
│   │   │   ├── Sidebar.jsx
│   │   │   ├── StatusBar.jsx
│   │   │   ├── PipelineStep.jsx
│   │   │   ├── WaveformChart.jsx
│   │   │   ├── FrequencyChart.jsx
│   │   │   ├── BandPowerChart.jsx
│   │   │   ├── SpectrogramChart.jsx
│   │   │   └── ModelLayerEditor.jsx
│   │   ├── pages/
│   │   │   ├── ImportPage.jsx
│   │   │   ├── PreprocessPage.jsx
│   │   │   ├── LabelPage.jsx
│   │   │   ├── TrainPage.jsx
│   │   │   ├── InferencePage.jsx         ← UPDATED Part 2: real hardware streaming
│   │   │   └── ExportPage.jsx            ← UPDATED Part 2: ONNX + microcontroller export
│   │   ├── hooks/
│   │   │   ├── useSession.js
│   │   │   └── useWebSocket.js
│   │   └── store/appStore.js
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── api/
│   │   ├── data_routes.py
│   │   ├── pipeline_routes.py
│   │   ├── analysis_routes.py
│   │   ├── label_routes.py
│   │   ├── train_routes.py
│   │   ├── inference_routes.py           ← UPDATED Part 2: BrainFlow hardware endpoint
│   │   └── export_routes.py              ← UPDATED Part 2: ONNX + microcontroller endpoints
│   ├── core/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── analysis.py
│   │   ├── models.py
│   │   ├── trainer.py
│   │   ├── inference.py
│   │   └── export.py                     ← UPDATED Part 2: ONNX export + C array generation
│   └── sessions/session_store.py
├── configs/
├── datasets/sample_eeg.csv
└── exports/
```

### Architecture rule: `api/` = route handlers only, `core/` = all logic. Never mix.

---

## Core Data Structures

These are fixed. Never invent duplicates.

```python
@dataclass
class DataSession:
    session_id: str
    signals: np.ndarray       # (n_channels, n_samples) — ALWAYS this shape
    sample_rate: int
    channel_names: list[str]
    duration_seconds: float
    file_path: str
    created_at: str
    metadata: dict = field(default_factory=dict)

@dataclass
class LabeledWindow:
    window_id: str
    data: np.ndarray           # (n_channels, n_samples_in_window)
    label: str
    start_time_sec: float
    session_id: str

@dataclass
class EventMarker:
    onset_sec: float
    duration_sec: float
    label: str
    description: str
```

PipelineConfig JSON format:
```json
{
  "pipeline_id": "uuid",
  "session_id": "uuid",
  "sample_rate": 256,
  "steps": [
    {"step": "notch",     "freq_hz": 60.0, "quality_factor": 30.0},
    {"step": "bandpass",  "low_hz": 1.0, "high_hz": 40.0},
    {"step": "reref",     "method": "average"},
    {"step": "normalize", "method": "zscore"},
    {"step": "window",    "window_sec": 1.0, "overlap_ratio": 0.5}
  ]
}
```

**Note: notch before bandpass. Notch removes 60Hz harmonics before bandpass cuts the range.**

Every API endpoint returns:
```json
{ "success": true, "data": {}, "error": null }
```

---

## SessionStore

Singleton in-memory store. All route files share the same instance.

```python
class SessionStore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sessions = {}         # session_id → DataSession
            cls._instance.pipeline_runs = {}    # run_id → {windows, start_times, sample_rate, pipeline_config}
            cls._instance.labeled_data = {}     # run_id → list[LabeledWindow]
            cls._instance.training_jobs = {}    # job_id → training state dict
            cls._instance.event_markers = {}    # session_id → list[EventMarker]
        return cls._instance
```

---

## API Endpoint Map

### Data `/api/data/`
| Method | Path | Does |
|---|---|---|
| POST | `/upload` | Upload .csv or .edf → session_id + metadata |
| POST | `/upload-events` | Upload event file → EventMarker list |
| GET | `/sessions` | List all sessions |
| GET | `/{session_id}` | Single session metadata |
| DELETE | `/{session_id}` | Remove session |

### Pipeline `/api/pipeline/`
| Method | Path | Does |
|---|---|---|
| POST | `/run` | Run pipeline → run_id + window count |
| GET | `/steps` | All available steps + parameters |
| POST | `/save` | Save config to configs/ |
| GET | `/saved` | List saved configs |

### Analysis `/api/analysis/`
| POST | `/fft` | FFT per channel |
| POST | `/psd` | Welch PSD per channel |
| POST | `/bandpower` | Delta/theta/alpha/beta/gamma per channel |
| POST | `/spectrogram` | STFT for one channel |

### Labels `/api/labels/`
| POST | `/save` | Save LabeledWindows (manual) |
| POST | `/from-events` | Auto-label from EventMarkers |
| GET | `/{session_id}` | All labels for session |
| GET | `/by-run/{run_id}` | Labels for specific run (must be defined BEFORE /{session_id}) |
| DELETE | `/{session_id}` | Clear labels |

### Train `/api/train/`
| POST | `/validate-config` | Validate ModelConfig + layer shapes |
| GET | `/layer-types` | All layer types + parameters |
| POST | `/start` | Start training → job_id |
| GET | `/{job_id}/status` | Progress (epoch, loss, accuracy) |
| GET | `/{job_id}/results` | Final metrics + confusion matrix |

### Inference `/api/inference/`
| POST | `/predict` | Single prediction → class + confidence |
| WS | `/ws/inference/stream` | WebSocket stream inference |
| GET | `/hardware/boards` | NEW: List supported BrainFlow board IDs |
| POST | `/hardware/connect` | NEW: Connect to live EEG device via BrainFlow |
| POST | `/hardware/disconnect` | NEW: Disconnect hardware |
| WS | `/ws/inference/hardware` | NEW: WebSocket stream from real hardware |

### Export `/api/export/`
| GET | `/pipeline/{run_id}` | Pipeline config JSON |
| GET | `/model/{job_id}` | Model weights .pt |
| GET | `/script/{job_id}` | Python inference script (with preprocessing embedded) |
| GET | `/onnx/{job_id}` | NEW: ONNX model file |
| GET | `/onnx-script/{job_id}` | NEW: Python ONNX inference script |
| GET | `/microcontroller/{job_id}` | NEW: ZIP with model_data.h + neutrol_inference.cpp + README |

---

## Part 1 — What Is Built (Sessions 1–9) ✅

All MVP features complete:
- EEG file import (CSV, EDF, event files)
- Preprocessing pipeline (notch, bandpass, reref, normalize, window, channel selector, epoch rejection, ICA)
- Signal analysis (FFT, PSD, band power, spectrogram) with Before/After toggle
- Manual + event-file auto-labeling
- Dynamic CNN training (fully editable layer architecture)
- Training results (loss curves, confusion matrix)
- Live inference with simulated stream (WebSocket)
- Export: pipeline JSON, model .pt, Python inference script
- UI polish: step reordering, session rename, sidebar progress dots

**Critical known issue from Part 1 (fixed in Part 2 Session 1):**
The exported Python inference script does NOT apply preprocessing. It expects a pre-processed array. This must be fixed — the script must embed the full pipeline and expose `predict(raw_window)` not `predict(preprocessed_window)`.

---

## Part 2 — What Gets Built Next

### Part 2 Session 1 — Live Data + Full Export (current session)

**Goal:** Make the exported scripts truly end-to-end, add real hardware streaming, add ONNX and microcontroller export.

#### Fix 1: Inference Script Must Embed Preprocessing

This is the most important fix. Update `backend/core/export.py`:

The generated `inference.py` script must:
1. Embed the full pipeline config at the top as an editable constant
2. Implement all preprocessing steps inline (no Neutrol import) — bandpass, notch, reref, normalize, sliding window as pure functions
3. Expose `predict(raw_window: np.ndarray) -> tuple[str, float]` — takes raw EEG, returns (class_label, confidence)
4. Have a queue-based threading live loop at the bottom:

```python
# ============================================================
# NEUTROL INFERENCE SCRIPT — v2 (with preprocessing)
# Generated by Neutrol. Edit the CONFIG section to customize.
# ============================================================

# ── SECTION 1: CONFIG (safe to edit) ────────────────────────
SAMPLE_RATE    = 256          # Hz — must match your EEG device
WINDOW_SEC     = 1.0          # seconds per prediction
CHANNELS       = ["C3","C4"]  # channels used — must match training
MODEL_PATH     = "model.pt"
CLASSES        = ["rest","left","right"]

# ── SECTION 2: PREPROCESSING (auto-generated from your pipeline)
# Only change if you retrain with different preprocessing.
PIPELINE_STEPS = [
    {"step": "notch",    "freq_hz": 60.0},
    {"step": "bandpass", "low_hz": 1.0, "high_hz": 40.0},
    {"step": "normalize","method": "zscore"},
]

# ── SECTION 3: YOUR OUTPUT CODE (edit this) ─────────────────
def on_prediction(label: str, confidence: float):
    """Called every time a prediction is made. Put your code here."""
    print(f"Prediction: {label} ({confidence:.1%})")
    # if label == "left": control_robot_left()

# ── SECTION 4: PREPROCESSING FUNCTIONS (do not edit) ────────
# [all filter functions implemented inline here]

# ── SECTION 5: INFERENCE LOOP (do not edit) ─────────────────
# Queue-based: one thread reads EEG, another runs inference
# This prevents dropped samples during processing
```

The filter coefficients for Section 4 are computed by `export.py` from the saved pipeline config using SciPy — same `butter()` and `iirnotch()` calls, coefficients extracted as float arrays and written as Python constants. This guarantees the exported script uses mathematically identical preprocessing to training.

#### Fix 2: ONNX Export

Add to `backend/core/export.py`:

```python
def export_onnx(job_id: str, export_path: str) -> str:
    """
    Export trained model to ONNX format.
    ONNX runs on onnxruntime — lighter than PyTorch, works on Raspberry Pi ARM.
    Returns path to the .onnx file.
    """
```

Process:
1. Load model from checkpoint using `build_model(config)`
2. Set to `model.eval()` — required before export
3. Create dummy input tensor of correct shape: `(1, 1, n_channels, n_samples)`
4. Call `torch.onnx.export(model, dummy_input, path, opset_version=12, input_names=['input'], output_names=['output'])`
5. Return the .onnx file path

Also generate `inference_onnx.py` — same structure as the Python script above but uses `onnxruntime` instead of PyTorch:
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
# preprocessing is identical — same embedded pipeline functions
output = session.run(None, {"input": window_array})[0]
```

Add to `backend/api/export_routes.py`:
- `GET /api/export/onnx/{job_id}` → returns .onnx file (FileResponse)
- `GET /api/export/onnx-script/{job_id}` → returns onnx inference script (PlainTextResponse)

#### Fix 3: Microcontroller Export (ESP32-S3, STM32)

Targets: ESP32-S3, STM32 F4/H7, Arduino Nano 33 BLE Sense. NOT Arduino Uno (insufficient RAM).

Conversion chain: `PyTorch → ONNX → TFLite → C array (.h file)`

Add to `backend/core/export.py`:

```python
def export_microcontroller(job_id: str, export_dir: str) -> str:
    """
    Generate a ZIP containing everything needed for ESP32-S3 / STM32 deployment.
    Requires: tensorflow and onnx2tf installed (flagged as optional dependency).
    Returns path to the .zip file.
    """
```

The ZIP contains three files:

**`model_data.h`** — model weights as a C array:
```c
// Auto-generated by Neutrol. Do not edit.
// Model: my_model | Classes: rest, left, right | Accuracy: 87.3%
alignas(8) const unsigned char neutrol_model_data[] = { 0x18, 0x00, ... };
const unsigned int neutrol_model_data_len = 12345;
```

**`neutrol_inference.cpp`** — TFLite Micro inference template:
```cpp
// ============================================================
// NEUTROL — Microcontroller Inference Template
// Tested on: ESP32-S3, STM32 H7, Arduino Nano 33 BLE Sense
// ============================================================

// ── SECTION 1: CONFIG (edit these) ──────────────────────────
#define SAMPLE_RATE      256
#define WINDOW_SAMPLES   256     // SAMPLE_RATE * WINDOW_SEC
#define N_CHANNELS       8
#define N_CLASSES        3
const char* CLASS_LABELS[] = {"rest", "left", "right"};

// ── SECTION 2: FILTER COEFFICIENTS (auto-generated) ─────────
// Bandpass 1-40Hz, Butterworth order 4, at 256Hz
// These match exactly what your model was trained on.
static const float BANDPASS_SOS[2][6] = {
    {0.0007f, 0.0014f, 0.0007f, 1.0f, -1.9259f, 0.9287f},
    {0.0007f, 0.0014f, 0.0007f, 1.0f, -1.7834f, 0.8108f}
};

// ── SECTION 3: YOUR OUTPUT CODE (edit this) ─────────────────
void on_prediction(int class_index, float confidence) {
    Serial.printf("Prediction: %s (%.1f%%)\n",
                  CLASS_LABELS[class_index], confidence * 100);
    // Add your GPIO/motor/servo control here:
    // if (class_index == 1) digitalWrite(MOTOR_LEFT, HIGH);
}

// ── SECTION 4: TFLITE + PREPROCESSING (do not edit) ─────────
#include "model_data.h"
#include <TensorFlowLite.h>
// [TFLite Micro boilerplate + C filter implementations]
```

**`README_microcontroller.md`** — step-by-step instructions:
- Install Arduino IDE or ESP-IDF
- Install TensorFlow Lite Micro library
- Copy both files into your sketch folder
- Connect your EEG hardware to the ADC pins
- Upload and open Serial Monitor

The C filter coefficients in Section 2 are computed by `export.py` using `scipy.signal.butter()` and extracted as float arrays — same math as Python, different language.

**Important**: The microcontroller export endpoint must check if `tensorflow` is installed and return a clear error if not:
```json
{
  "success": false,
  "error": "Microcontroller export requires TensorFlow. Run: pip install tensorflow onnx onnx2tf"
}
```

Add to `export_routes.py`:
- `GET /api/export/microcontroller/{job_id}` → returns .zip (FileResponse)

#### Fix 4: BrainFlow Hardware Streaming

BrainFlow is a Python library that provides a unified API for 30+ EEG devices (OpenBCI Cyton/Ganglion, Muse, PiEEG, etc.). One API, any device.

Add to `backend/core/inference.py`:

```python
class HardwareStreamer:
    """
    Manages a live EEG hardware connection via BrainFlow.
    One instance stored in SessionStore.hardware_streamer.
    """
    def connect(self, board_id: int, serial_port: str = None) -> DataSession
    def start_stream(self) -> None
    def get_window(self, n_samples: int) -> np.ndarray  # returns (n_channels, n_samples)
    def stop_stream(self) -> None
    def disconnect(self) -> None
```

Add `hardware_streamer: HardwareStreamer | None = None` to SessionStore.

Add to `backend/api/inference_routes.py`:
- `GET /api/inference/hardware/boards` — returns dict of common board IDs and names
- `POST /api/inference/hardware/connect` — body: `{board_id, serial_port?, ip_address?}` → connects and returns session metadata
- `POST /api/inference/hardware/disconnect` — disconnects cleanly
- `WS /ws/inference/hardware` — WebSocket that reads from BrainFlow, applies pipeline, runs model, streams predictions

The hardware WebSocket follows the same message format as the existing simulated stream WebSocket so the frontend can use the same `useWebSocket` hook.

Update `InferencePage.jsx`:
- Add a "Source" toggle at the top: **Simulated** | **Hardware**
- Hardware mode shows: Board selector dropdown, Serial port input (optional), Connect/Disconnect button
- Once connected, Start Inference works the same as simulated mode
- Show hardware status badge: Connected (green) / Disconnected (grey)

#### UI Updates for Export Page

Update `ExportPage.jsx` to show four export cards:

| Card | Files | When to use |
|---|---|---|
| **Python Script** | `inference.py` | Laptop, desktop, Raspberry Pi 4/5 |
| **ONNX Package** | `model.onnx` + `inference_onnx.py` | Raspberry Pi 3/Zero, edge devices |
| **Microcontroller** | ZIP with .h + .cpp + README | ESP32-S3, STM32, Nano 33 BLE |
| **Config JSON** | `pipeline_config.json` | Reference / manual use |

The Microcontroller card shows a warning badge: "Requires TensorFlow installed on your machine to generate."

---

## Part 2 Roadmap

| Session | Status | What gets built |
|---|---|---|
| **P2-S1** | **Current** | Fix inference script preprocessing + ONNX export + microcontroller export + BrainFlow hardware streaming |
| P2-S2 | Upcoming | Tauri desktop app conversion |
| P2-S3 | Upcoming | TBD based on user feedback |

---

## Known Issues Carried From Part 1

- **Inference script missing preprocessing** — fixed in P2-S1
- **Session nicknames reset on refresh** — stored in Zustand only, intentional for now
- **ICA step has no progress indicator** — long recordings can take minutes silently
- **No `.gitignore`** — `exports/` and `backend/temp/` should be excluded from git

---

## Code Standards

### Python
- Type hints on all function parameters and return values
- Docstring on every class and public function
- Named constants, no magic numbers
- `logging` module, never `print()`
- `@dataclass` for data structures

### React / JavaScript
- Functional components only
- `async/await` for API calls, never `.then()`
- All API calls use `API_BASE_URL` from `frontend/src/config.js`

### General
- Every API endpoint returns `{ success, data, error }` — no exceptions
- Human-readable error messages

---

## Running the Project

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# Frontend: http://localhost:5173  |  API: http://localhost:8000  |  Docs: http://localhost:8000/docs
```

---

## Final Reminder

You are a coding assistant, not the decision-maker.

Every session:
1. Read this file completely
2. Explain before building (Rule 1)
3. Build exactly what is specified
4. Explain connections (Rule 3)
5. Flag any unspecified decisions (Rule 4)
