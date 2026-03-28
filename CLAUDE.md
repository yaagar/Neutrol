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
| onnx | Export PyTorch model to ONNX format |
| onnxruntime | Run ONNX models on edge devices |
| onnxscript | Required by torch.onnx.export in PyTorch 2.x |
| brainflow | Unified API for 30+ live EEG hardware devices |
| pyserial | NEW (P2-S2) — Send predictions over serial/COM port to microcontrollers |

---

## Directory Structure

```
Neutrol/
├── CLAUDE.md
├── README.md
├── frontend/
│   ├── src/
│   │   ├── config.js                      ← API_BASE_URL = 'http://localhost:8000'
│   │   ├── App.jsx
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
│   │   │   ├── PreprocessPage.jsx         ← P2-S4: dashboard multi-graph view
│   │   │   ├── LabelPage.jsx
│   │   │   ├── TrainPage.jsx              ← P2-S3: fine-tuning + augmentation
│   │   │   ├── InferencePage.jsx          ← P2-S2: source toggle + output system
│   │   │   └── ExportPage.jsx
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
│   │   ├── train_routes.py                ← P2-S3: fine-tuning endpoint
│   │   ├── inference_routes.py            ← P2-S2: hardware + file playback + output
│   │   └── export_routes.py
│   ├── core/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── analysis.py
│   │   ├── models.py
│   │   ├── trainer.py                     ← P2-S3: fine-tuning support
│   │   ├── inference.py                   ← P2-S2: HardwareStreamer + OutputDispatcher
│   │   └── export.py
│   └── sessions/session_store.py          ← P2-S2: output_config + hardware_streamer
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

PipelineConfig JSON (notch before bandpass — notch removes 60Hz harmonics first):
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

Every API endpoint returns:
```json
{ "success": true, "data": {}, "error": null }
```

---

## SessionStore

```python
class SessionStore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sessions = {}          # session_id → DataSession
            cls._instance.pipeline_runs = {}     # run_id → {windows, start_times, sample_rate, pipeline_config}
            cls._instance.labeled_data = {}      # run_id → list[LabeledWindow]
            cls._instance.training_jobs = {}     # job_id → training state dict
            cls._instance.event_markers = {}     # session_id → list[EventMarker]
            cls._instance.hardware_streamer = None  # HardwareStreamer | None
            cls._instance.output_config = {}     # current output destination config
        return cls._instance
```

---

## API Endpoint Map

### Data `/api/data/`
| POST | `/upload` | Upload .csv or .edf → session_id + metadata |
| POST | `/upload-events` | Upload event file → EventMarker list |
| GET | `/sessions` | List all sessions |
| GET | `/{session_id}` | Single session metadata |
| DELETE | `/{session_id}` | Remove session |

### Pipeline `/api/pipeline/`
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
| GET | `/by-run/{run_id}` | Labels for run (define BEFORE `/{session_id}`) |
| DELETE | `/{session_id}` | Clear labels |

### Train `/api/train/`
| POST | `/validate-config` | Validate ModelConfig + layer shapes |
| GET | `/layer-types` | All layer types + parameters |
| POST | `/start` | Start training from scratch → job_id |
| POST | `/finetune` | NEW: Continue training from existing model → job_id |
| GET | `/{job_id}/status` | Progress (epoch, loss, accuracy) |
| GET | `/{job_id}/results` | Final metrics + confusion matrix |
| GET | `/history/{model_name}` | NEW: All training runs for a model name |

### Inference `/api/inference/`
| POST | `/predict` | Single prediction → class + confidence |
| WS | `/ws/stream-run` | WebSocket: simulated or file playback stream |
| GET | `/hardware/boards` | List supported BrainFlow board IDs |
| POST | `/hardware/connect` | Connect to live EEG device |
| POST | `/hardware/disconnect` | Disconnect hardware |
| WS | `/ws/hardware` | WebSocket: stream from real hardware |
| GET | `/output/serial-ports` | NEW: List available COM/serial ports |
| POST | `/output/config` | NEW: Set output destination config |
| GET | `/output/config` | NEW: Get current output config |

### Export `/api/export/`
| GET | `/pipeline/{run_id}` | Pipeline config JSON |
| GET | `/model/{job_id}` | Model weights .pt |
| GET | `/script/{job_id}` | Python inference script (preprocessing embedded) |
| GET | `/onnx/{job_id}` | ONNX model file |
| GET | `/onnx-script/{job_id}` | Python ONNX inference script |
| GET | `/microcontroller/{job_id}` | ZIP with model_data.h + neutrol_inference.cpp + README |

---

## Part 1 — Built (Sessions 1–9) ✅

All MVP features complete. Key items:
- EEG file import (CSV, EDF, event files)
- Preprocessing pipeline (notch, bandpass, reref, normalize, window, channel selector, epoch rejection, ICA)
- Signal analysis (FFT, PSD, band power, spectrogram) with Before/After toggle
- Manual + event-file auto-labeling
- Dynamic CNN training (fully editable layer architecture)
- Training results (loss curves, confusion matrix)
- Live inference with simulated stream (WebSocket)
- Export: pipeline JSON, model .pt, Python inference script, ONNX, microcontroller ZIP
- UI polish: step reordering, session rename, sidebar progress dots

---

## Part 2 — What Gets Built

---

### P2-S2 — Inference Page: Source Toggle + Output System

**Goal:** Complete the inference page so predictions are actually useful for real projects. Three input sources, two output destinations.

#### Input Source Toggle

Three modes on InferencePage, shown as a toggle at the top:

**1. Simulated** (already works)
Generates synthetic EEG sine waves. Always available. Use for testing without hardware.

**2. File Playback** (new)
User selects any loaded DataSession. The backend replays the session's signal array in real time — it slices windows sequentially with a configurable delay between them, sending each through the pipeline and model exactly as if it were live data. This lets anyone test the full inference pipeline without hardware.

Backend: add `playback` mode to the existing `/ws/stream-run` WebSocket endpoint. Client sends:
```json
{ "type": "start", "mode": "playback", "session_id": "...", "job_id": "...", "delay_ms": 200 }
```
Backend slices windows from the session's signal array sequentially, applies pipeline, runs model, streams predictions. When signal ends, sends `{ "type": "done" }`.

**3. Hardware** (BrainFlow — backend exists, UI was not built)
Connects to a real EEG device. Uses the existing `/ws/hardware` endpoint and `/hardware/connect` + `/hardware/disconnect` endpoints already built in P2-S1.

UI for hardware mode:
- Board selector dropdown (populated from `GET /api/inference/hardware/boards`)
- Serial port text input (shown only if board `needs_serial` is true)
- IP address input (shown only if board `needs_ip` is true)
- Connect / Disconnect button
- Status badge: 🟢 Connected / ⚫ Disconnected
- Once connected, Start Inference button activates

All three modes send predictions in the same WebSocket message format:
```json
{ "label": "left", "confidence": 0.87, "timestamp": 1234567890 }
```
So the prediction display UI (badge, confidence bars, history ribbon) is shared across all three modes.

#### Output Destination System

After every prediction, Neutrol can dispatch it to an external destination. This is how the prediction leaves Neutrol and reaches the user's robot, prosthetic, script, or device.

Two output destinations (can be active simultaneously):

**Output 1 — WebSocket Server**
Neutrol starts a WebSocket server on a configurable port (default 8765). External code connects to this server as a client and receives predictions as JSON. This works for: Python scripts, Node.js, any language that supports WebSocket.

Why WebSocket and not HTTP: robots and prosthetics need continuous real-time predictions. HTTP requires a new request for every prediction. WebSocket is a persistent connection — predictions arrive instantly as they're generated with no request overhead.

Example external Python client (shown in the UI as a code snippet):
```python
import asyncio, websockets, json

async def receive_predictions():
    async with websockets.connect("ws://localhost:8765") as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"Label: {data['label']}, Confidence: {data['confidence']:.1%}")
            # Your control code here

asyncio.run(receive_predictions())
```

**Output 2 — Serial Port**
Sends a configurable string over a serial/COM port after each prediction. This is the most direct way to control an Arduino, ESP32, or any microcontroller reading serial.

User configures:
- Port name (e.g. COM3, /dev/ttyUSB0) — dropdown populated from `GET /api/inference/output/serial-ports`
- Baud rate (default 9600)
- Per-class output string — what gets sent when each class is predicted:
  - "rest" → "0\n"
  - "left" → "L\n"
  - "right" → "R\n"
  These strings are fully editable by the user.

Uses `pyserial` library on the backend.

#### Action Mapper UI

On the InferencePage, below the source selector, show an **Output** section with two cards:

**WebSocket Output card:**
- Toggle: Enabled / Disabled
- Port number input (default 8765)
- Status: "Server running on ws://localhost:8765" or "Stopped"
- Code snippet tab showing the Python client example above

**Serial Output card:**
- Toggle: Enabled / Disabled
- Port dropdown (from serial-ports endpoint)
- Baud rate dropdown (9600, 115200, 57600)
- Per-class string table — editable rows, one per class label:
  | Class | Send string |
  |---|---|
  | rest | 0\n |
  | left | L\n |
  | right | R\n |

Backend: `OutputDispatcher` class in `backend/core/inference.py`:

```python
class OutputDispatcher:
    """
    Receives predictions and dispatches them to configured output destinations.
    Called after every prediction during inference, regardless of source.
    """
    
    def __init__(self):
        self.ws_server = None         # asyncio WebSocket server instance
        self.ws_clients = set()       # connected external clients
        self.serial_conn = None       # pyserial Serial instance
        self.serial_map = {}          # label → string to send
        self.ws_port = 8765
        self.enabled_ws = False
        self.enabled_serial = False
    
    async def start_ws_server(self, port: int) -> None
    async def stop_ws_server(self) -> None
    def start_serial(self, port: str, baud: int, label_map: dict) -> None
    def stop_serial(self) -> None
    async def dispatch(self, label: str, confidence: float) -> None
        # Sends to all connected WS clients AND serial port if enabled
```

Add `output_dispatcher: OutputDispatcher` to SessionStore (one instance, shared).

New API endpoints in `inference_routes.py`:
- `GET /api/inference/output/serial-ports` — returns list of available COM/serial ports using `serial.tools.list_ports`
- `POST /api/inference/output/config` — body: `{ ws_enabled, ws_port, serial_enabled, serial_port, serial_baud, serial_map }` — configures the dispatcher
- `GET /api/inference/output/config` — returns current output config and status

The inference WebSocket handlers (`/ws/stream-run` and `/ws/hardware`) call `store.output_dispatcher.dispatch(label, confidence)` after every prediction.

---

### P2-S3 — Training Improvements

**Goal:** Fine-tuning, model history, comparison, and data augmentation.

#### Fine-Tuning / Incremental Training

What it is: load an existing trained model's weights, then continue training on a new dataset without resetting those weights. The model keeps what it learned and improves further.

When it makes sense:
- Train on Person A's data, then fine-tune on Person B's small dataset → better cross-subject performance
- Train on Session 1, fine-tune on Session 2 → adapts to signal drift over time
- Train on one task, fine-tune on a related task → transfer learning

Constraint: the model architecture and number of output classes must be identical between the original training run and the fine-tuning run. If classes don't match, return a clear error.

Backend changes to `backend/core/trainer.py`:

Add optional parameter `finetune_from_job_id: str = None` to `run_training()`. When provided:
1. Load the checkpoint for `finetune_from_job_id` from the exports folder
2. Rebuild the model using `build_model(config)` — same architecture
3. Load the saved weights: `model.load_state_dict(checkpoint['model_state_dict'])`
4. Validate that the class lists match — raise clear error if they don't
5. Continue the normal training loop with these pre-loaded weights instead of random init
6. Save the new checkpoint as a new job_id (does not overwrite the original)
7. In the checkpoint, store `finetuned_from: job_id` so the history is traceable

New API endpoint in `train_routes.py`:

`POST /api/train/finetune`
- Body: `{ "base_job_id": str, "run_id": str, "model_config": ModelConfig dict }`
- `base_job_id`: the existing trained model to continue from
- `run_id`: the new labeled data to train on
- `model_config`: must match the base model's architecture exactly
- Validates class match before starting
- Returns `{ job_id }` — new job that can be polled with existing status endpoint

**UI changes to TrainPage:**
- Add a "Training Mode" toggle at the top: **Train from scratch** | **Fine-tune existing model**
- In fine-tune mode: show a "Base Model" dropdown listing all trained models
- Show a validation message: "✓ Compatible — same architecture and classes" or "✗ Class mismatch: base has [rest, left] but current labels have [rest, left, right]"
- Start button changes to "Fine-tune" in this mode

#### Model History View

A new section on TrainPage (or a separate tab) that shows all training runs grouped by model name.

For each model name, show a table of all its training runs:
| Run | Date | Dataset | Epochs | Val Accuracy | Fine-tuned from |
|---|---|---|---|---|---|
| Run 1 | 2025-03-01 | session_abc | 50 | 72.3% | — |
| Run 2 | 2025-03-03 | session_def | 30 | 81.1% | Run 1 |
| Run 3 | 2025-03-05 | session_ghi | 20 | 87.4% | Run 2 |

Also show a small line chart of accuracy over runs — visually shows improvement.

Backend: `GET /api/train/history/{model_name}` — scans the exports folder for all `.pt` checkpoints whose `model_name` field matches, returns them sorted by date with their metadata.

#### Model Comparison Table

A "Compare Models" button on TrainPage (or ExportPage) that opens a side-by-side comparison of selected models.

Shows per model: accuracy, precision, recall, F1 score per class.

Example:
| Metric | Model A | Model B | Model C |
|---|---|---|---|
| Overall Accuracy | 72.3% | 81.1% | 87.4% |
| rest precision | 0.71 | 0.82 | 0.89 |
| left precision | 0.68 | 0.79 | 0.85 |
| right precision | 0.78 | 0.82 | 0.88 |

No new backend endpoint needed — uses existing `GET /api/train/{job_id}/results` for each selected model.

#### Data Augmentation

On the TrainPage, add an "Augmentation" section (collapsible, off by default).

Three augmentation options with toggles:
1. **Gaussian Noise** — adds small random noise to each window. Strength parameter (default 0.05 = 5% of signal std). Makes model robust to signal variation.
2. **Time Shift** — shifts each window slightly left or right in time (±10% of window length). Makes model robust to temporal misalignment.
3. **Channel Dropout** — randomly zeroes one channel per window during training (not during validation). Makes model robust to electrode artifacts.

When any augmentation is enabled, the training data is expanded: each window becomes N copies with different augmentations applied. Default multiplier: 3x (each window generates 2 augmented copies + original).

Backend: add `augmentation_config` to the training start request body:
```json
{
  "augmentation": {
    "gaussian_noise": { "enabled": true, "strength": 0.05 },
    "time_shift": { "enabled": true, "max_shift_ratio": 0.1 },
    "channel_dropout": { "enabled": false },
    "multiplier": 3
  }
}
```

Implement `augment_windows(windows, labels, config)` in `trainer.py` — applies augmentations and returns expanded windows and labels lists.

---

### P2-S4 — PreprocessPage Dashboard View

**Goal:** Let users see all four signal analysis charts simultaneously in a 2×2 grid.

Add a **View Mode** toggle at the top right of the analysis section:
- **📊 Tabs** (current behavior — one chart at a time, full width)
- **⊞ Dashboard** (all four charts in a 2×2 grid, each at half-size)

In dashboard mode, all four charts render simultaneously:
```
┌─────────────────┬─────────────────┐
│   Waveform      │   Frequency     │
│   (Time)        │   (PSD)         │
├─────────────────┼─────────────────┤
│   Band Power    │   Spectrogram   │
│   (Bar chart)   │   (Heatmap)     │
└─────────────────┴─────────────────┘
```

Each chart in dashboard mode has a small "expand" button (⤢) that temporarily switches back to tab view for that specific chart at full width.

The Before/After toggle and channel selector still work in dashboard mode — they apply to all four charts simultaneously.

This is a **frontend-only change**. All four backend analysis endpoints already exist. The dashboard just calls all four at once instead of one at a time.

Implementation note: The four API calls happen in parallel using `Promise.all()` — not sequentially — so the dashboard loads all charts at the same time.

---

### P2-S5 — Tauri Desktop App Conversion

**Goal:** Package Neutrol as an installable desktop app for Windows, macOS, and Linux.

**Why Tauri over Electron:** Tauri uses the OS's native webview (no bundled Chromium) making the installer ~5MB instead of ~150MB. The backend runs as a sidecar process.

Key decisions for Tauri conversion:
1. The React frontend is already built — Tauri wraps it with no code changes
2. The FastAPI backend gets packaged using PyInstaller into a standalone executable
3. A Tauri "sidecar" config starts and stops the Python executable with the app
4. `frontend/src/config.js` already centralizes the API URL — change `API_BASE_URL` to `http://127.0.0.1:8000` (127.0.0.1 instead of localhost avoids some OS-level DNS lookup delays)

Files to create:
- `src-tauri/` — Tauri project folder
- `src-tauri/tauri.conf.json` — Tauri config (app name, window size, sidecar config)
- `src-tauri/src/main.rs` — minimal Rust entry point (mostly boilerplate)
- `build_backend.py` — PyInstaller script to package the Python backend
- `package_app.sh` / `package_app.bat` — one-command build script

Sidecar approach: in `tauri.conf.json`, register the packaged Python executable as a sidecar. Tauri starts it when the app opens and kills it when the app closes.

```json
{
  "tauri": {
    "bundle": {
      "externalBin": ["binaries/neutrol-backend"]
    }
  }
}
```

The Python backend is packaged with PyInstaller:
```bash
pyinstaller --onefile --name neutrol-backend backend/main.py
```

---

## Part 2 Session 2 — What Was Actually Built ✅

### Source Toggle — three inference modes

**`frontend/src/pages/InferencePage.jsx`** — complete rewrite. Source toggle at top switches between three modes without re-mounting the prediction display:

- **Simulated** — original behaviour, `mode: "run"` on `/ws/stream-run`
- **File Playback** — new `mode: "playback"` on `/ws/stream-run`. User picks any loaded DataSession + a reference pipeline run (for window shape). Backend re-slices the raw signal into windows on the fly — nothing pre-computed
- **Hardware** — board dropdown (from `/hardware/boards`), optional serial port input, Connect/Disconnect button, live status badge. Uses `/ws/hardware` once connected

All three feed the same prediction display: large class badge + confidence bars + scrolling history ribbon. No duplicate code.

### File Playback mode

**`backend/api/inference_routes.py`** — `/ws/stream-run` now reads a `mode` field from the start message:
- `"run"` → original: iterates `store.pipeline_runs[run_id]["windows"]`
- `"playback"` → new: fetches `store.sessions[session_id].signals`, slices into `n_samp` windows sequentially. Window shape comes from the reference run's `window_shape` field, or falls back to the model's `n_channels`/`n_samples`

### OutputDispatcher

**`backend/core/inference.py`** — new `OutputDispatcher` class. Two independent outputs, both optional, both can run simultaneously:

1. **WebSocket server** — Neutrol acts as a WS server (`websockets.serve`) on a configurable port. External Python/Node.js code connects as a client and receives `{label, confidence, timestamp}` JSON after every prediction. Client set managed as a Python `set`; dead clients removed on send error.

2. **Serial port** — opens a `pyserial.Serial` connection. After each prediction, looks up the label in a user-configured `serial_map` (e.g. `{"left": "L\n"}`) and writes that string to the port. Escape sequences in map values are decoded (so `"L\\n"` becomes `"L"` + newline).

`dispatch(label, confidence)` is called after every prediction in both `/ws/stream-run` and `/ws/hardware` — regardless of which source is active.

**`backend/sessions/session_store.py`** — added `output_dispatcher = OutputDispatcher()` and `output_config = {}` to singleton.

### Output API endpoints

**`backend/api/inference_routes.py`** — three new endpoints:
- `GET /output/serial-ports` — `serial.tools.list_ports.comports()`, returns `{port, description}` list + `pyserial_available` flag
- `POST /output/config` — body: `{ws_enabled, ws_port, serial_enabled, serial_port, serial_baud, serial_map}`. Starts/stops/restarts WS server, opens/closes serial port. Saves config to `store.output_config`
- `GET /output/config` — returns live status (`ws_running`, `ws_clients`, `serial_open`) + last saved config

### Output UI

InferencePage output section: two `OutputCard` tiles, each with a `ToggleSwitch`:
- **WebSocket card** — port input, running status with client count, collapsible Python code snippet, Apply button
- **Serial card** — port dropdown (refreshable), baud dropdown (9600/57600/115200), per-class string table (editable rows, auto-populated when class names arrive from the WS), Apply button

### Integration test results (all passed)

| Test | Result |
|---|---|
| run mode | 10 predictions streamed |
| playback mode | 10 predictions sliced from raw session signal |
| Output WS external client | Received `{label, confidence, timestamp}` |
| WS server stop | `ws_running: false` |
| Hardware (synthetic) | connect/disconnect OK |
| Serial ports endpoint | `pyserial_available: true` |

### Known issues from P2-S2

- Serial port send not tested with real hardware (no physical device in test environment — endpoint verified to open/close correctly)
- Output WS server port must not conflict with the FastAPI server (port 8000); default 8765 is safe

---

## Part 2 Session Roadmap

| Session | Status | What gets built |
|---|---|---|
| P2-S1 | ✅ Done | Inference script preprocessing + ONNX + microcontroller export + BrainFlow backend |
| P2-S2 | ✅ Done | InferencePage source toggle (Simulated/File Playback/Hardware) + Output system (WebSocket server + Serial port) |
| P2-S3 | ✅ Done | Fine-tuning training + model history + model comparison + data augmentation |
| P2-S4 | ✅ Done | PreprocessPage dashboard multi-graph view |
| P2-S4b | ✅ Done | Signal quality stats + Custom pipeline step + Custom CNN layer + Model import |
| P2-S5 | ✅ Done | Tauri desktop app conversion |

---

## Part 2 Session 3 — What Was Actually Built ✅

### Fine-tuning

**`backend/core/trainer.py`** — `run_training()` accepts two new optional params:
- `finetune_from_job_id`: loads `{job_id}.pt` from exports/, validates classes match exactly, calls `model.load_state_dict()` before the training loop. Class mismatch → fail() with clear error. `finetuned_from` saved in checkpoint for traceability.
- `augmentation_config`: passed to `augment_windows()` before train/val split.

`augment_windows(X, y, config)` — expands training data `multiplier` times:
- Gaussian noise: adds `strength × std` random normal per window
- Time shift: `np.roll()` ±`max_shift_ratio × n_samples` time steps
- Channel dropout: zeros one random channel per window copy
- Returns `(X_aug, y_aug)` with shape `(N × multiplier, ch, samp)`

**`backend/api/train_routes.py`** — two new endpoints:
- `POST /api/train/finetune` — validates base model exists + class match up-front (instant error), then launches `run_training(finetune_from_job_id=...)`. Returns new `job_id` — base model unchanged.
- `GET /api/train/history/{model_name}` — scans `store.training_jobs` for done jobs with matching `model_name`, returns sorted by `started_at` with val accuracy + `finetuned_from` chain.

`StartTrainingRequest` updated to accept optional `augmentation_config`.

### TrainPage UI

**`frontend/src/pages/TrainPage.jsx`** — complete rewrite with new sections:

**Training Mode toggle** — "Train from scratch" | "Fine-tune existing model":
- Fine-tune mode: base model dropdown (all done jobs), compatibility message (✓ green / ✗ red), start button label changes to "Fine-tune"
- Fine-tune result badge shows "Fine-tuned from model {id}…"

**Augmentation section** (collapsible, "active" badge when any option on):
- Gaussian Noise checkbox + strength input
- Time Shift checkbox + max_shift_ratio input
- Channel Dropout checkbox
- Multiplier input (dataset size × N)

**Model History section** (collapsible):
- Model name text input + Load button
- Val accuracy line chart across runs (Run 1 → Run N)
- Table: Run | Date | Epochs | Val Accuracy | Fine-tuned from

**Model Comparison section** (collapsible):
- Checkbox list of all done models
- Side-by-side table computed from confusion matrix: Overall Accuracy, per-class precision/recall/F1
- Results fetched on-demand via `GET /api/train/{job_id}/results`

## Part 2 Session 4 — What Was Actually Built ✅

### PreprocessPage Dashboard View

**`frontend/src/pages/PreprocessPage.jsx`** — frontend-only change. All 4 backend analysis endpoints unchanged.

**View mode toggle** added to the top of the analysis panel:
- `☰ Tabs` — existing behaviour (one chart at a time, full width)
- `⊞ Dashboard` — new 2×2 grid showing all 4 charts simultaneously

**Dashboard behaviour:**
- Switching to Dashboard fires `fetchAllDashboard()` which uses `Promise.all` to fire all 4 API calls simultaneously. Each tile has its own loading flag (`dashLoading.{time,frequency,bandpower,spectrogram}`) so they fill in independently as results arrive.
- Each tile has a `⤢` expand button that shows that chart full-width with a `← Back to Dashboard` nav link.
- Before/After toggle and channel selector apply to all 4 charts simultaneously (previously only shown for the active tab — in dashboard mode they're always visible).
- `viewMode` and `expandedChart` are local state (not persisted in Zustand) — intentional, tab layout preferences don't need to survive navigation.

---

### Label Class Manager (hotfix after P2-S3)

**Problem:** Fine-tuning gave "Class mismatch" errors because `'unknown'` labels in the current dataset were included in the class comparison.

**Fixes:**
- `backend/api/train_routes.py` — `/finetune` now filters `'unknown'` from `current_classes` before comparing against base model classes (consistent with how scratch training always excluded `'unknown'`).
- `frontend/src/pages/LabelPage.jsx` — new **Class Manager** panel below the label summary:
  - Shows each class with its window count as a removable chip
  - Clicking × shows an inline confirmation: "Remove N windows? [Confirm] [Cancel]"
  - On confirm: filters that class out of local `labels` state, then re-saves to backend
  - Follows the MNE-Python standard: filter labels at the labeling stage, raw EEG data stays intact in the session
  - Hint text explains the fine-tuning use case (why class lists must match)

This is standard EEG practice: EEGLAB and MNE both filter event types at the epoch/labeling stage, not by deleting raw data.

---

## Part 2 Session 4b — What Was Actually Built ✅

### Signal Quality Stats (Analyze button — Time tab)

**`backend/core/analysis.py`** — new `compute_signal_stats(signals, sample_rate, channel_names)`:
- Per-channel: RMS, peak-to-peak, std, Pearson kurtosis (fisher=False, normal=3.0), max gradient (sample-to-sample), zero-crossing rate (sign changes/second)
- `artifact_warning: True` when kurtosis > 5 (spike artifact threshold) or peak-to-peak > 200 µV
- `scipy.stats.kurtosis` imported inside function (not at top-level)

**`backend/api/analysis_routes.py`** — new `POST /api/analysis/stats` endpoint:
- Body: `{session_id, run_id}` (same as other analysis endpoints)
- Returns `{channels: [{channel, rms, peak_to_peak, std, kurtosis, max_gradient, zero_crossing_rate, artifact_warning}, ...]}}`

**`frontend/src/pages/PreprocessPage.jsx`** — stats card below waveform on Time tab:
- `fetchStats()` async function, `statsData`/`statsLoading`/`statsOpen` state
- Analyze button on Time tab now calls both `loadWaveform()` + `fetchStats()`
- Collapsible table with color-coded kurtosis column (red when > 5) and ⚠ Artifact? badge
- Hint text links artifact warnings to Epoch Rejection / Channel Selector steps

### Custom Pipeline Step

**`backend/core/preprocessing.py`** — new `CustomStep(PreprocessingStep)` class:
- `code` field: user-defined `def process(signal, sample_rate): return signal`
- `exec()` in scoped namespace: `{'np': np, 'numpy': np, 'scipy': scipy_signal}`
- Validates: function named `process` exists, returns `np.ndarray`, shape matches input exactly
- Registered as `"custom"` in `STEP_REGISTRY` — auto-discovered by `/api/pipeline/steps`

**`frontend/src/pages/PreprocessPage.jsx`** — new "Custom Step" in `STEP_PRESETS`:
- Renders a multi-line code textarea (green-on-black font) when `stepConfig.step === 'custom'`
- Name field lets user label the step; code field directly edits `stepConfig.code`

### Custom CNN Layer

**`backend/core/models.py`** — new `"CustomLayer"` in `LAYER_TYPES_SCHEMA`:
- Parameter: `code` (type `"code"`) — user writes `class CustomLayer(nn.Module):`
- `_build_layer_modules`: `exec(code, {'nn': nn, 'torch': torch})` → validates class exists and extends `nn.Module` → instantiates `CustomLayer()`

**`frontend/src/components/ModelLayerEditor.jsx`** — new `"code"` type in `renderParamInput`:
- Multi-line textarea with amber border, `font-mono`, dark background
- Hint text shows available namespace (`nn`, `torch`) and required class name

### Import External Model

**`backend/api/train_routes.py`** — new `POST /api/train/import` endpoint:
- `UploadFile` + `Form` — multipart upload; saves temp file, loads with `torch.load`
- Validates: `model_state_dict` and `class_names` keys required
- Copies validated checkpoint to `exports/{job_id}.pt`
- Creates `store.training_jobs[job_id]` with `status: 'done'`, `imported: True` flag
- Returns `{job_id, model_name, class_names, n_channels, n_samples}`
- Temp file always cleaned up in `finally` block

**`frontend/src/pages/TrainPage.jsx`** — new "Import External Model" collapsible section (Section 9):
- File picker (`.pt` only) + optional name input + Import button
- `importModel()` uses `FormData` for multipart upload
- On success: shows green card with job_id/class_names; calls `fetchAllJobs()` to refresh dropdowns immediately

---

## Part 2 Session 5 — What Was Actually Built ✅

### Tauri Desktop App Conversion

**Goal:** Package Neutrol as an installable desktop app for Windows, macOS, and Linux without changing any backend or frontend application code.

**Strategy:** Tauri v2 wraps the existing Vite/React frontend. The FastAPI backend is bundled with PyInstaller as a sidecar executable that Tauri launches on startup and kills on exit.

#### Files created

**`src-tauri/tauri.conf.json`** — Tauri v2 app config:
- App identity: `com.neutrol.app`, version `0.1.0`
- Build: `frontendDist: ../frontend/dist`, `devUrl: http://127.0.0.1:5173`
- Window: 1400×860, min 1024×640
- Sidecar: `externalBin: ["binaries/neutrol-backend"]`

**`src-tauri/Cargo.toml`** — Rust dependencies:
- `tauri = "2"`, `tauri-plugin-shell = "2"`, `serde`, `serde_json`
- Release profile: `lto = true`, `strip = true`, `opt-level = "s"` (small binary)

**`src-tauri/build.rs`** — required Tauri build script (boilerplate).

**`src-tauri/src/main.rs`** — entry point; `windows_subsystem = "windows"` hides the terminal in release builds.

**`src-tauri/src/lib.rs`** — application logic:
- Registers `tauri_plugin_shell`
- In `setup`: spawns `neutrol-backend` sidecar, stores `CommandChild` in managed state (`SidecarChild(Mutex<Option<CommandChild>>)`)
- `on_window_event(Destroyed)`: takes the child handle and calls `child.kill()` — backend is always cleaned up

**`src-tauri/capabilities/default.json`** — shell plugin permissions: `shell:allow-spawn`, `shell:allow-execute`, `shell:allow-kill`

**`src-tauri/binaries/`** — placeholder directory for the compiled backend binary (not committed — generated by `build_backend.py`)

**`build_backend.py`** — PyInstaller packaging script:
- Detects platform target triple (`x86_64-pc-windows-msvc`, `aarch64-apple-darwin`, etc.)
- Runs `pyinstaller --onefile --name neutrol-backend backend/main.py` with hidden imports for uvicorn/anyio/scipy
- Copies output to `src-tauri/binaries/neutrol-backend-<triple>[.exe]`

**`package_app.bat`** / **`package_app.sh`** — one-command build scripts:
1. Run `python build_backend.py`
2. Run `npm install` in frontend/
3. Run `npm run tauri build` (builds frontend → Tauri packages everything)
4. Output: `src-tauri/target/release/bundle/`

#### Frontend changes

**`frontend/src/config.js`** — changed `localhost` → `127.0.0.1`:
- `http://127.0.0.1:8000` and `ws://127.0.0.1:8000`
- Avoids IPv6/DNS lookup delay that `localhost` can cause on some OS versions

**`frontend/vite.config.js`** — added Tauri-required server config:
- `port: 5173, strictPort: true, host: '127.0.0.1'`
- `clearScreen: false` so Vite and Tauri output don't fight each other

**`frontend/package.json`** — added `@tauri-apps/cli: "^2"` devDep and `"tauri": "tauri"` script

**`.gitignore`** — added `src-tauri/target/`, `src-tauri/gen/`, `build/`, `dist/`, `*.spec`, `src-tauri/binaries/`

#### How to build

```bash
# 1. Install Python deps (if not done): pip install -r backend/requirements.txt pyinstaller
# 2. Install Rust: https://rustup.rs
# 3. Install frontend deps: cd frontend && npm install

# Windows — one command:
package_app.bat

# Linux / macOS — one command:
bash package_app.sh
```

#### How to develop (no packaging)

```bash
# Run backend normally:
cd backend && uvicorn main:app --reload --port 8000

# Run Tauri dev mode (hot-reload frontend in native window):
cd frontend && npm run tauri dev
```

---

## Known Issues

- **Old .pt checkpoints missing pipeline_config** — models trained before P2-S1 get "No preprocessing recorded" in generated script; retrain to fix
- **Microcontroller export optional deps** — `tensorflow` + `onnx2tf` not in requirements.txt; returns HTTP 422 with install message if missing
- **Session nicknames reset on refresh** — Zustand only, intentional for now
- **ICA step has no progress indicator** — long recordings can take minutes silently

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
