# CLAUDE.md — OpenEEG Project Memory

> This file is the single source of truth for the entire project.
> Read this file completely before writing a single line of code.
> Every decision about structure, naming, libraries, and patterns is defined here.
> Do not invent alternatives. Do not introduce unlisted libraries. Follow this file exactly.

---

## Mandatory Behavior Rules

These rules apply in every single session, without exception.

### Rule 1 — Explain Before You Build
Before writing any code, you must explain:
- What you are about to build
- Why it is being built this way (not just what it does, but the reasoning behind the approach)
- What files you will create or modify and why each one exists
- What the code will connect to or depend on

This explanation should be written in plain English that a beginner can follow.
Do not use jargon without explaining it.
Do not skip this step even if the task seems obvious.

**Format your explanation like this before any code block:**

```
=== BEFORE I BUILD ===

What I'm building: [describe the feature or module]
Why this approach: [explain the technical reasoning]
Files I'll create/modify:
  - path/to/file.py → [what this file is responsible for]
  - path/to/file.jsx → [what this file is responsible for]
What this connects to: [what calls this, what this calls]
======================
```

### Rule 2 — Explain Each Code Block Inline
Every non-trivial code block must have a plain-English comment above it explaining:
- What the block is doing
- Why it is written this way (if there's a meaningful reason)

Example:
```python
# We use a Butterworth filter here because it has a maximally flat frequency response
# in the passband — meaning it doesn't distort the signal amplitude within the range
# we care about. Order 4 is a standard choice for EEG: strong enough to reject noise
# without introducing phase distortion that would misalign events in time.
sos = butter(4, [low, high], btype='band', fs=sample_rate, output='sos')
```

### Rule 3 — Explain Connections Between Parts
When you finish a task, write a short section called `=== HOW THIS CONNECTS ===` that explains:
- What other part of the system will call or use what you just built
- What the next logical step is
- Any assumptions the next task depends on

### Rule 4 — Flag Decisions
If you make a decision that isn't specified in this file (e.g. choosing between two valid approaches), you must flag it:
```
=== DECISION ===
I chose X over Y because [reason]. If you want Y instead, change [specific thing].
================
```

### Rule 5 — Never Silently Deviate
If something in a session prompt contradicts or is unclear relative to this file, stop and ask before proceeding. Do not guess. Do not silently pick one interpretation.

---

## What This Project Is

**OpenEEG** is an open-source, beginner-friendly EEG (electroencephalography) development platform.

EEG measures electrical brain activity through electrodes on the scalp. The raw data is a set of voltage signals over time — one signal per electrode (called a "channel"). To do anything useful with it (detect mental states, control a prosthetic, classify blinks, etc.), you need to: clean the signal, segment it into windows, label those windows, train a machine learning model, and then run that model in real time.

Right now there's no beginner-friendly tool that does all of this in one place. This project builds that tool.

**The user journey in one line:**
Import EEG data → Preprocess signal → Label segments → Train a model → Test live predictions → Export everything

**Design philosophy:** Like the Arduino IDE. Simple enough for a student on day one. Fully editable for an expert.

---

## Tech Stack

Every library and framework listed here was chosen deliberately. Do not add libraries not on this list without flagging it as a decision (Rule 4).

### Frontend

| Tool | What it is | Why we use it |
|---|---|---|
| **React** | UI framework | Our app has complex, changing state — live signal charts, pipeline steps being added/removed, real-time predictions. React's component model handles this cleanly. |
| **Vite** | Build tool / dev server | Faster than Create React App. Handles hot-reloading during development. You mostly don't interact with it directly. |
| **TailwindCSS** | Utility-first CSS framework | Lets us style components without writing separate CSS files. Claude Code generates Tailwind reliably and consistently. |
| **Zustand** | State management | The currently loaded EEG session needs to be accessible from multiple pages (pipeline page, labeling page, training page). Zustand is a simple global store. Redux would be overkill. |
| **Recharts** | Charting library | For displaying EEG waveforms, training loss curves, and accuracy charts. Pure React, no external dependencies. |

### Backend

| Tool | What it is | Why we use it |
|---|---|---|
| **Python 3.11** | Language | Non-negotiable for anything EEG/ML. The entire scientific computing ecosystem is Python-first. |
| **FastAPI** | Web framework | Turns our Python functions into an API that the frontend can call over HTTP. Chosen over Flask because it has built-in WebSocket support (needed for real-time streaming) and auto-generates API docs. |
| **Uvicorn** | ASGI server | Runs the FastAPI app. You start the backend with `uvicorn main:app --reload`. |
| **NumPy** | Array computing | All EEG signal data is stored and manipulated as NumPy arrays. Shape convention: `(n_channels, n_samples)`. |
| **SciPy** | Scientific computing | Has battle-tested signal filtering implementations (bandpass, notch). We do not write filters from scratch. |
| **MNE-Python** | EEG-specific library | Standard library for reading EEG file formats (EDF, BDF). Also used for channel operations. Without it, reading `.edf` files would require writing a binary parser. |
| **PyTorch** | Deep learning | For building and training neural networks (CNN in MVP, LSTM later). More flexible than Keras for custom architectures. |
| **scikit-learn** | Classical ML | For SVM and Logistic Regression (post-MVP). These are one-liners in scikit-learn. No need for PyTorch for simple models. |
| **python-multipart** | File upload handling | Required by FastAPI to accept file uploads (EEG data files). |
| **websockets** | WebSocket support | Required for real-time EEG streaming and live inference pushing from backend to frontend. |

---

## Directory Structure

This is the exact structure of the project. Every file has a defined purpose. Do not create files outside this structure without explanation.

```
openeeg/
│
├── CLAUDE.md                         ← This file. Read before every session.
├── README.md                         ← How to run the project (human-readable)
│
├── frontend/
│   ├── index.html                    ← HTML entry point (Vite uses this)
│   ├── vite.config.js                ← Vite configuration
│   ├── tailwind.config.js            ← Tailwind configuration
│   ├── package.json                  ← Frontend dependencies
│   └── src/
│       ├── main.jsx                  ← React app entry point (mounts App)
│       ├── App.jsx                   ← Root component, routing, sidebar layout
│       ├── components/               ← Reusable UI pieces used across pages
│       │   ├── Sidebar.jsx           ← Navigation sidebar
│       │   ├── StatusBar.jsx         ← Shows current session info at top
│       │   ├── PipelineStep.jsx      ← One step card in the pipeline builder
│       │   └── WaveformChart.jsx     ← EEG waveform display using Recharts
│       ├── pages/                    ← Full-page views, one per workflow step
│       │   ├── ImportPage.jsx        ← File upload + session manager
│       │   ├── PreprocessPage.jsx    ← Pipeline builder UI
│       │   ├── LabelPage.jsx         ← Segment labeling tool
│       │   ├── TrainPage.jsx         ← Model config + training trigger
│       │   ├── InferencePage.jsx     ← Live prediction display
│       │   └── ExportPage.jsx        ← Export options and download
│       ├── hooks/                    ← Custom React hooks
│       │   ├── useSession.js         ← Hook for reading/writing current session from Zustand
│       │   └── useWebSocket.js       ← Hook for managing WebSocket connection
│       └── store/
│           └── appStore.js           ← Zustand global store definition
│
├── backend/
│   ├── main.py                       ← FastAPI app: creates app, registers all routers, CORS
│   ├── requirements.txt              ← All Python dependencies
│   │
│   ├── api/                          ← Route handlers only. Thin layer — no logic here.
│   │   ├── data_routes.py            ← /api/data/* endpoints
│   │   ├── pipeline_routes.py        ← /api/pipeline/* endpoints
│   │   ├── label_routes.py           ← /api/labels/* endpoints
│   │   ├── train_routes.py           ← /api/train/* endpoints
│   │   ├── inference_routes.py       ← /api/inference/* + WebSocket endpoint
│   │   └── export_routes.py          ← /api/export/* endpoints
│   │
│   ├── core/                         ← All actual logic. No HTTP knowledge here.
│   │   ├── data_loader.py            ← Load CSV/EDF files → DataSession objects
│   │   ├── preprocessing.py          ← Pipeline steps + Pipeline class
│   │   ├── models.py                 ← CNN architecture definition (PyTorch)
│   │   ├── trainer.py                ← Training loop, validation, metrics
│   │   ├── inference.py              ← Load model, run prediction on a window
│   │   └── export.py                 ← Serialize pipeline + model to exportable files
│   │
│   ├── sessions/
│   │   └── session_store.py          ← In-memory singleton store for all session data
│   │
│   └── plugins/                      ← Empty in MVP. Future: user-added custom steps/models.
│
├── configs/                          ← Saved pipeline config JSON files
├── datasets/                         ← Sample EEG data files for testing
│   └── sample_eeg.csv                ← Synthetic 8-channel EEG at 256Hz
└── exports/                          ← Output folder for exported models/scripts
```

### The key architectural rule: `api/` vs `core/`

The `api/` folder contains **route handlers only** — they receive HTTP requests, validate inputs, call functions from `core/`, and return responses. They contain zero signal processing or ML logic.

The `core/` folder contains **all real logic** — preprocessing, training, inference. These files have no knowledge of HTTP, requests, or responses. They are pure Python.

**Why this separation matters:**
- You can test any `core/` function without running a web server
- If you wanted to replace FastAPI with something else, you'd only rewrite `api/`, not your entire ML pipeline
- Claude Code stays organized across sessions — it always knows where logic lives

---

## Core Data Structures

These are the canonical data structures for the entire project. Every session must use these exactly. Never invent a new structure that duplicates these.

### 1. `DataSession` — the standard container for loaded EEG data

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class DataSession:
    session_id: str              # UUID, generated at load time
    signals: np.ndarray          # Shape: (n_channels, n_samples) — ALWAYS this shape
    sample_rate: int             # Hz, e.g. 256
    channel_names: list[str]     # e.g. ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "T3", "T4"]
    duration_seconds: float      # Total recording length
    file_path: str               # Original file path
    created_at: str              # ISO timestamp string
    metadata: dict = field(default_factory=dict)  # Optional extras
```

**Why `(n_channels, n_samples)` shape?**
MNE-Python uses this convention. Most EEG papers use this convention. All our preprocessing functions expect this shape. Keeping it consistent means we never have to transpose arrays mid-pipeline, which is a common source of bugs.

### 2. `PipelineConfig` — a saved preprocessing pipeline as JSON

```json
{
  "pipeline_id": "uuid-string",
  "session_id": "uuid-string",
  "sample_rate": 256,
  "steps": [
    {"step": "bandpass", "low_hz": 1.0, "high_hz": 40.0},
    {"step": "notch",    "freq_hz": 60.0, "quality_factor": 30.0},
    {"step": "normalize","method": "zscore"},
    {"step": "window",   "window_sec": 1.0, "overlap_ratio": 0.5}
  ]
}
```

**Why store the pipeline as JSON?**
A user can open this file, read it, understand it, and hand-edit it. It's transparent. It also means you can reproduce any result exactly by re-running the same config. This is what "not a black box" looks like in practice.

### 3. `LabeledWindow` — one labeled EEG segment ready for training

```python
@dataclass
class LabeledWindow:
    window_id: str               # UUID
    data: np.ndarray             # Shape: (n_channels, n_samples_in_window)
    label: str                   # e.g. "rest", "left", "right", "blink"
    start_time_sec: float        # Where in the original recording this window starts
    session_id: str              # Which session this came from
```

### 4. `ModelConfig` — defines the model to train

```json
{
  "model_type": "CNN",
  "classes": ["rest", "left", "right"],
  "input_channels": 8,
  "input_samples": 256,
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 32
}
```

### 5. `APIResponse` — every API endpoint returns this shape

```python
# Every route returns this. No exceptions.
{
  "success": True,        # bool
  "data": { ... },        # the actual payload, structure varies per endpoint
  "error": None           # string if success=False, None if success=True
}
```

**Why a consistent response shape?**
The frontend always knows exactly where to look. `if (response.success)` then use `response.data`. If not, show `response.error`. Without this, each route invents its own shape and the frontend code becomes inconsistent and fragile.

---

## Preprocessing Step Specification

Each preprocessing step is a class that follows this exact interface:

```python
class PreprocessingStep:
    def apply(self, signals: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply this step. Input and output are both (n_channels, n_samples)."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize to the JSON config format."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, config: dict) -> 'PreprocessingStep':
        """Reconstruct from a JSON config dict."""
        raise NotImplementedError
```

### Steps to implement (MVP):

**`BandpassFilter(low_hz: float, high_hz: float)`**
- Uses a Butterworth filter (order 4) via `scipy.signal.butter` + `sosfiltfilt`
- `sosfiltfilt` applies the filter forward AND backward to achieve zero phase distortion
- Typical EEG range: 1–40 Hz (removes DC drift below 1Hz, rejects muscle artifacts above 40Hz)

**`NotchFilter(freq_hz: float, quality_factor: float = 30.0)`**
- Uses `scipy.signal.iirnotch` to create a notch (band-stop) filter
- Removes power line interference: 60Hz in North America, 50Hz in Europe
- Quality factor controls how narrow the notch is — 30.0 is standard for EEG

**`Normalize(method: str = 'zscore')`**
- `zscore`: subtract mean, divide by std. Makes signals comparable across channels and subjects.
- `minmax`: scale to [0, 1] range. Less common but sometimes needed for certain models.
- Normalization happens per-channel (not across all channels at once)

**`SlidingWindow(window_sec: float, overlap_ratio: float)`**
- Cuts the continuous signal into fixed-length overlapping windows
- `window_sec=1.0, overlap_ratio=0.5` means 1-second windows with 50% overlap
- Returns a `list[np.ndarray]` — this step changes the return type (it's always the last step in a pipeline)
- Step size in samples = `window_samples * (1 - overlap_ratio)`

### `STEP_REGISTRY` — maps string names to classes

```python
STEP_REGISTRY = {
    "bandpass":  BandpassFilter,
    "notch":     NotchFilter,
    "normalize": Normalize,
    "window":    SlidingWindow,
}
```

This is how `Pipeline.from_dict()` reconstructs a pipeline from a saved JSON config — it looks up the class by the `"step"` string key.

---

## CNN Architecture Specification (MVP Model)

This is the only model in the MVP. Implement it exactly.

```
Input: (batch_size, 1, n_channels, n_samples)
  ↓
Conv2d(1, 16, kernel_size=(1, 32), padding='same')  → (batch, 16, n_channels, n_samples)
BatchNorm2d(16)
ELU activation
  ↓
Conv2d(16, 32, kernel_size=(n_channels, 1))          → (batch, 32, 1, n_samples)
BatchNorm2d(32)
ELU activation
AvgPool2d(kernel_size=(1, 4))                        → (batch, 32, 1, n_samples/4)
Dropout(0.5)
  ↓
Conv2d(32, 32, kernel_size=(1, 16), padding='same')  → (batch, 32, 1, n_samples/4)
BatchNorm2d(32)
ELU activation
AvgPool2d(kernel_size=(1, 8))                        → (batch, 32, 1, n_samples/32)
Dropout(0.5)
  ↓
Flatten
Linear(32 * (n_samples // 32), n_classes)
Output: class logits (use CrossEntropyLoss during training)
```

**Why this architecture?**
This is a simplified EEGNet-inspired CNN. The first conv layer applies filters along the time axis (temporal features). The second conv layer across channels (spatial features — which electrodes activate together). Average pooling downsamples aggressively to reduce overfitting on small EEG datasets, which is common in BCI work.

---

## API Endpoint Map

### Data Routes (`/api/data/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/data/upload` | Upload .csv or .edf file → returns session_id + metadata |
| GET | `/api/data/sessions` | List all loaded sessions (metadata only, no signal arrays) |
| GET | `/api/data/{session_id}` | Get single session metadata |
| DELETE | `/api/data/{session_id}` | Remove session from store |

### Pipeline Routes (`/api/pipeline/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/pipeline/run` | Run pipeline config on a session → returns run_id + window count |
| GET | `/api/pipeline/steps` | List all available steps + their parameters + defaults |
| POST | `/api/pipeline/save` | Save pipeline config JSON to `configs/` folder |
| GET | `/api/pipeline/saved` | List saved pipeline config files |

### Label Routes (`/api/labels/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/labels/save` | Save a list of LabeledWindow objects |
| GET | `/api/labels/{session_id}` | Get all labels for a session |
| DELETE | `/api/labels/{session_id}` | Clear all labels for a session |

### Train Routes (`/api/train/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/train/start` | Start training with a ModelConfig → returns job_id |
| GET | `/api/train/{job_id}/status` | Get training progress (epoch, loss, accuracy) |
| GET | `/api/train/{job_id}/results` | Get final metrics + confusion matrix |

### Inference Routes (`/api/inference/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/inference/predict` | Run single prediction on a window → returns class + confidence |
| WS | `/ws/inference/stream` | WebSocket: push windows in, receive predictions in real time |

### Export Routes (`/api/export/`)
| Method | Path | What it does |
|---|---|---|
| GET | `/api/export/pipeline/{pipeline_id}` | Download pipeline config as JSON |
| GET | `/api/export/model/{job_id}` | Download trained model weights (.pt file) |
| GET | `/api/export/script/{job_id}` | Download auto-generated Python inference script |

---

## SessionStore Specification

```python
# backend/sessions/session_store.py
# A singleton in-memory store. All routes access data through this.
# "Singleton" means there is only one instance of this object for the
# entire lifetime of the running backend server.

class SessionStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sessions = {}       # session_id → DataSession
            cls._instance.pipeline_runs = {}  # run_id → list[np.ndarray] (windows)
            cls._instance.labeled_data = {}   # session_id → list[LabeledWindow]
            cls._instance.training_jobs = {}  # job_id → training state dict
        return cls._instance
```

All route files import and use this: `store = SessionStore()`

---

## MVP Scope

### Build these 8 things. Nothing else.

1. EEG file import — CSV and EDF
2. Preprocessing pipeline — bandpass, notch, normalize, sliding windows
3. Manual segment labeling UI
4. CNN model training on labeled data
5. Train/validation results display (loss curve, accuracy, confusion matrix)
6. Real-time prediction view via WebSocket (using simulated EEG stream, not real hardware)
7. Pipeline config export as JSON
8. Basic session manager

### Do NOT build any of these in MVP:
- LSTM, SVM, or Logistic Regression models
- AI assistant or natural language pipeline editing
- Plugin system or marketplace
- TinyML or embedded export
- Real hardware EEG streaming (use a sine-wave simulator)
- Cloud storage or user accounts
- Any kind of database (use in-memory + local files only)

---

## Code Standards

These apply to every file in every session.

### Python
- Type hints on every function parameter and return value
- Docstring on every class and every public function
- No magic numbers — use named constants at the top of the file
- All preprocessing steps must be **pure functions**: same input always produces same output, no side effects
- Use `@dataclass` for data structures

### React / JavaScript
- Functional components only — no class components
- Each component file exports one default component
- Props must be documented with a comment above the component
- Use `async/await` for API calls, never `.then()` chains

### General
- Every API endpoint returns the `APIResponse` shape defined above
- Never `print()` for logging in production code — use Python's `logging` module
- Error messages must be human-readable — "Sample rate must be a positive integer" not "invalid sr"

---

## Running the Project

```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
# Backend API available at http://localhost:8000
# FastAPI auto-docs at http://localhost:8000/docs
```

---

## Session Roadmap

Each session builds one complete vertical slice. Always start a new session by saying: **"Read CLAUDE.md before doing anything."**

| Session | What gets built |
|---|---|
| **1** | Project scaffold + data import backend + preprocessing backend |
| **2** | Import page UI + session manager UI + waveform display |
| **3** | Labeling page UI + label save/load API |
| **4** | CNN model + training endpoint + training progress via WebSocket |
| **5** | Training results page (loss curves, confusion matrix) |
| **6** | Inference engine + real-time prediction WebSocket + inference page UI |
| **7** | Export system (pipeline JSON, model weights, inference script) |
| **8** | Pipeline builder UI + session polish + README |

---

## Final Reminder

You are a coding assistant on this project, not the decision-maker. The decision-maker is the human reading this file.

Your job in every session:
1. Read this file
2. Explain what you're about to do and why (Rule 1)
3. Build exactly what was asked, following the patterns here
4. Explain how what you built connects to the rest of the system (Rule 3)
5. Flag any decision you make that isn't explicitly covered here (Rule 4)

When in doubt, explain more, not less.
