# CLAUDE.md — Neutrol Project Memory

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

**Neutrol** is an open-source, beginner-friendly EEG (electroencephalography) development platform.

EEG measures electrical brain activity through electrodes on the scalp. The raw data is a set of voltage signals over time — one signal per electrode (called a "channel"). To do anything useful with it (detect mental states, control a prosthetic, classify blinks, etc.), you need to: clean the signal, segment it into windows, label those windows, train a machine learning model, and then run that model in real time.

Right now there's no beginner-friendly tool that does all of this in one place. This project builds that tool.

**The user journey in one line:**
Import EEG data → Preprocess + Analyze signal → Label segments (manually or from event file) → Train a model → Test live predictions → Export everything

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
| **Recharts** | Charting library | For displaying EEG waveforms, training loss curves, accuracy charts, FFT plots, and band power bars. Pure React, no external dependencies. |

### Backend

| Tool | What it is | Why we use it |
|---|---|---|
| **Python 3.11** | Language | Non-negotiable for anything EEG/ML. The entire scientific computing ecosystem is Python-first. |
| **FastAPI** | Web framework | Turns our Python functions into an API that the frontend can call over HTTP. Chosen over Flask because it has built-in WebSocket support (needed for real-time streaming) and auto-generates API docs. |
| **Uvicorn** | ASGI server | Runs the FastAPI app. You start the backend with `uvicorn main:app --reload`. |
| **NumPy** | Array computing | All EEG signal data is stored and manipulated as NumPy arrays. Shape convention: `(n_channels, n_samples)`. |
| **SciPy** | Scientific computing | Signal filtering (bandpass, notch), FFT, Welch PSD, STFT spectrogram. We do not implement these from scratch. |
| **MNE-Python** | EEG-specific library | Reading EDF/BDF files, parsing event/annotation files, ICA artifact removal, re-referencing. The industry standard for EEG in Python. |
| **PyTorch** | Deep learning | For building and training neural networks (CNN, LSTM). Used for the dynamic model builder — layers are constructed at runtime from a config. |
| **scikit-learn** | Classical ML | For SVM and Logistic Regression (post-MVP). Also used for confusion matrix and classification report utilities. |
| **python-multipart** | File upload handling | Required by FastAPI to accept file uploads (EEG data files). |
| **websockets** | WebSocket support | Required for real-time EEG streaming and live inference pushing from backend to frontend. |

---

## Directory Structure

This is the exact structure of the project. Every file has a defined purpose. Do not create files outside this structure without explanation.

```
Neutrol/
│
├── CLAUDE.md                             ← This file. Read before every session.
├── README.md                             ← How to run the project (human-readable)
│
├── frontend/
│   ├── index.html                        ← HTML entry point (Vite uses this)
│   ├── vite.config.js                    ← Vite configuration
│   ├── tailwind.config.js                ← Tailwind configuration
│   ├── package.json                      ← Frontend dependencies
│   └── src/
│       ├── main.jsx                      ← React app entry point (mounts App)
│       ├── App.jsx                       ← Root component, routing, sidebar layout
│       ├── App.css                       ← Global styles
│       ├── index.css                     ← Tailwind base imports
│       ├── components/                   ← Reusable UI pieces used across pages
│       │   ├── Sidebar.jsx               ← Navigation sidebar
│       │   ├── StatusBar.jsx             ← Shows current session info at top
│       │   ├── PipelineStep.jsx          ← One step card in the pipeline builder
│       │   ├── WaveformChart.jsx         ← EEG time-domain display (already built)
│       │   ├── FrequencyChart.jsx        ← FFT/PSD frequency-domain plot (NEW)
│       │   ├── BandPowerChart.jsx        ← Delta/theta/alpha/beta/gamma bar chart (NEW)
│       │   ├── SpectrogramChart.jsx      ← Time-frequency heatmap (NEW)
│       │   └── ModelLayerEditor.jsx      ← Visual CNN layer builder (NEW)
│       ├── pages/                        ← Full-page views, one per workflow step
│       │   ├── ImportPage.jsx            ← File upload + session manager
│       │   ├── PreprocessPage.jsx        ← Pipeline builder + signal analysis tabs (UPDATED)
│       │   ├── LabelPage.jsx             ← Segment labeling + event file import (UPDATED)
│       │   ├── TrainPage.jsx             ← Model config + layer editor + training (UPDATED)
│       │   ├── InferencePage.jsx         ← Live prediction display
│       │   └── ExportPage.jsx            ← Export options and download
│       ├── hooks/                        ← Custom React hooks
│       │   ├── useSession.js             ← Hook for reading/writing current session from Zustand
│       │   └── useWebSocket.js           ← Hook for managing WebSocket connection
│       └── store/
│           └── appStore.js               ← Zustand global store definition
│
├── backend/
│   ├── main.py                           ← FastAPI app: creates app, registers all routers, CORS
│   ├── requirements.txt                  ← All Python dependencies
│   │
│   ├── api/                              ← Route handlers only. Thin layer — no logic here.
│   │   ├── data_routes.py                ← /api/data/* endpoints (UPDATED)
│   │   ├── pipeline_routes.py            ← /api/pipeline/* endpoints
│   │   ├── analysis_routes.py            ← /api/analysis/* endpoints (NEW)
│   │   ├── label_routes.py               ← /api/labels/* endpoints (UPDATED)
│   │   ├── train_routes.py               ← /api/train/* endpoints (UPDATED)
│   │   ├── inference_routes.py           ← /api/inference/* + WebSocket endpoint
│   │   └── export_routes.py              ← /api/export/* endpoints
│   │
│   ├── core/                             ← All actual logic. No HTTP knowledge here.
│   │   ├── data_loader.py                ← Load CSV/EDF/event files → DataSession + EventMarkers
│   │   ├── preprocessing.py              ← Pipeline steps + Pipeline class (UPDATED)
│   │   ├── analysis.py                   ← FFT, PSD, band power, spectrogram (NEW)
│   │   ├── models.py                     ← Dynamic CNN builder from layer config (UPDATED)
│   │   ├── trainer.py                    ← Training loop, validation, metrics
│   │   ├── inference.py                  ← Load model, run prediction on a window
│   │   └── export.py                     ← Serialize pipeline + model to exportable files
│   │
│   ├── sessions/
│   │   └── session_store.py              ← In-memory singleton store (UPDATED)
│   │
│   └── plugins/                          ← Empty now. Future: user-added custom steps/models.
│
├── configs/                              ← Saved pipeline config JSON files
├── datasets/                             ← Sample EEG data files for testing
│   └── sample_eeg.csv                    ← Synthetic 8-channel EEG at 256Hz
├── exports/                              ← Output folder for exported models/scripts
└── backend/temp/                         ← Temporary storage for uploaded files
```

### The key architectural rule: `api/` vs `core/`

The `api/` folder contains **route handlers only** — they receive HTTP requests, validate inputs, call functions from `core/`, and return responses. They contain zero signal processing or ML logic.

The `core/` folder contains **all real logic** — preprocessing, training, inference, analysis. These files have no knowledge of HTTP, requests, or responses. They are pure Python.

---

## Core Data Structures

These are the canonical data structures for the entire project. Every session must use these exactly. Never invent a new structure that duplicates these.

### 1. `DataSession` — the standard container for loaded EEG data

```python
from dataclasses import dataclass, field
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
    metadata: dict = field(default_factory=dict)
```

**Why `(n_channels, n_samples)` shape?**
MNE-Python uses this convention. Most EEG papers use this convention. All preprocessing functions expect this shape. Keeping it consistent means we never have to transpose arrays mid-pipeline.

### 2. `PipelineConfig` — a saved preprocessing pipeline as JSON

```json
{
  "pipeline_id": "uuid-string",
  "session_id": "uuid-string",
  "sample_rate": 256,
  "steps": [
    {"step": "bandpass",  "low_hz": 1.0, "high_hz": 40.0},
    {"step": "notch",     "freq_hz": 60.0, "quality_factor": 30.0},
    {"step": "reref",     "method": "average"},
    {"step": "normalize", "method": "zscore"},
    {"step": "window",    "window_sec": 1.0, "overlap_ratio": 0.5}
  ]
}
```

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

### 4. `EventMarker` — a single event parsed from an event/annotation file

```python
@dataclass
class EventMarker:
    onset_sec: float             # When the event starts (seconds from recording start)
    duration_sec: float          # How long the event lasts (0.0 if instantaneous)
    label: str                   # The class label, e.g. "T1", "T2", "rest"
    description: str             # Raw annotation string from the file
```

**What is an event file?**
Many EEG datasets store class labels separately from the signal. A `.vmrk`, `.eve`, `.tsv`, or EDF annotation file lists timestamps and event codes — "at 3.5 seconds, event T1 occurred, lasting 4 seconds." Parsing this and pairing it with the signal lets us skip manual labeling entirely.

The PhysioNet EDF files already in `backend/temp/` (S001R03.edf, S001R04.edf) use embedded EDF annotations. MNE reads these automatically with `mne.events_from_annotations(raw)`.

### 5. `ModelConfig` — defines the full model architecture and training settings

The model is **fully dynamic** — it is built at runtime from the `layers` array. Users can change any layer's type, size, activation, or parameters from the UI without writing Python.

```json
{
  "model_type": "CNN",
  "classes": ["rest", "left", "right"],
  "input_channels": 8,
  "input_samples": 256,
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 32,
  "layers": [
    {
      "type": "Conv2d",
      "out_channels": 16,
      "kernel_size": [1, 32],
      "padding": "same",
      "activation": "ELU",
      "batch_norm": true
    },
    {
      "type": "DepthwiseConv2d",
      "depth_multiplier": 2,
      "kernel_size": ["n_channels", 1],
      "activation": "ELU",
      "batch_norm": true
    },
    {
      "type": "AvgPool2d",
      "kernel_size": [1, 4]
    },
    {
      "type": "Dropout",
      "rate": 0.5
    },
    {
      "type": "Conv2d",
      "out_channels": 32,
      "kernel_size": [1, 16],
      "padding": "same",
      "activation": "ELU",
      "batch_norm": true
    },
    {
      "type": "AvgPool2d",
      "kernel_size": [1, 8]
    },
    {
      "type": "Dropout",
      "rate": 0.5
    },
    {
      "type": "Flatten"
    },
    {
      "type": "Linear",
      "out_features": "n_classes"
    }
  ]
}
```

**Supported layer types:**

| Type | What it does | Parameters |
|---|---|---|
| `Conv2d` | 2D convolution — learns spatial and temporal filters | `out_channels`, `kernel_size`, `padding`, `activation`, `batch_norm` |
| `DepthwiseConv2d` | Convolution applied per input channel — learns channel-specific patterns | `depth_multiplier`, `kernel_size`, `activation`, `batch_norm` |
| `AvgPool2d` | Average pooling — downsamples by averaging blocks | `kernel_size` |
| `MaxPool2d` | Max pooling — downsamples by taking the max value | `kernel_size` |
| `Dropout` | Randomly zeroes outputs during training to prevent overfitting | `rate` (0.0–1.0) |
| `BatchNorm2d` | Normalizes layer outputs to stabilize training | none (auto-sized) |
| `Flatten` | Converts 2D feature maps to 1D vector for the final layer | none |
| `Linear` | Fully connected layer | `out_features` (use `"n_classes"` for final layer) |
| `LSTM` | Recurrent layer for sequential temporal patterns | `hidden_size`, `num_layers`, `dropout` |

**Supported activation functions** (used in `activation` field of any layer):

| Name | What it does | When to use |
|---|---|---|
| `ELU` | Smooth, allows small negative values | Default for EEG CNNs |
| `ReLU` | Sets negatives to zero | Most common general purpose |
| `LeakyReLU` | Like ReLU but passes small negatives | When ReLU causes dead neurons |
| `Tanh` | Outputs between -1 and 1 | Recurrent layers |
| `Sigmoid` | Outputs between 0 and 1 | Binary output only |
| `GELU` | Smooth ReLU variant | Deep networks / transformers |

**Special keyword values in layer configs:**
- `"n_channels"` in `kernel_size` → replaced at build time with actual `input_channels`
- `"n_classes"` in `out_features` → replaced at build time with `len(classes)`

### 6. `APIResponse` — every API endpoint returns this shape

```python
{
  "success": True,        # bool
  "data": { ... },        # the actual payload
  "error": None           # string if success=False, None if success=True
}
```

---

## Preprocessing Step Specification

Each step follows this base interface:

```python
class PreprocessingStep:
    def apply(self, signals: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError
    def to_dict(self) -> dict:
        raise NotImplementedError
    @classmethod
    def from_dict(cls, config: dict) -> 'PreprocessingStep':
        raise NotImplementedError
```

### Original steps (already implemented, do not modify):

**`BandpassFilter(low_hz, high_hz)`** — Butterworth order 4, `sosfiltfilt`, zero-phase.

**`NotchFilter(freq_hz, quality_factor=30.0)`** — `iirnotch`, removes power line noise.

**`Normalize(method='zscore')`** — per-channel zscore or minmax.

**`SlidingWindow(window_sec, overlap_ratio)`** — always the last step, returns `list[np.ndarray]`.

### New steps to implement in Session 4B:

**`Rereferencing(method: str = 'average')`**
- Re-referencing changes what each electrode's voltage is measured relative to.
- Raw EEG is relative to a single physical reference electrode. This introduces bias — the reference electrode's own brain activity contaminates every channel.
- `'average'`: subtract the mean of all channels from each channel. Makes the reference a neutral scalp average. This is the most common method in BCI research.
- `'linked_mastoid'`: subtracts the average of channels named 'M1' and 'M2'. Used in clinical EEG.
- Use MNE's `set_eeg_reference()` internally for correctness, then extract the resulting numpy array.

**`EpochRejection(threshold_uv: float = 100.0)`**
- Must be applied AFTER `SlidingWindow`. It receives `list[np.ndarray]` and returns a filtered `list[np.ndarray]`.
- Rejects any window where any sample in any channel exceeds ±threshold microvolts.
- Why: spikes of ±100µV or more are almost always artifacts (blink, jaw clench, electrode pop), not brain signal. Training on them will confuse the model.
- Log how many windows were rejected and why.

**`ChannelSelector(channels: list[str])`**
- Keeps only the specified channel names, drops the rest.
- Must be applied BEFORE windowing.
- Why: many EEG devices record 32–256 channels. For a motor imagery task, you only need ~8–16 channels over the motor cortex (C3, Cz, C4 and neighbors). Fewer channels = faster training, less noise.
- Raises a clear error if any requested channel name doesn't exist in the session.

**`ICAFilter(n_components: int = None, exclude_indices: list[int] = None)`**
- ICA (Independent Component Analysis) mathematically separates the mixed EEG signal into statistically independent source signals.
- Some sources correspond to artifacts: eye blinks appear as large deflections in frontal channels, heartbeat as a periodic signal across all channels. These components can be identified and removed.
- `n_components`: how many components to compute. Default = number of channels.
- `exclude_indices`: which component numbers to remove (e.g. `[0, 1]`). User identifies these by inspecting component topomaps (future UI feature).
- Uses MNE's `ICA` class.
- **Warning**: ICA is computationally expensive (minutes on long recordings). Always apply after bandpass filtering and before windowing. Never apply after normalization.

### Updated `STEP_REGISTRY`:

```python
STEP_REGISTRY = {
    "bandpass":  BandpassFilter,
    "notch":     NotchFilter,
    "normalize": Normalize,
    "window":    SlidingWindow,
    "reref":     Rereferencing,
    "reject":    EpochRejection,
    "channels":  ChannelSelector,
    "ica":       ICAFilter,
}
```

---

## Signal Analysis Specification (NEW — `backend/core/analysis.py`)

Signal analysis is separate from preprocessing. Preprocessing **transforms** the signal to clean it. Analysis **computes read-only information** from the signal to display it. Different purpose, different file.

None of these functions modify the signal — they only compute and return data for visualization.

### `compute_fft(signals: np.ndarray, sample_rate: int) -> dict`
- FFT (Fast Fourier Transform) converts a time-domain signal into the frequency domain.
- **What it tells you**: which frequencies are present and how strong they are.
- Uses `numpy.fft.rfft` (real FFT — since EEG is real-valued, only positive frequencies matter).
- Frequency resolution = `sample_rate / n_samples`. More data = finer resolution.
- Returns: `{ "frequencies": [0.5, 1.0, ...], "magnitudes": { "Fp1": [...], "Fp2": [...] } }`

### `compute_psd(signals: np.ndarray, sample_rate: int, nperseg: int = 256) -> dict`
- PSD (Power Spectral Density) using Welch's method.
- **Why Welch over raw FFT?** Welch divides the signal into overlapping segments, computes FFT on each, and averages them. This dramatically reduces noise in the estimate. Raw FFT on a long signal is noisy. Welch is smoother and much more reliable for EEG.
- Uses `scipy.signal.welch`.
- Returns: `{ "frequencies": [...], "power": { "Fp1": [...], ... } }`

### `compute_band_power(signals: np.ndarray, sample_rate: int) -> dict`
- Computes power within each standard EEG frequency band per channel.
- Bands and their meaning:
  - **Delta** (0.5–4 Hz): deep sleep, unconsciousness
  - **Theta** (4–8 Hz): drowsiness, meditation, memory encoding
  - **Alpha** (8–13 Hz): relaxed wakefulness, eyes closed — the most visible EEG rhythm
  - **Beta** (13–30 Hz): active thinking, focus, motor activity
  - **Gamma** (30–100 Hz): high-level cognition, sensory processing
- Uses Welch PSD internally, then integrates power within each band's frequency range using `scipy.integrate.trapezoid`.
- Returns: `{ "channels": ["Fp1", ...], "bands": { "delta": [...], "theta": [...], "alpha": [...], "beta": [...], "gamma": [...] } }`

### `compute_spectrogram(signals: np.ndarray, sample_rate: int, channel_index: int = 0) -> dict`
- Spectrogram shows how frequency content changes over time.
- **Why**: FFT gives one frequency picture of the whole signal. Spectrogram gives a movie — you can see alpha waves appear when eyes close, disappear when eyes open. Essential for understanding non-stationary EEG.
- Uses `scipy.signal.stft` (Short-Time Fourier Transform) — many short FFTs along the time axis, stacked into a 2D matrix.
- Computed per channel (all channels at once = too much data for the API).
- Returns: `{ "times": [...], "frequencies": [...], "power_db": [[...]] }` — 2D matrix, power in decibels.

---

## Event File Parsing Specification (NEW — additions to `data_loader.py`)

### `load_edf_with_events(file_path: str) -> tuple[DataSession, list[EventMarker]]`
- Loads the EDF file with MNE (same as existing `load_edf`)
- After loading, calls `mne.events_from_annotations(raw)` to extract embedded event markers
- Maps MNE's integer event IDs back to their string annotation labels
- Returns both the DataSession AND a pre-populated list of EventMarker objects
- If no annotations exist, returns the DataSession + an empty list (no error)

### `parse_event_file(file_path: str, format: str = 'auto') -> list[EventMarker]`
- Parses a standalone event file into EventMarkers
- `format='auto'`: detect format from file extension
- Supported formats:
  - `'bids_tsv'` — `.tsv` file with columns: `onset`, `duration`, `trial_type`
  - `'mne_eve'` — `.eve` file: tab-separated `sample_index`, `prev_id`, `event_id`
  - `'vmrk'` — BrainVision `.vmrk` marker file (text format)
  - `'edf_annotations'` — `.edf.event` file (MNE-exported annotations)

### `apply_events_to_windows(windows: list[np.ndarray], events: list[EventMarker], sample_rate: int, window_sec: float) -> list[LabeledWindow]`
- Takes windowed signal data + EventMarkers
- For each window, finds which event overlaps with that time range
- Rule: an event is assigned to a window if `event.onset_sec` falls within the window's time range
- Windows with no matching event get label `"unknown"` (user can filter these before training)
- Returns a fully-labeled list of LabeledWindow objects

---

## Dynamic CNN Builder Specification (NEW — updates to `models.py`)

### Why dynamic?
The original spec had a hardcoded CNN. Different EEG tasks need different architectures:
- A 64-channel dataset needs different spatial filters than an 8-channel one
- Seizure detection needs longer temporal receptive fields than blink detection
- Some users want to experiment with deeper networks or different activations

The dynamic builder reads the `layers` array from `ModelConfig` and constructs the PyTorch model at runtime.

### Functions to implement in `models.py`:

**`build_model(config: ModelConfig) -> nn.Module`**
- Iterates through `config.layers`
- Resolves special keywords: `"n_channels"` → `config.input_channels`, `"n_classes"` → `len(config.classes)`
- Dispatches each layer dict to the appropriate builder function
- Returns a ready-to-train `nn.Sequential` PyTorch model

**`get_activation(name: str) -> nn.Module`**
- Maps name strings to PyTorch activation modules
- Raises a clear `ValueError` listing all valid options if name is unrecognized

**`validate_model_config(config: ModelConfig) -> list[str]`**
- Returns a list of human-readable error strings (empty list = valid)
- Checks: last layer must be `Linear` with `out_features="n_classes"`, no `Flatten` before conv layers, dropout rate 0.0–1.0, at least one layer defined

**`estimate_output_shape(config: ModelConfig) -> list[list[int]]`**
- Runs a single dummy forward pass with a zero tensor of shape `(1, 1, input_channels, input_samples)`
- Returns the tensor shape after each layer: `[[1,16,8,256], [1,32,1,256], ...]`
- Used by the UI to show shape flowing through the network as the user edits it

### New API endpoints to add to `train_routes.py`:

`POST /api/train/validate-config`
- Body: ModelConfig dict
- Returns: `{ valid: bool, errors: list[str], layer_shapes: list }`

`GET /api/train/layer-types`
- Returns all supported layer types with their parameters, types, defaults, and plain-English descriptions
- Used by the frontend "Add Layer" dropdown

---

## Signal Analysis UI Specification (NEW — updates to `PreprocessPage.jsx`)

Add a **view toggle** above the signal display area with four tabs. The existing `WaveformChart` is the Time tab — do not modify it.

| Tab | Component | What it shows | API call |
|---|---|---|---|
| **Time** | `WaveformChart.jsx` | Voltage over time, all channels | none (already in state) |
| **Frequency** | `FrequencyChart.jsx` | FFT or PSD line chart | `POST /api/analysis/psd` |
| **Band Power** | `BandPowerChart.jsx` | Bar chart per band per channel | `POST /api/analysis/bandpower` |
| **Spectrogram** | `SpectrogramChart.jsx` | Time-frequency heatmap | `POST /api/analysis/spectrogram` |

- Channel selector dropdown: for Frequency and Spectrogram tabs, let user pick which channel(s) to show
- **Before/After toggle**: when a pipeline run exists, show a toggle to compare raw signal vs preprocessed signal in any view. This is one of the most powerful teaching features — users can literally see what each filter is doing.

---

## Full API Endpoint Map

### Data Routes (`/api/data/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/data/upload` | Upload .csv or .edf → session_id + metadata |
| POST | `/api/data/upload-events` | Upload standalone event file → parsed EventMarker list |
| GET | `/api/data/sessions` | List all sessions (metadata only) |
| GET | `/api/data/{session_id}` | Get single session metadata |
| DELETE | `/api/data/{session_id}` | Remove session |

### Pipeline Routes (`/api/pipeline/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/pipeline/run` | Run pipeline on session → run_id + window count |
| GET | `/api/pipeline/steps` | List all available steps + parameters + defaults |
| POST | `/api/pipeline/save` | Save pipeline config to `configs/` |
| GET | `/api/pipeline/saved` | List saved pipeline configs |

### Analysis Routes (`/api/analysis/`) — NEW
| Method | Path | What it does |
|---|---|---|
| POST | `/api/analysis/fft` | Compute FFT → per-channel frequencies + magnitudes |
| POST | `/api/analysis/psd` | Compute Welch PSD → per-channel frequencies + power |
| POST | `/api/analysis/bandpower` | Compute delta/theta/alpha/beta/gamma per channel |
| POST | `/api/analysis/spectrogram` | Compute STFT spectrogram for one channel |

### Label Routes (`/api/labels/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/labels/save` | Save LabeledWindow list (manual labeling) |
| POST | `/api/labels/from-events` | Auto-label windows from EventMarker list |
| GET | `/api/labels/{session_id}` | Get all labels for a session |
| DELETE | `/api/labels/{session_id}` | Clear all labels |

### Train Routes (`/api/train/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/train/validate-config` | Validate ModelConfig + return layer shapes |
| GET | `/api/train/layer-types` | Return all supported layer types + parameters |
| POST | `/api/train/start` | Start training → job_id |
| GET | `/api/train/{job_id}/status` | Training progress (epoch, loss, accuracy) |
| GET | `/api/train/{job_id}/results` | Final metrics + confusion matrix |

### Inference Routes (`/api/inference/`)
| Method | Path | What it does |
|---|---|---|
| POST | `/api/inference/predict` | Single prediction on a window → class + confidence |
| WS | `/ws/inference/stream` | WebSocket: stream windows in, receive predictions |

### Export Routes (`/api/export/`)
| Method | Path | What it does |
|---|---|---|
| GET | `/api/export/pipeline/{pipeline_id}` | Download pipeline config JSON |
| GET | `/api/export/model/{job_id}` | Download model weights (.pt) |
| GET | `/api/export/script/{job_id}` | Download auto-generated Python inference script |

---

## SessionStore Specification (UPDATED)

```python
class SessionStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sessions = {}          # session_id → DataSession
            cls._instance.pipeline_runs = {}     # run_id → list[np.ndarray] (windows)
            cls._instance.labeled_data = {}      # session_id → list[LabeledWindow]
            cls._instance.training_jobs = {}     # job_id → training state dict
            cls._instance.event_markers = {}     # session_id → list[EventMarker] (NEW)
        return cls._instance
```

The new `event_markers` dict stores parsed events keyed by session_id. They can be retrieved later and applied to any pipeline run for that session.

---

## ModelLayerEditor Component Specification (NEW — `frontend/src/components/ModelLayerEditor.jsx`)

This component renders the `layers` array of a `ModelConfig` as editable cards.

Each layer card shows:
- Layer type (dropdown — populated from `GET /api/train/layer-types`)
- All parameters for that layer type (input fields with their types and defaults)
- Activation function selector (dropdown showing all activation options, only for layers that support it)
- Delete button (red X)
- Drag handle to reorder layers

Below the layer list:
- **"+ Add Layer"** button → dropdown of all supported layer types
- **Shape display**: after calling validate-config, show the tensor shape flowing through each layer in small text under each card: e.g. `→ (1, 16, 8, 256)`
- **"Validate Architecture"** button → calls `POST /api/train/validate-config`, shows errors inline in red under affected layers
- The default state loads the EEGNet-inspired default architecture from `ModelConfig` above

---

## Code Standards

### Python
- Type hints on every function parameter and return value
- Docstring on every class and every public function
- No magic numbers — use named constants at the top of the file
- Preprocessing steps must be pure functions (same input = same output, no side effects)
- Use `@dataclass` for data structures
- Use `logging` module, never `print()`

### React / JavaScript
- Functional components only — no class components
- Each component file exports one default component
- Props documented with a comment above the component
- Use `async/await` for API calls, never `.then()` chains

### General
- Every API endpoint returns the `APIResponse` shape — no exceptions
- Error messages must be human-readable: "Sample rate must be a positive integer, got 'abc'" not "invalid sr"

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
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# FastAPI auto-docs: http://localhost:8000/docs
```

---

## Session Roadmap

Always start every new session with: **"Read CLAUDE.md before doing anything."**

| Session | Status | What gets built |
|---|---|---|
| 1 | ✅ Done | Scaffold + data import backend + preprocessing backend |
| 2 | ✅ Done | Import page UI + session manager + waveform display |
| 3 | ✅ Done | Labeling page UI + label save/load API |
| 4 | 🔄 Test first | CNN model + training endpoint + training progress WebSocket |
| **4B** | **Next** | Test + fix Session 4 → signal analysis backend (FFT/PSD/bands/spectrogram) → event file parsing → new preprocessing steps (reref, epoch rejection, channel selector, ICA) → analysis tabs on PreprocessPage |
| 5 | Upcoming | Dynamic CNN layer editor UI + model validation endpoint + layer shape display |
| 6 | Upcoming | Training results page (loss curves, accuracy, confusion matrix) |
| 7 | Upcoming | Inference engine + real-time WebSocket + inference page UI |
| 8 | Upcoming | Export system (pipeline JSON, model weights, Python inference script) |
| 9 | Upcoming | Polish — error handling, loading states, full README walkthrough |

---

## Final Reminder

You are a coding assistant on this project, not the decision-maker. The decision-maker is the human reading this file.

Your job in every session:
1. Read this file completely
2. Explain what you are about to do and why (Rule 1)
3. Build exactly what was asked, following every pattern defined here
4. Explain how what you built connects to the rest of the system (Rule 3)
5. Flag any decision you make that isn't explicitly covered here (Rule 4)

When in doubt, explain more, not less.
