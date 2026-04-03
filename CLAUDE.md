# CLAUDE.md — Neutrol Project Memory

> Single source of truth. Read completely before writing any code.
> Do not invent alternatives. Do not introduce unlisted libraries. Follow this file exactly.

---

## Mandatory Behavior Rules

**Rule 1 — Explain Before You Build:** Write `=== BEFORE I BUILD ===` block with: what you're building, why this approach, files you'll create/modify, what it connects to.

**Rule 2 — Explain Code Inline:** Every non-trivial block gets a plain-English comment.

**Rule 3 — Explain Connections:** After finishing, write `=== HOW THIS CONNECTS ===`.

**Rule 4 — Flag Decisions:** Any choice not specified here: `=== DECISION === I chose X over Y because [reason] ===`

**Rule 5 — Never Silently Deviate:** If a prompt contradicts this file, stop and ask.

---

## What This Project Is

**Neutrol** — open-source, beginner-friendly EEG development platform.

User journey: Import EEG → Preprocess + Analyze → Label segments → Train model → Test live → Export

Design philosophy: Like Arduino IDE / MATLAB. Simple for beginners, fully editable for experts.

---

## Tech Stack

**Frontend:** React + Vite, TailwindCSS, Zustand (global state), Recharts (charts)

**Backend:** Python 3.11, FastAPI + Uvicorn, NumPy, SciPy, MNE-Python, PyTorch, scikit-learn, python-multipart, websockets, onnx/onnxruntime/onnxscript, brainflow, pyserial

---

## Directory Structure

```
Neutrol/
├── frontend/src/
│   ├── config.js                      ← API_BASE_URL = 'http://127.0.0.1:8000'
│   ├── App.jsx, components/, hooks/, store/appStore.js
│   └── pages/
│       ├── ImportPage.jsx
│       ├── PreprocessPage.jsx         ← pipeline builder + 5-tab analysis + split view + electrode map
│       ├── LabelPage.jsx              ← keyboard shortcuts + timeline strip
│       ├── TrainPage.jsx              ← DL + Classical ML + ROC curves + arch JSON save/load
│       ├── InferencePage.jsx          ← Simulated/Playback/Hardware + prediction log + CSV export
│       └── ExportPage.jsx
├── backend/
│   ├── main.py, requirements.txt
│   ├── api/                           ← route handlers only (no logic)
│   └── core/                         ← all logic
│       ├── preprocessing.py, models.py, trainer.py
│       ├── classical_ml.py, inference.py, export.py, analysis.py
└── sessions/session_store.py
```

**Architecture rule:** `api/` = route handlers only, `core/` = all logic. Never mix.

---

## Core Data Structures

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
```

Every API endpoint returns: `{ "success": true, "data": {}, "error": null }`

`pipeline_runs` store: `run_id → {windows, start_times, sample_rate, pipeline_config}`

---

## Preprocessing Steps (STEP_REGISTRY)

| step key | description |
|---|---|
| `truncate` | Trim signal to time range |
| `bandpass` | Butterworth bandpass (low_hz, high_hz) |
| `highpass` | High-pass filter (cutoff_hz, filter_type, order) |
| `lowpass` | Low-pass filter (cutoff_hz, filter_type, order) |
| `notch` | IIR notch at freq_hz |
| `normalize` | zscore or minmax per channel |
| `window` | Sliding window (window_sec, overlap_ratio) |
| `reref` | Re-reference: average, linked_mastoid, or electrode (ref_channel) |
| `reject` | Epoch rejection (threshold_uv) — post-window |
| `channels` | Keep subset of channels |
| `ica` | ICA artifact removal (no progress indicator — can take minutes) |
| `resample` | Resample to target_rate_hz via scipy resample_poly |
| `custom` | User-defined Python function |

`Pipeline.run()` returns `(windows, final_sample_rate)` — accounts for resampling.

---

## DL Model Config

```json
{
  "classes": ["left", "right"],
  "input_channels": 8,
  "input_samples": 256,
  "layers": [...],
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 32,
  "optimizer": "adam",
  "weight_decay": 0.0,
  "early_stopping_patience": 0,
  "gradient_clip_norm": 0.0
}
```

**Layer types:** Conv2d, DepthwiseConv2d, AvgPool2d, MaxPool2d, Dropout, BatchNorm2d, BatchNorm1d, Flatten, Linear, Dense, LSTM, GRU, SimpleRNN, TemporalAttention, BahdanauAttention, CustomLayer

**Optimizers:** adam (default), adamw, rmsprop, sgd

---

## Classical ML

`POST /api/train/classical/start` → LDA, SVM, RandomForest, GBM, KMeans, GMM

- Supervised (LDA/SVM/RF/GBM): 5-fold CV, returns accuracy + confusion matrix + ROC data
- Unsupervised (KMeans/GMM): fits on all windows, returns `fit_metric` (inertia or log-likelihood), no confusion matrix
- Results endpoint exposes `fit_metric` and `fit_metric_name` for unsupervised
- Saved as `.pkl` with keys: `classifier, class_names, algorithm, n_channels, n_samples, unsupervised`
- Script export: `GET /api/export/script/{job_id}` now works for both `.pt` (DL) and `.pkl` (classical)

---

## Training Results

`GET /api/train/{job_id}/results` returns:
- `final_val_acc`, `confusion_matrix`, `class_names`, `roc_data`
- `fit_metric`, `fit_metric_name` (unsupervised classical only)
- `roc_data`: `[{class_idx, auc, points:[{fpr,tpr}]}]` — one-vs-rest, max 200 points per curve

---

## Inference Output System

**OutputDispatcher** (singleton in SessionStore):
- WebSocket server — broadcasts predictions to external clients (default port 8765)
- Serial port — sends string per class to Arduino via pyserial

Hardware WS messages include `board_info: {data_rate_hz, board_id, n_channels, sample_rate}`.

---

## Frontend Components

| Component | Purpose |
|---|---|
| `ElectrodeMap.jsx` | SVG 10-20 head diagram; mode="select" (channel picker) or mode="reref" (ref electrode picker). Shown when ≥4 session channels match standard 10-20 names. |
| `PipelineProgress.jsx` | 5-stage horizontal progress bar in Sidebar |
| `WaveformChart.jsx` | Waveform with time-range controls + amplitude zoom (×1/2/5/10) + Brush |
| `ModelLayerEditor.jsx` | Visual layer card editor with validate-architecture button |

---

## Code Standards

**Python:** Type hints everywhere, docstrings on classes and public functions, `logging` (never `print()`), `@dataclass` for data structures.

**React:** Functional components only, `async/await`, all API calls use `API_BASE_URL` from `frontend/src/config.js`.

**General:** Every endpoint returns `{ success, data, error }`. Human-readable errors.

---

## Running the Project

```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
# Frontend: http://localhost:5173 | API: http://127.0.0.1:8000 | Docs: http://127.0.0.1:8000/docs
```
