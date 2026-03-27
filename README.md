# Neutrol — EEG Development Platform

An open-source, beginner-friendly platform for EEG signal processing, model training, and inference. Simple enough for a student on day one, fully editable for an expert.

**Full workflow:**
```
Import EEG → Preprocess → Label → Train → Inference → Export
```

---

## Requirements

- Python 3.11+
- Node.js 18+

---

## Quick Start

**Terminal 1 — Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API docs at `http://localhost:8000/docs`.

---

## Workflow

### 1 — Import
Upload a `.csv` or `.edf` EEG recording. Multiple sessions can be loaded and switched between.

- Drag and drop or click to browse
- Rename sessions with the pencil icon (✎)
- Preview the raw waveform with per-channel toggles

### 2 — Preprocess
Build a preprocessing pipeline by chaining steps. Steps run in order on the raw signal.

| Step | What it does |
|---|---|
| Truncate Signal | Crop to a time range |
| Bandpass Filter | Keep only frequencies in a band (e.g. 1–40 Hz) |
| Notch Filter | Remove power line noise (e.g. 60 Hz) |
| Channel Selector | Drop channels you don't need |
| Re-reference | Subtract average or linked mastoid reference |
| Normalize | Z-score or min-max per channel |
| Sliding Window | Chop into fixed-length overlapping windows |
| Epoch Rejection | Discard windows with artifact spikes |

**Tips:**
- Use ↑ ↓ to reorder steps without deleting them
- Use ⧉ to duplicate a step and tweak parameters
- The Before/After toggle shows exactly what each filter does to the signal

### 3 — Label
Assign class labels to windows. Two methods:

- **Manual** — navigate window by window, click a label
- **Auto from event file** — upload `.edf.event`, `.tsv`, `.eve`, or `.vmrk` and labels apply automatically from event timestamps

### 4 — Train
Design a neural network architecture with the layer editor, then start training. Loss and accuracy stream live via WebSocket. View the confusion matrix when done.

Default architecture: EEGNet-inspired (Conv2d → DepthwiseConv2d → AvgPool → Dropout → Flatten → Linear)

### 5 — Inference
Select a trained model and a pipeline run, then stream live predictions at a configurable speed (100 ms – 2 s per window).

### 6 — Export
Download everything needed to use your model outside Neutrol:

| File | What it is |
|---|---|
| `.pt` | PyTorch model weights |
| `.py` | Standalone inference script (only needs `torch` + `numpy`) |
| `.json` | Pipeline configuration |

**Using the inference script:**
```bash
python neutrol_inference_<id>.py   # self-test
```
```python
from neutrol_inference_<id> import predict
import numpy as np

window = np.zeros((8, 256), dtype=np.float32)  # (n_channels, n_samples)
result = predict(window)
# {"class": "T1", "confidence": 0.87, "probabilities": {...}}
```

Keep the `.py` and `.pt` files in the same folder. No Neutrol installation needed.

---

## Directory Structure

```
Neutrol/
├── backend/
│   ├── main.py           FastAPI app, CORS, all routers
│   ├── api/              HTTP route handlers (no logic)
│   └── core/             Signal processing, training, inference
├── frontend/
│   └── src/
│       ├── pages/        One page per workflow step
│       ├── components/   Sidebar, charts, layer editor
│       └── store/        Zustand global state
├── exports/              Saved model weights (.pt)
├── configs/              Saved pipeline configs (.json)
└── datasets/             Sample EEG data
```

---

## Sample Data

`datasets/sample_eeg.csv` — synthetic 8-channel EEG at 256 Hz, 10 seconds.

PhysioNet EDF files (e.g. `S001R03.edf`) with embedded motor imagery annotations work out of the box — upload the `.edf` and use auto-labeling on the Label page.
