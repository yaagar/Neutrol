# OpenEEG

An open-source, beginner-friendly EEG development platform.

EEG (electroencephalography) measures electrical brain activity through electrodes on the scalp. The raw output is a set of voltage signals over time — one per electrode (channel). To do anything useful with it, you need to clean the signal, segment it into windows, label those windows, train a model, and then run that model in real time.

**OpenEEG does all of that in one place.** The design goal is the same as the Arduino IDE: simple enough for a student on day one, fully editable for an expert.

**User workflow:**
```
Import EEG file → Preprocess signal → Label segments → Train CNN → Live predictions → Export
```

---

## How to Run

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API runs at **http://localhost:8000**
Interactive API docs (Swagger UI): **http://localhost:8000/docs**

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at **http://localhost:5173**

---

## What Session 1 Built

**Project scaffold** — the complete directory structure, all config files, and all empty placeholder files that future sessions will fill in.

**Data import backend** (`backend/core/data_loader.py`, `backend/api/data_routes.py`):
- Upload `.csv` or `.edf` EEG files via `POST /api/data/upload`
- Sessions are stored in memory (no database) and identified by a UUID
- List, get, and delete sessions via `GET /api/data/sessions`, `GET /api/data/{id}`, `DELETE /api/data/{id}`

**Preprocessing pipeline backend** (`backend/core/preprocessing.py`, `backend/api/pipeline_routes.py`):
- Four preprocessing steps: `BandpassFilter`, `NotchFilter`, `Normalize`, `SlidingWindow`
- Steps serialize to/from JSON so pipelines can be saved and reloaded exactly
- Run a pipeline on a session via `POST /api/pipeline/run`
- List available steps, save configs, list saved configs

**Sample data** (`datasets/sample_eeg.csv`):
- 8 channels (Fp1, Fp2, C3, C4, O1, O2, T3, T4)
- 10 seconds at 256 Hz (2560 rows)
- Each channel is a sine wave at a unique frequency (6–25 Hz) with small random noise

---

## Directory Structure

```
├── backend/
│   ├── main.py               FastAPI app, CORS, all routers
│   ├── requirements.txt      Python dependencies
│   ├── api/                  Route handlers (HTTP layer only)
│   ├── core/                 All signal processing logic (no HTTP)
│   ├── sessions/             In-memory singleton data store
│   └── plugins/              Empty — future custom steps/models
├── frontend/
│   └── src/
│       ├── App.jsx           Two-column layout (sidebar + content)
│       ├── pages/            One page per workflow step
│       ├── components/       Reusable UI components
│       ├── hooks/            useSession, useWebSocket
│       └── store/            Zustand global state
├── configs/                  Saved pipeline JSON files
├── datasets/                 Sample EEG data
└── exports/                  Exported models and scripts
```

---

## Session Roadmap

| Session | What gets built |
|---|---|
| **1** ✅ | Project scaffold + data import + preprocessing backend |
| **2** | Import page UI + session manager + waveform display |
| **3** | Labeling page UI + label save/load API |
| **4** | CNN model + training endpoint + training WebSocket |
| **5** | Training results page (loss curves, confusion matrix) |
| **6** | Inference engine + real-time prediction WebSocket + page |
| **7** | Export system (pipeline JSON, model weights, inference script) |
| **8** | Pipeline builder UI + session polish |
