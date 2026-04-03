"""
analysis_routes.py — HTTP endpoints for read-only signal analysis.

Analysis is separate from preprocessing:
  - Preprocessing TRANSFORMS the signal (filters, windows, normalizes)
  - Analysis INSPECTS the signal (FFT, PSD, band power, spectrogram)

All endpoints in this file are POST because they take a session_id in the body.
They never modify any stored state — they compute and return results only.

Registered in main.py with prefix /api/analysis.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.analysis import (
    compute_band_power,
    compute_erp,
    compute_fft,
    compute_psd,
    compute_signal_stats,
    compute_spectrogram,
)
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter()
store = SessionStore()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def api_response(success: bool, data=None, error: str | None = None) -> dict:
    """Standard APIResponse shape from CLAUDE.md."""
    return {"success": success, "data": data, "error": error}


def _get_signals(session_id: str, run_id: str | None) -> tuple[np.ndarray, int, list[str]] | None:
    """
    Return (signals, sample_rate, channel_names) for a session.

    If run_id is provided, concatenates all preprocessed windows from that
    pipeline run back into a single signal array for analysis.
    If run_id is None, returns the raw session signals.

    Returns None if the session or run cannot be found.
    """
    session = store.sessions.get(session_id)
    if session is None:
        return None

    if run_id:
        run = store.pipeline_runs.get(run_id)
        if run is None or not isinstance(run, dict):
            return None
        windows = run["windows"]  # list[np.ndarray], each shape (n_channels, n_samples)
        if not windows:
            # Fall back to raw signals if no windows
            return session.signals, session.sample_rate, session.channel_names
        # Concatenate windows along the time axis to reconstruct a continuous signal.
        # This lets us analyze the "after preprocessing" state in one view.
        signals = np.concatenate(windows, axis=1)
        n_channels = signals.shape[0]
        # Channel names may have changed if ChannelSelector was applied.
        # The window shape tells us how many channels remain.
        if n_channels == len(session.channel_names):
            channel_names = session.channel_names
        else:
            channel_names = [f"ch_{i}" for i in range(n_channels)]
        return signals, session.sample_rate, channel_names

    return session.signals, session.sample_rate, session.channel_names


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Body for analysis endpoints that don't need a specific channel."""
    session_id: str
    run_id: str | None = None   # if set, analyze preprocessed windows instead of raw signal


class SpectrogramRequest(BaseModel):
    """Body for POST /api/analysis/spectrogram (needs a specific channel)."""
    session_id:    str
    run_id:        str | None = None
    channel_index: int = 0     # which channel to compute the spectrogram for


class PSDRequest(BaseModel):
    """Body for POST /api/analysis/psd (supports custom nperseg)."""
    session_id: str
    run_id:     str | None = None
    nperseg:    int = 256      # Welch segment length in samples


class ERPRequest(BaseModel):
    """Body for POST /api/analysis/erp — needs a run_id to find labeled windows."""
    session_id: str
    run_id:     str


# ---------------------------------------------------------------------------
# POST /api/analysis/fft
# ---------------------------------------------------------------------------

@router.post("/fft")
async def fft_endpoint(request: AnalysisRequest) -> JSONResponse:
    """
    Compute the FFT magnitude spectrum for all channels.

    Returns per-channel magnitude arrays and the shared frequency axis.
    The frontend uses this to draw line charts showing which frequencies are
    dominant in the EEG signal.
    """
    result = _get_signals(request.session_id, request.run_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    signals, sample_rate, channel_names = result

    try:
        data = compute_fft(signals, sample_rate, channel_names)
    except Exception as exc:
        logger.error("FFT failed for session %s: %s", request.session_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"FFT computation failed: {exc}"),
        )

    return JSONResponse(status_code=200, content=api_response(True, data=data))


# ---------------------------------------------------------------------------
# POST /api/analysis/psd
# ---------------------------------------------------------------------------

@router.post("/psd")
async def psd_endpoint(request: PSDRequest) -> JSONResponse:
    """
    Compute the Welch Power Spectral Density for all channels.

    Welch's method is preferred over raw FFT for EEG because it averages multiple
    overlapping segments, reducing noise in the power estimate.
    """
    result = _get_signals(request.session_id, request.run_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    signals, sample_rate, channel_names = result

    try:
        data = compute_psd(signals, sample_rate, request.nperseg, channel_names)
    except Exception as exc:
        logger.error("PSD failed for session %s: %s", request.session_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"PSD computation failed: {exc}"),
        )

    return JSONResponse(status_code=200, content=api_response(True, data=data))


# ---------------------------------------------------------------------------
# POST /api/analysis/bandpower
# ---------------------------------------------------------------------------

@router.post("/bandpower")
async def bandpower_endpoint(request: AnalysisRequest) -> JSONResponse:
    """
    Compute power in delta/theta/alpha/beta/gamma bands for all channels.

    Returns a single scalar per band per channel — perfect for a grouped bar chart.
    This is the most common way to compare brain states across sessions and channels.
    """
    result = _get_signals(request.session_id, request.run_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    signals, sample_rate, channel_names = result

    try:
        data = compute_band_power(signals, sample_rate, channel_names)
    except Exception as exc:
        logger.error("Band power failed for session %s: %s", request.session_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Band power computation failed: {exc}"),
        )

    return JSONResponse(status_code=200, content=api_response(True, data=data))


# ---------------------------------------------------------------------------
# POST /api/analysis/spectrogram
# ---------------------------------------------------------------------------

@router.post("/spectrogram")
async def spectrogram_endpoint(request: SpectrogramRequest) -> JSONResponse:
    """
    Compute the STFT spectrogram for one channel.

    Returns a 2D power matrix (frequencies × time), power in dB.
    The frontend renders this as a color heatmap where bright = high power.

    Only one channel at a time is computed because a full multi-channel spectrogram
    would produce too much data to send over the API.
    """
    result = _get_signals(request.session_id, request.run_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    signals, sample_rate, channel_names = result

    try:
        data = compute_spectrogram(signals, sample_rate, request.channel_index)
        # Add the channel name to the response so the frontend can label the chart
        n_channels = signals.shape[0]
        if 0 <= request.channel_index < len(channel_names):
            data["channel_name"] = channel_names[request.channel_index]
        else:
            data["channel_name"] = f"ch_{request.channel_index}"
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=api_response(False, error=str(exc)),
        )
    except Exception as exc:
        logger.error("Spectrogram failed for session %s: %s", request.session_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Spectrogram computation failed: {exc}"),
        )

    return JSONResponse(status_code=200, content=api_response(True, data=data))


# ---------------------------------------------------------------------------
# POST /api/analysis/erp
# ---------------------------------------------------------------------------

@router.post("/erp")
async def erp_endpoint(request: ERPRequest) -> JSONResponse:
    """
    Compute Event-Related Potentials by averaging labeled windows per class.

    Returns per-class, per-channel average waveforms and a millisecond time axis.
    The frontend draws overlaid lines for each class label so users can visually
    compare ERP components (P300, N200, alpha ERD) across conditions.

    Requires that the pipeline run has associated labeled windows in the store
    (i.e. the Label page has been used after running the pipeline).
    """
    session = store.sessions.get(request.session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    labeled_windows = store.labeled_data.get(request.run_id, [])
    if not labeled_windows:
        return JSONResponse(
            status_code=404,
            content=api_response(
                False,
                error=(
                    f"No labeled windows found for run '{request.run_id}'. "
                    "Go to the Label page and label some windows first."
                ),
            ),
        )

    try:
        # Use channel names that match the run's windows (post-ChannelSelector)
        run = store.pipeline_runs.get(request.run_id)
        if run and isinstance(run, dict) and run.get("windows"):
            n_channels = run["windows"][0].shape[0]
            ch_names = (
                session.channel_names[:n_channels]
                if len(session.channel_names) >= n_channels
                else [f"ch_{i}" for i in range(n_channels)]
            )
        else:
            ch_names = session.channel_names

        data = compute_erp(labeled_windows, session.sample_rate, ch_names)
    except Exception as exc:
        logger.error("ERP failed for run %s: %s", request.run_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"ERP computation failed: {exc}"),
        )

    return JSONResponse(status_code=200, content=api_response(True, data=data))


# ---------------------------------------------------------------------------
# POST /api/analysis/stats
# ---------------------------------------------------------------------------

@router.post("/stats")
async def stats_endpoint(request: AnalysisRequest) -> JSONResponse:
    """
    Compute per-channel signal quality metrics.

    Called by the Analyze button on PreprocessPage (Time tab).
    Returns RMS, peak-to-peak, std, kurtosis, max_gradient, zero_crossing_rate
    per channel, plus an artifact_warning flag for channels that look suspicious.
    """
    result = _get_signals(request.session_id, request.run_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content=api_response(False, error=f"Session '{request.session_id}' not found."),
        )

    signals, sample_rate, channel_names = result

    try:
        data = compute_signal_stats(signals, sample_rate, channel_names)
    except Exception as exc:
        logger.error("Stats failed for session %s: %s", request.session_id, exc)
        return JSONResponse(
            status_code=500,
            content=api_response(False, error=f"Stats computation failed: {exc}"),
        )

    return JSONResponse(status_code=200, content=api_response(True, data=data))
