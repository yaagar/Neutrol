"""
trainer.py — Training loop, validation, and metrics for EEGNet.

The main entry point is run_training(), which is designed to be called in a
background daemon thread. It writes progress updates into store.training_jobs[job_id]
as a list of dicts. The WebSocket handler in train_routes.py reads this list
and forwards updates to the browser in real time.

Why a thread instead of async?
  PyTorch training is synchronous and CPU-bound. Running it with await would
  block the entire FastAPI event loop, making the server unresponsive.
  A daemon thread runs independently and only communicates via the shared
  store dict — no locks needed because Python's GIL protects simple list.append().

New in P2-S3:
  - augment_windows(): optional data augmentation (gaussian noise, time shift,
    channel dropout) expands the training set to improve robustness.
  - run_training() accepts finetune_from_job_id: loads weights from an existing
    checkpoint before the training loop begins instead of random initialisation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from core.models import EEGNet, build_model
from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

# Path where trained model weights are saved
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")

# Minimum number of windows needed to attempt training
MIN_SAMPLES = 4


# ---------------------------------------------------------------------------
# EEGDataset — wraps numpy arrays as a PyTorch Dataset
# ---------------------------------------------------------------------------

class EEGDataset(Dataset):
    """
    Wraps numpy signal windows and integer class labels as a PyTorch Dataset.

    PyTorch's DataLoader requires a Dataset object to iterate in mini-batches.
    This is the standard thin wrapper for in-memory numpy data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: Shape (N, n_channels, n_samples) — the signal windows.
            y: Shape (N,) — integer class indices.
        """
        # unsqueeze(1) adds the channel dimension PyTorch CNN expects:
        # (N, n_channels, n_samples) → (N, 1, n_channels, n_samples)
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# compute_confusion_matrix — pure numpy, no sklearn required
# ---------------------------------------------------------------------------

def compute_confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int) -> list[list[int]]:
    """
    Build an n_classes × n_classes confusion matrix without sklearn.

    Row i = true class i, column j = predicted class j.
    Cell [i][j] = number of times a sample of class i was predicted as class j.
    The diagonal is correct predictions.
    """
    # Initialise all zeros
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            matrix[true][pred] += 1
    return matrix


# ---------------------------------------------------------------------------
# augment_windows — expand training data with synthetic perturbations
# ---------------------------------------------------------------------------

def augment_windows(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the training set by applying augmentations to each window.

    Each original window produces (multiplier - 1) augmented copies, so the
    total set size becomes multiplier × original size. Augmentations are only
    applied to copies — the original window is always preserved unchanged.

    Args:
        X:      Shape (N, n_channels, n_samples) — original windows.
        y:      Shape (N,) — class indices.
        config: Augmentation config dict from the training request:
                {
                  "gaussian_noise":  {"enabled": bool, "strength": float},
                  "time_shift":      {"enabled": bool, "max_shift_ratio": float},
                  "channel_dropout": {"enabled": bool},
                  "multiplier":      int  (how many total copies per window)
                }

    Returns:
        Tuple (X_aug, y_aug) with shape (N * multiplier, n_channels, n_samples).
    """
    multiplier = int(config.get("multiplier", 3))
    if multiplier <= 1:
        # No expansion requested — return originals unchanged
        return X, y

    noise_cfg   = config.get("gaussian_noise",  {})
    shift_cfg   = config.get("time_shift",       {})
    dropout_cfg = config.get("channel_dropout",  {})

    noise_enabled   = noise_cfg.get("enabled",   False)
    shift_enabled   = shift_cfg.get("enabled",   False)
    dropout_enabled = dropout_cfg.get("enabled", False)

    n_samples_time = X.shape[2]
    rng = np.random.default_rng(seed=None)  # unseeded so each training run is different

    all_X = [X]   # start with the original windows
    all_y = [y]

    for _ in range(multiplier - 1):
        # Work on a copy so augmentations don't stack across copies
        aug = X.copy().astype(np.float32)

        if noise_enabled:
            # Add Gaussian noise proportional to each window's own std.
            # strength=0.05 means noise amplitude = 5% of the signal std.
            strength = float(noise_cfg.get("strength", 0.05))
            for i in range(len(aug)):
                std = aug[i].std()
                aug[i] += rng.normal(0, std * strength, aug[i].shape).astype(np.float32)

        if shift_enabled:
            # Roll each window by a random amount along the time axis.
            # roll() wraps samples from the end to the beginning, which is
            # a reasonable approximation of a temporal shift for cyclic-ish signals.
            max_ratio = float(shift_cfg.get("max_shift_ratio", 0.1))
            max_shift = max(1, int(n_samples_time * max_ratio))
            for i in range(len(aug)):
                shift = int(rng.integers(-max_shift, max_shift + 1))
                if shift != 0:
                    aug[i] = np.roll(aug[i], shift, axis=1)

        if dropout_enabled:
            # Zero out one randomly chosen channel per window.
            # This forces the model not to rely on any single electrode.
            n_channels = aug.shape[1]
            for i in range(len(aug)):
                ch = int(rng.integers(0, n_channels))
                aug[i, ch, :] = 0.0

        all_X.append(aug)
        all_y.append(y.copy())

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ---------------------------------------------------------------------------
# run_training — the main training function, designed for a background thread
# ---------------------------------------------------------------------------

def run_training(
    job_id:                str,
    session_id:            str,
    model_config:          dict,
    run_id:                str | None = None,
    finetune_from_job_id:  str | None = None,
    augmentation_config:   dict | None = None,
) -> None:
    """
    Run the full training pipeline for one job.

    This function is called in a background thread by train_routes.py.
    It appends progress dicts to store.training_jobs[job_id]["updates"] after
    each epoch so the WebSocket handler can forward them to the browser.

    Progress message types:
      {"type": "progress", "epoch": int, "total_epochs": int,
       "train_loss": float, "val_loss": float, "train_acc": float, "val_acc": float}
      {"type": "done", "final_val_acc": float, "confusion_matrix": list, "class_names": list, "model_path": str}
      {"type": "error", "message": str}

    Args:
        job_id:               UUID identifying this training job.
        session_id:           UUID of the session whose labeled_data to train on.
        model_config:         ModelConfig dict from CLAUDE.md.
        run_id:               Optional pipeline run_id. labeled_data is keyed by run_id.
                              If provided, look up directly. Otherwise aggregate all
                              labeled windows for the session (fallback).
        finetune_from_job_id: If set, load weights from this existing checkpoint.
                              Classes must match exactly; raises if they don't.
        augmentation_config:  Optional dict controlling data augmentation (see augment_windows).
    """
    store = SessionStore()

    def push(update: dict) -> None:
        """Append an update to the job's update list."""
        store.training_jobs[job_id]["updates"].append(update)

    def fail(message: str) -> None:
        """Mark the job as failed and push an error message."""
        logger.error("Training job %s failed: %s", job_id, message)
        store.training_jobs[job_id]["status"] = "failed"
        push({"type": "error", "message": message})

    try:
        # ------------------------------------------------------------------
        # 1. Load labeled windows from the store.
        # labeled_data is keyed by run_id (not session_id). If run_id is provided,
        # look it up directly. Otherwise fall back to aggregating all windows
        # for this session across all runs.
        # ------------------------------------------------------------------
        if run_id:
            labeled_windows = store.labeled_data.get(run_id, [])
        else:
            labeled_windows = [
                lw
                for lw_list in store.labeled_data.values()
                for lw in lw_list
                if lw.session_id == session_id
            ]

        if len(labeled_windows) < MIN_SAMPLES:
            fail(
                f"Need at least {MIN_SAMPLES} labeled windows to train, "
                f"but only {len(labeled_windows)} exist. "
                "Go to the Label page and label more windows."
            )
            return

        # ------------------------------------------------------------------
        # 2. Convert labels to integer indices
        # Only keep windows whose label is in model_config["classes"].
        # ------------------------------------------------------------------
        classes: list[str] = model_config["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        valid_pairs = [
            (lw.data, class_to_idx[lw.label])
            for lw in labeled_windows
            if lw.label in class_to_idx
        ]

        if len(valid_pairs) < MIN_SAMPLES:
            fail(
                f"Only {len(valid_pairs)} windows have labels matching the configured classes "
                f"{classes}. Check that the labeled windows match the class list."
            )
            return

        # Stack into numpy arrays: X shape (N, n_channels, n_samples)
        X = np.stack([pair[0] for pair in valid_pairs]).astype(np.float32)
        y = np.array([pair[1] for pair in valid_pairs], dtype=np.int64)

        n_channels, n_samples = X.shape[1], X.shape[2]
        n_classes = len(classes)

        logger.info(
            "Training job %s: %d samples, %d channels, %d time steps, %d classes",
            job_id, len(X), n_channels, n_samples, n_classes,
        )

        # ------------------------------------------------------------------
        # 2b. Data augmentation (optional)
        # Apply before train/val split so augmented copies stay in the training
        # set. The split is random, so a tiny fraction may end up in validation,
        # but that is acceptable and common practice.
        # ------------------------------------------------------------------
        if augmentation_config:
            any_enabled = any([
                augmentation_config.get("gaussian_noise",  {}).get("enabled", False),
                augmentation_config.get("time_shift",      {}).get("enabled", False),
                augmentation_config.get("channel_dropout", {}).get("enabled", False),
            ])
            if any_enabled and augmentation_config.get("multiplier", 1) > 1:
                X, y = augment_windows(X, y, augmentation_config)
                logger.info(
                    "Job %s: augmented to %d samples (multiplier=%d)",
                    job_id, len(X), augmentation_config.get("multiplier", 3),
                )

        # ------------------------------------------------------------------
        # 3. Build Dataset and train/val split (80/20)
        # ------------------------------------------------------------------
        dataset = EEGDataset(X, y)
        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size

        # random_split uses a fixed generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        batch_size = max(1, min(model_config.get("batch_size", 32), train_size))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # ------------------------------------------------------------------
        # 4. Build model, optimizer, loss function
        # ------------------------------------------------------------------

        # Always use CPU — we don't assume the user has a GPU.
        # For small EEG datasets this is fine (training is fast on CPU).
        device = torch.device("cpu")

        # Use the dynamic builder when a 'layers' config is provided.
        # Fall back to the hardcoded EEGNet for configs without 'layers'
        # (backward compatibility with old saved jobs).
        if "layers" in model_config and model_config["layers"]:
            model = build_model(model_config).to(device)
            # Materialize lazy modules with a dummy forward pass before training
            dummy = torch.zeros(1, 1, n_channels, n_samples).to(device)
            with torch.no_grad():
                model.eval()
                model(dummy)
            model.train()
            logger.info("Job %s: using dynamic model (%d layers)", job_id, len(model_config["layers"]))
        else:
            model = EEGNet(n_channels, n_samples, n_classes).to(device)
            logger.info("Job %s: using default EEGNet", job_id)

        # ------------------------------------------------------------------
        # 4b. Fine-tuning: load weights from an existing checkpoint.
        # This overwrites the random initialisation with trained weights so
        # the model starts from a better starting point.
        # ------------------------------------------------------------------
        if finetune_from_job_id:
            base_path = os.path.join(EXPORTS_DIR, f"{finetune_from_job_id}.pt")
            if not os.path.exists(base_path):
                fail(f"Base model file not found: {base_path}")
                return

            base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
            base_classes = base_ckpt.get("class_names", [])

            # Classes must match exactly — the final linear layer's output size
            # is fixed to n_classes, so swapping or adding classes would break
            # the weight shapes.
            if sorted(base_classes) != sorted(classes):
                fail(
                    f"Class mismatch: base model has {base_classes} "
                    f"but current training data has {classes}. "
                    "Fine-tuning requires identical class lists."
                )
                return

            model.load_state_dict(base_ckpt["model_state_dict"])
            model.train()
            logger.info(
                "Job %s: fine-tuning from %s (classes: %s)",
                job_id, finetune_from_job_id, classes,
            )

        # Adam is the standard choice for EEG models — handles sparse gradients
        # and adapts learning rate per-parameter.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config.get("learning_rate", 0.001),
        )

        # CrossEntropyLoss combines log-softmax + negative log-likelihood.
        # It expects raw logits from the model (no softmax in forward()).
        criterion = nn.CrossEntropyLoss()

        epochs = model_config.get("epochs", 50)

        # Update the job with the config it's running with
        store.training_jobs[job_id]["n_classes"] = n_classes
        store.training_jobs[job_id]["class_names"] = classes

        # ------------------------------------------------------------------
        # 5. Training loop
        # ------------------------------------------------------------------
        history = []

        for epoch in range(epochs):
            # ---- Training phase ----
            model.train()
            train_loss_sum = 0.0
            train_correct  = 0
            train_total    = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()           # clear gradients from previous step
                logits = model(X_batch)         # forward pass
                loss   = criterion(logits, y_batch)  # compute loss
                loss.backward()                 # backpropagation
                optimizer.step()                # update weights

                train_loss_sum += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total   += len(y_batch)

            train_loss = train_loss_sum / train_total
            train_acc  = train_correct  / train_total

            # ---- Validation phase ----
            # torch.no_grad() disables gradient tracking — saves memory and speeds
            # up inference since we don't need gradients for evaluation.
            model.eval()
            val_loss_sum = 0.0
            val_correct  = 0
            val_total    = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss   = criterion(logits, y_batch)

                    val_loss_sum += loss.item() * len(y_batch)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total   += len(y_batch)

            val_loss = val_loss_sum / val_total
            val_acc  = val_correct  / val_total

            epoch_result = {
                "epoch":       epoch + 1,
                "train_loss":  round(train_loss, 4),
                "val_loss":    round(val_loss,   4),
                "train_acc":   round(train_acc,  4),
                "val_acc":     round(val_acc,    4),
            }
            history.append(epoch_result)

            # Push a progress update — the WebSocket handler will forward this to React
            push({
                "type":         "progress",
                "total_epochs": epochs,
                **epoch_result,
            })

            # Update current epoch in the job so GET /status can read it
            store.training_jobs[job_id]["current_epoch"] = epoch + 1
            store.training_jobs[job_id]["history"]       = history

            logger.info(
                "Job %s epoch %d/%d — train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
                job_id, epoch + 1, epochs, train_loss, val_loss, train_acc, val_acc,
            )

        # ------------------------------------------------------------------
        # 6. Final evaluation — confusion matrix on validation set
        # ------------------------------------------------------------------
        model.eval()
        all_true = []
        all_pred = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                preds  = logits.argmax(dim=1).cpu().numpy()
                all_pred.extend(preds.tolist())
                all_true.extend(y_batch.numpy().tolist())

        confusion = compute_confusion_matrix(all_true, all_pred, n_classes)

        # ------------------------------------------------------------------
        # 7. Save model weights to exports/
        # ------------------------------------------------------------------
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        model_path = os.path.join(EXPORTS_DIR, f"{job_id}.pt")

        # Fetch sample_rate and pipeline_config so the inference script can embed preprocessing.
        # sample_rate comes from the session; pipeline_config from the pipeline run.
        # Both may be None if the backend was restarted mid-session, which is acceptable —
        # the inference script falls back to expecting a pre-processed window in that case.
        session          = store.sessions.get(session_id)
        sample_rate      = session.sample_rate if session else 256
        pipeline_config  = None
        if run_id:
            run_entry = store.pipeline_runs.get(run_id, {})
            if isinstance(run_entry, dict):
                pipeline_config = run_entry.get("pipeline_config")

        # Save both weights and the config so we can reconstruct the model later.
        # finetuned_from is stored for history traceability — None means scratch.
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config":     model_config,
            "n_channels":       n_channels,
            "n_samples":        n_samples,
            "n_classes":        n_classes,
            "class_names":      classes,
            "trained_at":       datetime.now(timezone.utc).isoformat(),
            "sample_rate":      sample_rate,
            "pipeline_config":  pipeline_config,
            "finetuned_from":   finetune_from_job_id,
        }, model_path)

        logger.info("Model saved to %s", model_path)

        # ------------------------------------------------------------------
        # 8. Mark job as done and push final result
        # ------------------------------------------------------------------
        final_val_acc   = history[-1]["val_acc"]   if history else 0.0
        final_train_acc = history[-1]["train_acc"] if history else 0.0

        store.training_jobs[job_id]["status"]         = "done"
        store.training_jobs[job_id]["model_path"]     = model_path
        store.training_jobs[job_id]["finetuned_from"] = finetune_from_job_id

        push({
            "type":             "done",
            "final_train_acc":  final_train_acc,
            "final_val_acc":    final_val_acc,
            "confusion_matrix": confusion,
            "class_names":      classes,
            "model_path":       model_path,
            "finetuned_from":   finetune_from_job_id,
        })

    except Exception as exc:
        logger.exception("Unexpected error in training job %s", job_id)
        fail(f"Unexpected training error: {exc}")
