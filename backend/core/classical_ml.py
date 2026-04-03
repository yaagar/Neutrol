"""
classical_ml.py — Classical ML training for EEG classification.

Provides lightweight, fast alternatives to deep learning:
  LDA   — Linear Discriminant Analysis. Best for P300, SSVEP, simple paradigms.
          No hyperparameter tuning needed; often outperforms DL on small datasets.
  SVM   — Support Vector Machine. Strong for non-linear decision boundaries.
          Kernel trick separates classes in high-dimensional feature space.
  RF    — Random Forest. Ensemble of decision trees; robust to noise.
          Good when EEG features span many frequency bands.
  GBM   — Gradient Boosting Machine. Sequential ensemble; often highest accuracy.
          Slower to train but frequently beats other classical methods on EEG.

Why classical ML alongside deep learning?
  Deep learning requires many labeled examples (100+ per class is a minimum).
  Classical ML works well with as few as 20–30 examples — critical for clinical
  BCI users who can't sustain long recording sessions. LDA in particular is the
  standard in online P300 spellers and SSVEP systems.

Feature extraction:
  Raw flattened windows (n_channels × n_samples) are used as features.
  This is the same approach as PyBCI-dev and standard in EEG ML literature.
  For small datasets this works well; for large datasets band-power features
  (extracted from PSD) would be more efficient, but the raw approach lets users
  start immediately without choosing a feature extractor.

Training uses stratified k-fold cross-validation (5 folds, or fewer if limited
data) to report a reliable accuracy estimate. The final model is then trained on
ALL labeled data and saved as a .pkl file.

Called by train_routes.py in a background thread — same pattern as trainer.py.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from sessions.session_store import SessionStore

logger = logging.getLogger(__name__)

# Where .pkl model files are saved (same folder as .pt files for consistency)
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")

# Minimum labeled windows to attempt training
MIN_SAMPLES = 4

# Supported algorithm identifiers
SUPPORTED_ALGORITHMS: dict[str, str] = {
    "LDA":          "Linear Discriminant Analysis",
    "SVM":          "Support Vector Machine",
    "RandomForest": "Random Forest",
    "GBM":          "Gradient Boosting Machine",
    "KMeans":       "K-Means Clustering (unsupervised)",
    "GMM":          "Gaussian Mixture Model (unsupervised)",
}

# Algorithms that do not require class labels
UNSUPERVISED_ALGORITHMS = {"KMeans", "GMM"}


def _build_classifier(algorithm: str, hyperparams: dict):
    """
    Build a scikit-learn classifier from an algorithm name and hyperparameter dict.

    Each algorithm's hyperparameters are independently validated so that a missing
    param falls back to a sensible EEG-specific default rather than crashing.

    Args:
        algorithm:   One of the SUPPORTED_ALGORITHMS keys.
        hyperparams: Dict of hyperparameter overrides.

    Returns:
        Fitted-ready sklearn estimator.
    """
    hp = hyperparams or {}

    if algorithm == "LDA":
        solver = hp.get("solver", "svd")
        # Shrinkage is only valid with lsqr/eigen solvers
        shrinkage = hp.get("shrinkage", None) if solver != "svd" else None
        if isinstance(shrinkage, (int, float)) and float(shrinkage) == 0.0:
            shrinkage = None
        return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    elif algorithm == "SVM":
        kernel  = hp.get("kernel", "rbf")
        C       = float(hp.get("C", 1.0))
        gamma   = hp.get("gamma", "scale")
        # probability=True enables predict_proba for confidence reporting
        return SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    elif algorithm == "RandomForest":
        n_estimators = int(hp.get("n_estimators", 100))
        max_depth    = hp.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth) if int(max_depth) > 0 else None
        min_samples_split = int(hp.get("min_samples_split", 2))
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,       # use all CPU cores
            random_state=42,
        )

    elif algorithm == "GBM":
        n_estimators  = int(hp.get("n_estimators", 100))
        learning_rate = float(hp.get("learning_rate", 0.1))
        max_depth     = int(hp.get("max_depth", 3))
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )

    elif algorithm == "KMeans":
        n_clusters = int(hp.get("n_clusters", 8))
        return KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)

    elif algorithm == "GMM":
        n_components    = int(hp.get("n_components", 4))
        covariance_type = hp.get("covariance_type", "full")
        return GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Supported: {list(SUPPORTED_ALGORITHMS.keys())}"
        )


def _confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int) -> list[list[int]]:
    """Pure-numpy confusion matrix — mirrors trainer.compute_confusion_matrix."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            matrix[true][pred] += 1
    return matrix


def run_classical_training(
    job_id:     str,
    session_id: str,
    run_id:     str | None,
    algorithm:  str,
    hyperparams: dict,
    model_name:  str = "",
) -> None:
    """
    Train a classical ML classifier on labeled EEG windows.

    This is designed to be called in a background thread by train_routes.py,
    exactly like trainer.run_training(). It writes to store.training_jobs[job_id]
    so the same GET /api/train/{job_id}/status endpoint works for both DL and ML.

    Training steps:
      1. Load labeled windows (same source as DL trainer).
      2. Flatten: (N, n_channels, n_samples) → (N, n_channels * n_samples).
      3. Build classifier from algorithm + hyperparams.
      4. Stratified k-fold cross-validation → per-fold accuracy scores.
      5. Train final model on all data.
      6. Compute confusion matrix via cross_val_predict (out-of-fold predictions).
      7. Save model as exports/{job_id}.pkl.
      8. Push "done" update to training_jobs.

    Args:
        job_id:      UUID identifying this job.
        session_id:  Session to pull labeled data for.
        run_id:      Optional pipeline run_id. If set, uses that run's labeled_data.
        algorithm:   One of 'LDA', 'SVM', 'RandomForest', 'GBM'.
        hyperparams: Dict of hyperparameter overrides (empty = use defaults).
        model_name:  Human-readable name stored with the job.
    """
    store = SessionStore()
    is_unsupervised = algorithm in UNSUPERVISED_ALGORITHMS

    def push(update: dict) -> None:
        store.training_jobs[job_id]["updates"].append(update)

    def fail(message: str) -> None:
        logger.error("Classical ML job %s failed: %s", job_id, message)
        store.training_jobs[job_id]["status"] = "failed"
        push({"type": "error", "message": message})

    try:
        # ------------------------------------------------------------------
        # 1. Load windows
        # For supervised algorithms: load labeled windows and extract classes.
        # For unsupervised (KMeans/GMM): load all windows from the pipeline run.
        # ------------------------------------------------------------------
        if is_unsupervised:
            # Load raw windows from the pipeline run — no labels needed
            if run_id and run_id in store.pipeline_runs:
                all_windows = store.pipeline_runs[run_id]["windows"]
            else:
                fail("Unsupervised training requires a pipeline run_id. Run preprocessing first.")
                return

            if len(all_windows) < MIN_SAMPLES:
                fail(f"Need at least {MIN_SAMPLES} windows, but only {len(all_windows)} found.")
                return

            X_3d = np.stack(all_windows).astype(np.float32)
            n_windows, n_channels, n_samples_win = X_3d.shape
            X = X_3d.reshape(n_windows, -1)
            classes = []

            logger.info(
                "Classical ML job %s (unsupervised): %d windows, %d features — algorithm=%s",
                job_id, n_windows, X.shape[1], algorithm,
            )
        else:
            # Supervised: load labeled windows
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
                    f"Need at least {MIN_SAMPLES} labeled windows, "
                    f"but only {len(labeled_windows)} found."
                )
                return

            # ------------------------------------------------------------------
            # 2. Convert to numpy — derive classes from labeled data
            # ------------------------------------------------------------------
            classes = sorted({
                lw.label for lw in labeled_windows
                if lw.label and lw.label != "unknown"
            })
            if len(classes) < 2:
                fail("Need at least 2 distinct class labels to train a classifier.")
                return

            class_to_idx = {c: i for i, c in enumerate(classes)}
            valid_pairs  = [
                (lw.data, class_to_idx[lw.label])
                for lw in labeled_windows
                if lw.label in class_to_idx
            ]

            if len(valid_pairs) < MIN_SAMPLES:
                fail(f"Only {len(valid_pairs)} windows have valid class labels.")
                return

            X_3d = np.stack([p[0] for p in valid_pairs]).astype(np.float32)
            y    = np.array([p[1] for p in valid_pairs], dtype=np.int64)

            n_windows, n_channels, n_samples_win = X_3d.shape
            X = X_3d.reshape(n_windows, -1)

            logger.info(
                "Classical ML job %s: %d samples, %d features (%d ch × %d samp), "
                "%d classes — algorithm=%s",
                job_id, n_windows, X.shape[1], n_channels, n_samples_win,
                len(classes), algorithm,
            )

        # Update job meta before training starts
        store.training_jobs[job_id]["class_names"] = classes
        store.training_jobs[job_id]["n_classes"]   = len(classes)

        # ------------------------------------------------------------------
        # 3. Build model
        # ------------------------------------------------------------------
        clf = _build_classifier(algorithm, hyperparams)

        # ------------------------------------------------------------------
        # 4. Train & evaluate
        # Supervised: stratified k-fold CV + confusion matrix.
        # Unsupervised: fit on all data, report inertia (KMeans) or log-likelihood (GMM).
        # ------------------------------------------------------------------
        if is_unsupervised:
            clf.fit(X)
            cv_scores = []
            confusion  = []
            y_cv_pred  = []

            if algorithm == "KMeans":
                fit_metric = round(float(clf.inertia_), 4)
                fit_metric_name = "inertia"
            else:  # GMM
                fit_metric = round(float(clf.score(X)), 4)
                fit_metric_name = "log_likelihood"

            cv_mean = fit_metric
            logger.info("Job %s (unsupervised) fit: %s=%.4f", job_id, fit_metric_name, fit_metric)
        else:
            min_class_count = min(int(np.sum(y == c)) for c in range(len(classes)))
            n_splits = min(5, min_class_count)

            if n_splits < 2:
                logger.warning("Job %s: not enough data for CV, training on all data.", job_id)
                clf.fit(X, y)
                cv_scores = []
                y_cv_pred = clf.predict(X)
                cv_mean   = float(np.mean(y_cv_pred == y))
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                y_cv_pred = cross_val_predict(clf, X, y, cv=cv)
                cv_scores_raw = []
                for train_idx, test_idx in cv.split(X, y):
                    clf.fit(X[train_idx], y[train_idx])
                    cv_scores_raw.append(float(np.mean(clf.predict(X[test_idx]) == y[test_idx])))
                cv_scores = [round(s, 4) for s in cv_scores_raw]
                cv_mean   = round(float(np.mean(cv_scores_raw)), 4)
                clf.fit(X, y)

            confusion = _confusion_matrix(y.tolist(), y_cv_pred.tolist(), len(classes))
            fit_metric_name = "cv_accuracy"
            logger.info("Job %s CV complete: mean=%.4f, folds=%s", job_id, cv_mean, cv_scores)

        # ------------------------------------------------------------------
        # 5. Save model as .pkl
        # ------------------------------------------------------------------
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        model_path = os.path.join(EXPORTS_DIR, f"{job_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "classifier":  clf,
                "class_names": classes,
                "algorithm":   algorithm,
                "hyperparams": hyperparams,
                "n_features":  X.shape[1],
                "n_channels":  n_channels,
                "n_samples":   n_samples_win,
                "trained_at":  datetime.now(timezone.utc).isoformat(),
                "unsupervised": is_unsupervised,
            }, f)

        logger.info("Classical ML model saved to %s", model_path)

        # ------------------------------------------------------------------
        # 6. Mark job done
        # ------------------------------------------------------------------
        store.training_jobs[job_id]["status"]         = "done"
        store.training_jobs[job_id]["model_path"]     = model_path
        store.training_jobs[job_id]["ml_type"]        = "classical"
        store.training_jobs[job_id]["algorithm"]      = algorithm
        store.training_jobs[job_id]["cv_scores"]      = cv_scores
        store.training_jobs[job_id]["final_val_acc"]  = cv_mean

        push({
            "type":              "done",
            "final_val_acc":     cv_mean,
            "fit_metric_name":   fit_metric_name,
            "cv_scores":         cv_scores,
            "confusion_matrix":  confusion,
            "class_names":       classes,
            "model_path":        model_path,
            "algorithm":         algorithm,
            "ml_type":           "classical",
            "unsupervised":      is_unsupervised,
        })

    except Exception as exc:
        logger.exception("Unexpected error in classical ML job %s", job_id)
        fail(f"Unexpected training error: {exc}")
