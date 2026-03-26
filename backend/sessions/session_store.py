"""
session_store.py — In-memory singleton store for all session data.

A "singleton" means Python will only ever create ONE instance of this class,
no matter how many times you call SessionStore() in different files.
This is how all route handlers share the same data without a database:
they all call SessionStore() and get back the same object.

Why not just use a global dict?
  A class gives us a clear namespace (store.sessions, store.pipeline_runs),
  makes it easy to add helper methods later, and is easier to mock in tests.
"""

from __future__ import annotations


class SessionStore:
    """Singleton in-memory store. Import and call SessionStore() anywhere to get the same instance."""

    # _instance holds the single shared object after first creation.
    # Class-level attribute (not instance-level) so it persists across calls.
    _instance: "SessionStore | None" = None

    def __new__(cls) -> "SessionStore":
        # __new__ is called before __init__. We override it to implement the singleton:
        # if _instance already exists, return it instead of creating a new object.
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # Initialize all storage dicts exactly once.
            # These are the four data stores the entire backend uses:

            # Maps session_id (str) → DataSession object
            # Populated by data_routes.py on file upload.
            cls._instance.sessions: dict = {}

            # Maps run_id (str) → list[np.ndarray] (preprocessed windows)
            # Populated by pipeline_routes.py after running a pipeline.
            cls._instance.pipeline_runs: dict = {}

            # Maps session_id (str) → list[LabeledWindow]
            # Populated by label_routes.py when the user saves labels.
            cls._instance.labeled_data: dict = {}

            # Maps job_id (str) → dict with training state
            # (epoch, loss, accuracy, status, model path when done)
            # Populated by train_routes.py when training starts.
            cls._instance.training_jobs: dict = {}

        return cls._instance
