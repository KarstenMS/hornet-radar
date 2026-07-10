"""Hornet Radar: background YOLO inference worker.

YOLO inference on the Pi CPU takes ~0.5-1 s and would stall the capture/tracking
loop if run inline. This worker runs the model in a dedicated thread so the fast
loop never blocks: the main thread submits a snapshot (frame + track boxes) and
later polls for the finished result, applying it to the (still-living) tracks.

Design keeps threading simple and lock-light:
    - exactly one job in flight (submit is refused while busy),
    - the model is only ever touched by the worker thread (or by detect_sync in
      batch image/video mode, where the worker is idle),
    - tracks are only mutated by the main thread (the worker returns plain data).
"""

import logging
import queue
import threading
from typing import Any, List, Tuple

from detection import load_model, run_detection

logger = logging.getLogger(__name__)


class YoloWorker:
    """A single background thread that runs YOLO on submitted frames."""

    def __init__(self) -> None:
        self.model = load_model()
        self._requests: "queue.Queue" = queue.Queue(maxsize=1)
        self._results: "queue.Queue" = queue.Queue()
        self._in_flight = threading.Event()          # set from submit until the result is queued
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="yolo-worker", daemon=True)
        self._thread.start()

    # --- Async API (camera mode) ---

    def busy(self) -> bool:
        """Whether a job is currently queued or being processed."""
        return self._in_flight.is_set()

    def submit(self, frame, track_boxes: List[Tuple[int, tuple]]) -> bool:
        """Queue one inference job. Refused (returns False) while busy.

        Args:
            frame: BGR frame to run YOLO on (copied by the caller if needed).
            track_boxes: list of (track_id, bbox) snapshots the caller wants
                results matched against.
        """
        if self._in_flight.is_set():
            return False
        self._in_flight.set()
        self._requests.put((frame, track_boxes))
        return True

    def poll(self) -> List[Tuple[List[Tuple[int, tuple]], Any, list]]:
        """Return all finished results (usually 0 or 1). Non-blocking.

        Each result is (track_boxes, frame, detections) from a prior submit.
        """
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    # --- Sync API (batch image/video mode; worker thread is idle then) ---

    def detect_sync(self, frame) -> list:
        """Run YOLO inline in the calling thread (no threading)."""
        return run_detection(frame, self.model)

    # --- Lifecycle ---

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, track_boxes = self._requests.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                detections = run_detection(frame, self.model)
            except Exception:
                logger.exception("YOLO inference failed")
                detections = []
            self._results.put((track_boxes, frame, detections))
            self._in_flight.clear()

    def stop(self) -> None:
        """Signal the worker to stop and wait briefly for it to exit."""
        self._stop.set()
        self._thread.join(timeout=2.0)
