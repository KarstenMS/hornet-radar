"""Hornet Radar: geometry + greedy data association for the multi-tracker.

Boxes are (x, y, w, h) tuples in full-resolution pixel coordinates.

Association strategy (cheap and robust for small, fast insects on a fixed
top-down camera):
    1. Prefer boxes that overlap a track (IoU >= iou_threshold).
    2. Fall back to proximity (center distance <= max_distance) for fast movers
       whose consecutive boxes no longer overlap.
Matches are resolved greedily (best candidate first); each track and each box
is used at most once. The optimal (Hungarian) assignment is deliberately
skipped -- greedy is sufficient at the handful-of-insects scale we expect.
"""
import math
from typing import List, Sequence, Tuple

Box = Tuple[float, float, float, float]


def center(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def center_distance(a: Box, b: Box) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match(
    track_boxes: Sequence[Box],
    detection_boxes: Sequence[Box],
    *,
    iou_threshold: float,
    max_distance: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedily associate detection boxes to track boxes.

    Args:
        track_boxes: last known box per existing track (index = track index).
        detection_boxes: candidate boxes this frame (index = detection index).
        iou_threshold: minimum overlap to accept an overlap match.
        max_distance: maximum center distance (px) to accept a proximity match.

    Returns:
        (matches, unmatched_tracks, unmatched_detections) where matches is a
        list of (track_index, detection_index) pairs.
    """
    candidates = []  # (priority, score_key, track_idx, det_idx)
    for ti, tb in enumerate(track_boxes):
        for di, db in enumerate(detection_boxes):
            overlap = iou(tb, db)
            if overlap >= iou_threshold:
                # Priority 1: overlap match, larger IoU is better.
                candidates.append((1, overlap, ti, di))
                continue
            dist = center_distance(tb, db)
            if dist <= max_distance:
                # Priority 0: proximity match, smaller distance is better.
                candidates.append((0, -dist, ti, di))

    # Best candidates first: higher priority, then better score.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    used_tracks: set = set()
    used_dets: set = set()
    matches: List[Tuple[int, int]] = []
    for _prio, _score, ti, di in candidates:
        if ti in used_tracks or di in used_dets:
            continue
        used_tracks.add(ti)
        used_dets.add(di)
        matches.append((ti, di))

    unmatched_tracks = [ti for ti in range(len(track_boxes)) if ti not in used_tracks]
    unmatched_dets = [di for di in range(len(detection_boxes)) if di not in used_dets]
    return matches, unmatched_tracks, unmatched_dets
