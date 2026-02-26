import math
import logging
from typing import List, Optional, Tuple
from config import VECTOR_WINDOW, VECTOR_MIN_DISTANCE

logger = logging.getLogger(__name__)

def vector_from_points(
    points: List[Tuple[float, float]],
    mode: str = "approach"  # "approach" or "departure"
) -> Optional[Tuple[float, float]]:
    """Compute a normalized movement vector from a list of points.

    For 'approach': uses the first VECTOR_WINDOW points.
    For 'departure': uses the last VECTOR_WINDOW points.

    The resulting vector is normalized to length 1.0. For 'approach' it is inverted
    to point towards the nest direction (as implemented in the original code).

    Args:
        points: List of (x, y) center points.
        mode: "approach" or "departure".

    Returns:
        (vx, vy) normalized vector, or None if movement is too small / insufficient data.

    Raises:
        ValueError: If an unknown mode is provided.
    """

    if len(points) < VECTOR_WINDOW:
        return None

    if mode == "approach":
        sub_points = points[:VECTOR_WINDOW]
    elif mode == "departure":
        sub_points = points[-VECTOR_WINDOW:]
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    x0, y0 = sub_points[0]
    x1, y1 = sub_points[-1]

    dx = x1 - x0
    dy = y1 - y0

    length = math.hypot(dx, dy)
    if length < VECTOR_MIN_DISTANCE:
        return None
    
    vx = dx / length
    vy = dy / length

    # -- invert approach vector to point towards nest direction --
    if mode == "approach":
        vx = -vx
        vy = -vy

    logger.debug("Vector (%s): dx=%.3f dy=%.3f -> vx=%.3f vy=%.3f", mode, dx, dy, vx, vy)
    return float(vx), float(vy)