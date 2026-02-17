import math
from typing import List, Optional, Tuple
from config import VECTOR_WINDOW, VEKTOR_MIN_DISTANCE

def vector_from_points(
    points: List[Tuple[float, float]],
    mode: str = "approach"  # "approach" oder "departure"
) -> Optional[Tuple[float, float]]:
    """
    Computes a normalized movement vector from a list of points.
    
    - For 'approach': uses first and last of the first half of points.
    - For 'departure': uses first and last of the last half of points.
    Returns None if movement is too small.
    """

    if len(points) < 2 or len(points) < VECTOR_WINDOW:
        return None

    if mode == "approach":
        # Take first n frames (or all if < VECTOR_WINDOW)
        sub_points = points[:VECTOR_WINDOW]
    elif mode == "departure":
        # Take last n frames
        sub_points = points[-VECTOR_WINDOW:]
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    print(f"Calculating {mode} vector from points: {sub_points} out of {len(points)} total points")

    x0, y0 = sub_points[0]
    x1, y1 = sub_points[-1]

    dx = x1 - x0
    dy = y1 - y0

    length = math.hypot(dx, dy)
    if length < VEKTOR_MIN_DISTANCE:
        return None

    return (dx / length, dy / length)