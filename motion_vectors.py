import math
from typing import List, Optional, Tuple

def vector_from_points(
    points: List[Tuple[float, float]],
    min_distance: float = 5.0
) -> Optional[Tuple[float, float]]:
    """
    Computes a normalized movement vector from a list of points.

    Uses first and last point.
    Returns None if movement is too small.
    """

    if len(points) < 2:
        return None

    x0, y0 = points[0]
    x1, y1 = points[-1]

    dx = x1 - x0
    dy = y1 - y0

    length = math.hypot(dx, dy)
    if length < min_distance:
        return None

    return (dx / length, dy / length)
