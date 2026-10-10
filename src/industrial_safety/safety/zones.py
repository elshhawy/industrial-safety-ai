"""Zone geometry helpers for danger-zone checks."""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Sequence[float]]) -> bool:
    """Return True if ``point`` is inside ``polygon`` (ray casting).

    The polygon is a list of ``[x, y]`` vertices in pixel coordinates. A ray is
    cast from the point to the right; an odd number of edge crossings means the
    point is inside.
    """
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside
