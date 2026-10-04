"""ByteTrack wrapper for the industrial-safety-ai project.

This module is intentionally decoupled from whatever detection module
produces the boxes (YOLO, another model, a mocked list in a unit test...).
The only thing ``PersonTracker.update()`` needs each frame is a plain list
of ``Detection`` objects (bbox + confidence + class_id). It returns a list
of ``Track`` objects with a stable ``track_id`` that persists across frames
for the same physical person.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class Detection:
    """One raw detection for a single frame, as produced by any detector.

    The box is (x1, y1, x2, y2) in pixel coordinates, top-left to bottom-right.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0  # 0 = person, kept generic on purpose


@dataclass(frozen=True)
class Track:
    """One tracked object for the current frame, after ByteTrack matching."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Reference point used by the danger-zone logic (feet position)."""
        return ((self.x1 + self.x2) / 2, self.y2)


class PersonTracker:
    """Thin wrapper around supervision's ByteTrack implementation.

    Usage::

        tracker = PersonTracker()
        for frame in video_frames:
            detections = my_detector.run(frame)  # -> list[Detection]
            tracks = tracker.update(detections)  # -> list[Track], stable IDs

    One instance is one tracking session (for example one camera stream).
    Do not share an instance across unrelated video sources, or IDs and
    motion state will mix between them.

    Note: ``supervision.ByteTrack`` is deprecated upstream and scheduled for
    removal in a future release, so the supervision version must be pinned.
    If it disappears, replace ``self._tracker`` with its successor. The public
    API of this class does not need to change.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        """Create a tracker.

        Args:
            track_activation_threshold: minimum confidence for a detection to
                start a new track. Lower-confidence detections are still used
                in ByteTrack's second matching stage to keep tracks alive.
            lost_track_buffer: frames a track is kept alive without a
                matching detection (occlusion tolerance) before it is dropped.
            minimum_matching_threshold: IoU threshold used for matching.
            frame_rate: expected FPS of the input stream.
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Feed one frame's detections in and get tracks with stable IDs.

        Call once per frame, in order.
        """
        sv_detections = _to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_detections)
        return _from_sv_detections(tracked)

    def reset(self) -> None:
        """Clear all track state (for example when switching to a new video)."""
        self._tracker.reset()


def _to_sv_detections(detections: Sequence[Detection]) -> sv.Detections:
    if len(detections) == 0:
        return sv.Detections.empty()
    xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in detections], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _from_sv_detections(detections: sv.Detections) -> list[Track]:
    tracks: list[Track] = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]

        tracker_id = -1
        if detections.tracker_id is not None:
            tracker_id = int(detections.tracker_id[i])

        confidence = 0.0
        if detections.confidence is not None:
            confidence = float(detections.confidence[i])

        class_id = 0
        if detections.class_id is not None:
            class_id = int(detections.class_id[i])

        tracks.append(
            Track(
                track_id=tracker_id,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=confidence,
                class_id=class_id,
            )
        )
    return tracks
