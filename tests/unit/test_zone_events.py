"""Entry/exit state machine with debouncing for danger-zone checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

OUTSIDE = "OUTSIDE"
INSIDE = "INSIDE"

ENTRY = "ENTRY"
EXIT = "EXIT"
LOST_INSIDE = "LOST_INSIDE"


@dataclass(frozen=True)
class ZoneEvent:
    """A zone event for one tracked worker."""

    kind: str
    track_id: int
    frame: int
    enter_frame: int | None


@dataclass
class _TrackState:
    state: str = OUTSIDE
    in_count: int = 0
    out_count: int = 0
    last_seen: int = 0
    enter_frame: int | None = None


class ZoneStateMachine:
    """Turn per-frame inside/outside checks into ENTRY/EXIT events.

    A worker's confirmed state only changes after the new observation is
    seen for several consecutive frames (debouncing), so a worker standing
    near the zone border does not produce a burst of false events.
    """

    def __init__(
        self,
        enter_frames: int = 5,
        exit_frames: int = 10,
        lost_frames: int = 60,
    ) -> None:
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.lost_frames = lost_frames
        self._tracks: dict[int, _TrackState] = {}

    def state(self, track_id: int) -> str:
        """Return the confirmed state (INSIDE or OUTSIDE) of a track."""
        track = self._tracks.get(track_id)
        return track.state if track else OUTSIDE

    def update(
        self,
        frame: int,
        observations: Mapping[int, bool | None],
    ) -> list[ZoneEvent]:
        """Process one frame.

        ``observations`` maps track id to True (inside), False (outside) or
        None (unreliable, for example a box cut by the frame edge).
        """
        events: list[ZoneEvent] = []
        for tid, raw_inside in observations.items():
            track = self._tracks.setdefault(tid, _TrackState())
            track.last_seen = frame
            if raw_inside is None:
                continue
            event = self._step(tid, track, raw_inside, frame)
            if event is not None:
                events.append(event)
        events.extend(self._drop_lost(frame))
        return events

    def _step(
        self,
        tid: int,
        track: _TrackState,
        raw_inside: bool,
        frame: int,
    ) -> ZoneEvent | None:
        if raw_inside:
            track.in_count += 1
            track.out_count = 0
            confirmed = track.in_count >= self.enter_frames
            if track.state == OUTSIDE and confirmed:
                track.state = INSIDE
                track.enter_frame = frame
                return ZoneEvent(ENTRY, tid, frame, frame)
            return None

        track.out_count += 1
        track.in_count = 0
        confirmed = track.out_count >= self.exit_frames
        if track.state == INSIDE and confirmed:
            event = ZoneEvent(EXIT, tid, frame, track.enter_frame)
            track.state = OUTSIDE
            track.enter_frame = None
            return event
        return None

    def _drop_lost(self, frame: int) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        for tid in list(self._tracks):
            track = self._tracks[tid]
            if frame - track.last_seen <= self.lost_frames:
                continue
            if track.state == INSIDE:
                events.append(
                    ZoneEvent(LOST_INSIDE, tid, frame, track.enter_frame)
                )
            del self._tracks[tid]
        return events
