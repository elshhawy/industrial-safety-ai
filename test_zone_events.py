from industrial_safety.safety.zone_events import (
    ENTRY,
    EXIT,
    INSIDE,
    LOST_INSIDE,
    OUTSIDE,
    ZoneStateMachine,
)


def run(machine, values, start=1):
    """Feed one observation per frame for track 1 and collect events."""
    events = []
    for offset, value in enumerate(values):
        events += machine.update(start + offset, {1: value})
    return events


def kinds(events):
    return [event.kind for event in events]


def test_entry_after_enough_consecutive_frames():
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)
    events = run(machine, [True, True, True])
    assert kinds(events) == [ENTRY]
    assert machine.state(1) == INSIDE


def test_no_entry_when_too_few_frames():
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)
    events = run(machine, [True, True])
    assert events == []
    assert machine.state(1) == OUTSIDE


def test_flicker_is_ignored():
    machine = ZoneStateMachine(enter_frames=3, exit_frames=3)
    events = run(machine, [True, False] * 10)
    assert events == []


def test_exit_after_enough_frames_outside():
    machine = ZoneStateMachine(enter_frames=2, exit_frames=3)
    events = run(machine, [True, True, False, False, False])
    assert kinds(events) == [ENTRY, EXIT]
    assert events[1].enter_frame == 2


def test_unreliable_frames_are_skipped():
    machine = ZoneStateMachine(enter_frames=2, exit_frames=2)
    events = run(machine, [True, None, None, None, True])
    assert kinds(events) == [ENTRY]


def test_lost_inside_when_track_disappears():
    machine = ZoneStateMachine(enter_frames=1, exit_frames=3, lost_frames=5)
    assert kinds(run(machine, [True])) == [ENTRY]
    later = machine.update(10, {})
    assert kinds(later) == [LOST_INSIDE]
    assert machine.state(1) == OUTSIDE


def test_lost_outside_is_silent():
    machine = ZoneStateMachine(enter_frames=3, lost_frames=5)
    run(machine, [False])
    assert machine.update(20, {}) == []
