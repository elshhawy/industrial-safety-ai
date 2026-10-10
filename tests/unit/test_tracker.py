from industrial_safety.tracking.tracker import Detection, PersonTracker, Track


def person(x1=100.0, y1=100.0, x2=140.0, y2=220.0, confidence=0.9):
    return Detection(x1, y1, x2, y2, confidence)


def valid_ids(tracks):
    return [track.track_id for track in tracks if track.track_id >= 0]


def test_empty_frame_returns_no_tracks():
    tracker = PersonTracker()
    assert tracker.update([]) == []


def test_track_geometry_helpers():
    track = Track(track_id=1, x1=10.0, y1=20.0, x2=30.0, y2=80.0, confidence=0.9)
    assert track.bbox == (10.0, 20.0, 30.0, 80.0)
    assert track.center == (20.0, 50.0)
    assert track.bottom_center == (20.0, 80.0)


def test_same_person_keeps_same_id():
    tracker = PersonTracker()
    ids = []
    for step in range(15):
        shift = step * 2.0
        box = person(100 + shift, 100, 140 + shift, 220)
        ids.extend(valid_ids(tracker.update([box])))
    assert len(ids) > 0
    assert len(set(ids)) == 1


def test_two_people_get_different_ids():
    tracker = PersonTracker()
    ids = set()
    for step in range(15):
        shift = step * 2.0
        left = person(100 + shift, 100, 140 + shift, 220)
        right = person(400 + shift, 100, 440 + shift, 220)
        ids.update(valid_ids(tracker.update([left, right])))
    assert len(ids) == 2


def test_reset_clears_state():
    tracker = PersonTracker()
    for _ in range(5):
        tracker.update([person()])
    tracker.reset()
    assert tracker.update([]) == []
