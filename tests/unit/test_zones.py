from industrial_safety.safety.zones import point_in_polygon

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
L_SHAPE = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]


def test_point_inside_square():
    assert point_in_polygon((5, 5), SQUARE)


def test_point_outside_square():
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((-1, 5), SQUARE)
    assert not point_in_polygon((5, 20), SQUARE)


def test_concave_polygon():
    assert point_in_polygon((2, 8), L_SHAPE)
    assert not point_in_polygon((8, 8), L_SHAPE)
