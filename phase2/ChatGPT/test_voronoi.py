import os
import numpy as np
import pytest

from voronoi_app import (
    read_points,
    compute_perpendicular_bisector,
    generate_voronoi_edges,
)


def test_read_points_valid(tmp_path):
    file = tmp_path / "points.txt"
    file.write_text("0,0\n1,1\n2,2")

    points = read_points(str(file))

    assert isinstance(points, np.ndarray)
    assert points.shape == (3, 2)


def test_read_points_invalid_format(tmp_path):
    file = tmp_path / "points.txt"
    file.write_text("0;0\n1,1")

    with pytest.raises(ValueError):
        read_points(str(file))


def test_read_points_less_than_two(tmp_path):
    file = tmp_path / "points.txt"
    file.write_text("0,0")

    with pytest.raises(ValueError):
        read_points(str(file))


def test_read_points_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_points("inexistant.txt")


def test_perpendicular_bisector():
    p1 = np.array([0, 0])
    p2 = np.array([2, 0])

    a, b, c = compute_perpendicular_bisector(p1, p2)

    # La médiatrice doit être x = 1
    assert abs(a * 1 + b * 0 + c) < 1e-6


def test_generate_edges():
    points = np.array([[0, 0], [2, 0]])
    edges = generate_voronoi_edges(points, (-10, 10, -10, 10))

    assert len(edges) == 1
    assert len(edges[0][0]) == 500