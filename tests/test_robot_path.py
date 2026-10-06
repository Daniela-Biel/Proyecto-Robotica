"""Pruebas de la etapa robot_path (ejecutar con: python -m pytest tests/)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_processing import robot_path  # noqa: E402


def _segments(stroke):
    return np.linalg.norm(np.diff(np.asarray(stroke), axis=0), axis=1)


def test_densify_respects_max_segment():
    out = robot_path.densify_strokes([[(0, 0), (10, 0), (10, 7.3)]], 1.0)
    assert _segments(out[0]).max() <= 1.0 + 1e-9
    assert out[0][0] == (0.0, 0.0) and out[0][-1] == (10.0, 7.3)


def test_split_long_strokes_keeps_continuity_and_length():
    stroke = [(float(x), 0.0) for x in range(0, 101)]  # 100 mm
    parts = robot_path.split_long_strokes([stroke], 30.0)
    assert len(parts) == 4
    assert all(robot_path.stroke_length(p) <= 30.0 + 1e-9 for p in parts)
    for a, b in zip(parts[:-1], parts[1:]):
        assert np.allclose(a[-1], b[0])  # el siguiente empieza donde termino el anterior
    assert np.isclose(sum(robot_path.stroke_length(p) for p in parts), 100.0)


def test_order_and_join_reverses_and_joins():
    a = [(0.0, 0.0), (10.0, 0.0)]
    b = [(20.0, 0.0), (11.0, 0.0)]  # su FIN esta junto al fin de a
    out = robot_path.order_and_join_strokes([b, a], join_gap=1.5)
    assert len(out) == 1
    assert out[0][0] == (0.0, 0.0) and out[0][-1] == (20.0, 0.0)


def test_fit_to_area_uniform_scale_and_flip():
    # imagen 100x200 px en un area 200x200 mm sin margen -> 1 mm/px, centrada en x
    strokes, scale = robot_path.fit_to_area([[(0, 0), (100, 200)]], 100, 200, 200, 200)
    assert np.isclose(scale, 1.0)
    (x0, y0), (x1, y1) = strokes[0]
    assert np.isclose(x0, 50) and np.isclose(y0, 200)   # arriba-izq de la imagen -> y alta
    assert np.isclose(x1, 150) and np.isclose(y1, 0)


def test_smooth_keeps_endpoints():
    zigzag = [(i, i % 2) for i in range(20)]
    out = robot_path.smooth_stroke(zigzag, 5)
    assert out[0] == (0.0, 0.0) and out[-1] == (19.0, 1.0)
    assert np.ptp([p[1] for p in out[3:-3]]) < 0.5
