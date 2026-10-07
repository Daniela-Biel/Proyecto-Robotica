"""
image_processing/simplification.py
====================================
Etapa 5 del pipeline: simplificacion de trayectorias.

Reduce el numero de puntos de cada stroke conservando su forma general,
usando el algoritmo Ramer-Douglas-Peucker (implementado por OpenCV como
cv2.approxPolyDP). Esto es importante porque:

    - Strokes extraidos de contornos/skeleton pueden tener miles de
      puntos casi colineales (un punto por cada pixel recorrido).
    - Un robot no necesita ni deberia recibir esa cantidad de puntos:
      basta con los "puntos relevantes" que definen la forma (esquinas,
      cambios de curvatura).

La tolerancia (epsilon) es configurable: valores mas altos simplifican
mas agresivamente (menos puntos, mayor perdida de detalle).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

Point = Tuple[int, int]
Stroke = List[Point]


def simplify_stroke(stroke: Stroke, epsilon: Optional[float] = None) -> Stroke:
    """Simplifica un unico stroke usando Ramer-Douglas-Peucker.

    Args:
        stroke: lista de puntos (x, y).
        epsilon: tolerancia maxima de distancia (px) entre la curva
            original y la simplificada. Por defecto
            config.SIMPLIFICATION_EPSILON.

    Returns:
        Stroke simplificado (subconjunto de puntos originales).
    """
    if epsilon is None:
        epsilon = config.SIMPLIFICATION_EPSILON

    if len(stroke) < 3:
        # No hay nada que simplificar en una linea de 1-2 puntos.
        return list(stroke)

    contour = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(contour, epsilon, closed=False)
    simplified = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    # Salvaguarda: approxPolyDP no deberia devolver menos de 2 puntos,
    # pero por robustez verificamos.
    if len(simplified) < 2:
        return [stroke[0], stroke[-1]]

    return simplified


def simplify_strokes(
    strokes: List[Stroke], epsilon: Optional[float] = None
) -> List[Stroke]:
    """Aplica simplify_stroke a una lista completa de strokes."""
    return [simplify_stroke(stroke, epsilon) for stroke in strokes]
