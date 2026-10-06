"""
image_processing/strokes.py
============================
Etapas 3 y 4 del pipeline: deteccion de lineas + extraccion de strokes.

Este modulo es el punto central para el requisito de NO acoplar el
sistema a una sola estrategia de deteccion (ver seccion 6 del brief).
Implementa tres estrategias intercambiables:

    - "contours"  -> cv2.findContours()
    - "edges"     -> cv2.Canny() + contornos sobre el mapa de bordes
    - "skeleton"  -> skeletonization (image_processing/skeleton.py)

Todas las estrategias devuelven el mismo formato de salida:

    strokes: List[List[Tuple[int, int]]]

es decir, una lista de trayectorias, cada una una lista de puntos (x, y)
en pixeles, representando una linea continua que el robot podria dibujar
sin levantar el lapiz.

Limitaciones conocidas de esta V1 (ver tambien README):
    - Ninguna de las tres estrategias resuelve perfectamente cruces de
      lineas (por ejemplo, una "X"): pueden fragmentarse en mas strokes
      de los esteticamente "correctos". Esto no afecta la validez
      geometrica del resultado (el robot igual puede dibujar cada
      fragmento), pero puede generar mas subidas/bajadas de lapiz de las
      estrictamente necesarias.
    - El orden de los strokes es simplemente el orden en que se
      descubren durante el recorrido (no hay optimizacion de ruta /
      "traveling salesman" para minimizar desplazamientos del robot).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from image_processing import skeleton as skeleton_module

Point = Tuple[int, int]
Stroke = List[Point]


def extract_strokes_from_contours(
    binary: np.ndarray,
    min_contour_length: Optional[float] = None,
) -> List[Stroke]:
    """Extrae strokes usando cv2.findContours().

    Cada contorno externo/interno detectado se convierte directamente en
    un stroke. Es la estrategia mas simple y robusta para formas cerradas
    (circulos, poligonos), pero tiende a "duplicar" trazos abiertos (una
    sola linea dibujada a mano puede generar un contorno que la rodea por
    ambos lados, ya que findContours trabaja sobre regiones, no sobre
    lineas de 1px).
    """
    if min_contour_length is None:
        min_contour_length = config.MIN_CONTOUR_LENGTH

    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    strokes: List[Stroke] = []
    for contour in contours:
        length = cv2.arcLength(contour, closed=True)
        if length < min_contour_length:
            continue
        points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
        if len(points) >= config.MIN_STROKE_POINTS:
            strokes.append(points)

    return strokes


def extract_strokes_from_edges(
    gray_or_binary: np.ndarray,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
) -> Tuple[List[Stroke], np.ndarray]:
    """Extrae strokes usando deteccion de bordes (Canny) + contornos.

    Canny produce un mapa de bordes de 1px de ancho aproximadamente, sobre
    el cual se aplica cv2.findContours() para obtener trayectorias.

    Returns:
        Tupla (strokes, edge_map) para poder tambien visualizar el mapa
        de bordes generado por Canny.
    """
    if low_threshold is None:
        low_threshold = config.CANNY_LOW_THRESHOLD
    if high_threshold is None:
        high_threshold = config.CANNY_HIGH_THRESHOLD

    edges = cv2.Canny(gray_or_binary, low_threshold, high_threshold)
    strokes = extract_strokes_from_contours(edges)
    return strokes, edges


def extract_strokes_from_skeleton(
    binary: np.ndarray,
) -> Tuple[List[Stroke], np.ndarray]:
    """Extrae strokes mediante skeletonization.

    Returns:
        Tupla (strokes, skeleton_image) para poder visualizar el
        skeleton generado.
    """
    skel = skeleton_module.skeletonize_image(binary)
    strokes = skeleton_module.skeleton_to_strokes(skel)
    return strokes, skel


def extract_strokes(
    binary: np.ndarray,
    gray: np.ndarray,
    method: Optional[str] = None,
) -> dict:
    """Punto de entrada unico: despacha a la estrategia configurada.

    Args:
        binary: mascara binaria limpia (trazo = 255).
        gray: imagen en escala de grises (usada por el metodo "edges",
            que suele funcionar mejor sobre grises suavizados que sobre
            una mascara ya binarizada, aunque tambien acepta binaria).
        method: "contours", "edges" o "skeleton". Por defecto
            config.LINE_EXTRACTION_METHOD.

    Returns:
        dict con:
            - "strokes": List[Stroke]
            - "method": str
            - "extra": imagen intermedia relevante para visualizacion
              (edge map para "edges", skeleton para "skeleton", None
              para "contours").
    """
    if method is None:
        method = config.LINE_EXTRACTION_METHOD

    method = method.lower()

    if method == "contours":
        strokes = extract_strokes_from_contours(binary)
        extra = None
    elif method == "edges":
        strokes, extra = extract_strokes_from_edges(binary)
    elif method == "skeleton":
        strokes, extra = extract_strokes_from_skeleton(binary)
    else:
        raise ValueError(
            f"LINE_EXTRACTION_METHOD desconocido: '{method}'. "
            "Usa 'contours', 'edges' o 'skeleton'."
        )

    return {"strokes": strokes, "method": method, "extra": extra}


def total_points(strokes: List[Stroke]) -> int:
    """Cuenta el numero total de puntos en una lista de strokes."""
    return sum(len(stroke) for stroke in strokes)


def total_length(strokes: List[Stroke]) -> float:
    """Calcula la longitud total (en px) de todas las trayectorias."""
    total = 0.0
    for stroke in strokes:
        for i in range(1, len(stroke)):
            x1, y1 = stroke[i - 1]
            x2, y2 = stroke[i]
            total += float(np.hypot(x2 - x1, y2 - y1))
    return total
