"""
visualization/visualize.py
============================
Etapa final del pipeline (y utilidades usadas durante todas las demas):
generacion de imagenes de salida para poder inspeccionar visualmente cada
paso del procesamiento.

Guardar cada etapa por separado (en vez de solo el resultado final) es un
requisito explicito del proyecto: si una imagen falla, se debe poder
determinar en que paso ocurrio el problema.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

Point = Tuple[int, int]
Stroke = List[Point]


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen de 1 canal (grises/binaria) a BGR para poder
    dibujar sobre ella en color, o la devuelve sin cambios si ya es BGR."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def save_step_image(image: np.ndarray, output_path: str | Path) -> None:
    """Guarda una imagen intermedia en disco, si SAVE_INTERMEDIATE_IMAGES
    esta activo en config."""
    if not config.SAVE_INTERMEDIATE_IMAGES:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_strokes(
    base_image: np.ndarray,
    strokes: List[Stroke],
    thickness: Optional[int] = None,
    draw_points: bool = False,
) -> np.ndarray:
    """Dibuja los strokes sobre una copia de la imagen base.

    Cada stroke se dibuja con un color distinto (ciclando sobre
    config.STROKE_COLORS) para poder distinguir visualmente donde termina
    una trayectoria y empieza otra.
    """
    if thickness is None:
        thickness = config.STROKE_LINE_THICKNESS

    canvas = _ensure_bgr(base_image)
    colors = config.STROKE_COLORS

    for idx, stroke in enumerate(strokes):
        color = colors[idx % len(colors)]
        points = np.array(stroke, dtype=np.int32)

        if len(points) >= 2:
            cv2.polylines(
                canvas, [points], isClosed=False, color=color, thickness=thickness
            )
        elif len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), thickness + 1, color, -1)

        if draw_points:
            for x, y in stroke:
                cv2.circle(
                    canvas, (int(x), int(y)), config.SIMPLIFIED_POINT_RADIUS, color, -1
                )

    return canvas


def draw_simplified_points(
    base_image: np.ndarray, strokes: List[Stroke]
) -> np.ndarray:
    """Genera una visualizacion enfocada en los puntos tras la
    simplificacion (util para juzgar si el epsilon es adecuado)."""
    return draw_strokes(base_image, strokes, draw_points=True)


def draw_final_trajectory(
    original_image: np.ndarray, strokes: List[Stroke]
) -> np.ndarray:
    """Genera la imagen final: original + trayectorias superpuestas."""
    return draw_strokes(original_image, strokes, draw_points=False)


def draw_robot_path(
    strokes_mm: List[List[Tuple[float, float]]],
    area_mm: Tuple[float, float],
    origin_mm: Tuple[float, float] = (0.0, 0.0),
    px_per_mm: Optional[int] = None,
    show_travel: bool = True,
    color_strokes: bool = True,
) -> np.ndarray:
    """Dibuja la trayectoria FINAL del robot (en mm) sobre una hoja blanca
    del tamano del area de dibujo.

    - color_strokes=True: cada stroke de un color (se ve donde se corta).
    - color_strokes=False: todo en negro -> simulacion de lo que el robot
      va a dibujar realmente.
    - show_travel: desplazamientos con lapiz arriba como lineas grises
      finas (permite juzgar el orden de los strokes).

    El eje Y del robot crece hacia arriba, asi que se invierte al pintar.
    """
    if px_per_mm is None:
        px_per_mm = config.PREVIEW_PX_PER_MM
    w_mm, h_mm = area_mm
    W, H = int(round(w_mm * px_per_mm)), int(round(h_mm * px_per_mm))
    canvas = np.full((H, W, 3), 255, np.uint8)

    def to_px(p):
        x = (p[0] - origin_mm[0]) * px_per_mm
        y = H - (p[1] - origin_mm[1]) * px_per_mm if config.FLIP_Y else (p[1] - origin_mm[1]) * px_per_mm
        return int(round(x)), int(round(y))

    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (200, 200, 200), 1)
    prev = to_px(origin_mm)  # el robot parte del origen del area
    colors = config.STROKE_COLORS
    for idx, stroke in enumerate(strokes_mm):
        pts = np.array([to_px(p) for p in stroke], np.int32)
        if show_travel:
            cv2.line(canvas, prev, tuple(pts[0]), (190, 190, 190), 1, cv2.LINE_AA)
        color = colors[idx % len(colors)] if color_strokes else (0, 0, 0)
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
        prev = tuple(pts[-1])
    return canvas


def draw_face_detection(image: np.ndarray, face, crop_rect) -> np.ndarray:
    """Original con el rectangulo de la cara (verde) y del recorte (azul)."""
    canvas = _ensure_bgr(image)
    t = max(2, image.shape[0] // 300)
    if crop_rect is not None:
        x, y, w, h = crop_rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 128, 0), t)
    if face is not None:
        x, y, w, h = face
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), t)
    else:
        cv2.putText(canvas, "SIN CARA DETECTADA", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
    return canvas
