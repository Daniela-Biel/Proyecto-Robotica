"""
image_processing/coordinates.py
=================================
Etapa 6 del pipeline: coordenadas 2D.

En esta V1 las trayectorias se manejan en pixeles. Este modulo provee una
etapa INDEPENDIENTE y explicita para convertir de pixeles a milimetros,
de forma que en el futuro sea trivial insertar aqui una calibracion mas
sofisticada (por ejemplo, a partir de la correccion de perspectiva) sin
tocar el resto del pipeline.

Tambien se encarga de serializar las trayectorias a JSON y CSV, dejando
la estructura preparada para agregar en el futuro campos como Z,
pen_down/pen_up, velocidad, etc. (necesarios para el control real del
robot, que NO se implementa en esta V1).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

Point = Tuple[int, int]
Stroke = List[Point]


def pixel_strokes_to_mm(
    strokes: List[Stroke],
    image_width_px: int,
    image_height_px: int,
    physical_width_mm: float,
    physical_height_mm: float,
) -> List[List[Tuple[float, float]]]:
    """Convierte una lista de strokes de pixeles a milimetros.

    Se asume una escala lineal e independiente en X e Y (es decir, no se
    asume que el pixel sea cuadrado en el mundo fisico). Si la imagen no
    tiene la misma relacion de aspecto que el area fisica, esto puede
    introducir una distorsion leve; para la V1 es un compromiso aceptable
    y queda documentado como limitacion conocida.
    """
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("image_width_px y image_height_px deben ser > 0")

    scale_x = physical_width_mm / image_width_px
    scale_y = physical_height_mm / image_height_px

    mm_strokes: List[List[Tuple[float, float]]] = []
    for stroke in strokes:
        mm_stroke = [
            (round(x * scale_x, 3), round(y * scale_y, 3)) for x, y in stroke
        ]
        mm_strokes.append(mm_stroke)

    return mm_strokes


def strokes_to_dict(
    strokes: List[Stroke],
    image_name: str,
    width_px: int,
    height_px: int,
) -> dict:
    """Construye la estructura de datos que luego se serializa a JSON.

    La estructura queda preparada para agregar en el futuro, por cada
    punto o por cada stroke, campos como "z", "pen_down", "pen_up",
    "speed", etc., sin romper compatibilidad con esta version.
    """
    data = {
        "image": image_name,
        "width_px": width_px,
        "height_px": height_px,
        "strokes": [],
    }

    for idx, stroke in enumerate(strokes, start=1):
        data["strokes"].append(
            {
                "id": idx,
                "points": [[int(x), int(y)] for x, y in stroke],
            }
        )

    return data


def save_json(data: dict, output_path: str | Path) -> None:
    """Guarda la estructura de trayectorias en un archivo JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(strokes: List[Stroke], output_path: str | Path) -> None:
    """Guarda las trayectorias en un CSV con columnas:
    stroke_id, point_id, x, y
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stroke_id", "point_id", "x", "y"])
        for stroke_id, stroke in enumerate(strokes, start=1):
            for point_id, (x, y) in enumerate(stroke, start=1):
                writer.writerow([stroke_id, point_id, x, y])
