"""
image_processing/robot_path.py
==============================
Etapa final: convierte strokes en PIXELES en una trayectoria lista para el
robot (en MILIMETROS, en el marco del area de dibujo).

Por que hace falta esta etapa
-----------------------------
El control se hara con cinematica inversa (IK) punto a punto. Entre dos
waypoints consecutivos el controlador interpola (normalmente en espacio
articular), y una interpolacion articular entre dos puntos LEJANOS NO es
una linea recta en el papel: el lapiz se curva. Para que el trazo sea fiel:

    1. Los segmentos entre waypoints deben ser cortos (MAX_SEGMENT_MM):
       densify_strokes() inserta puntos intermedios.
    2. Los strokes no deben ser demasiado largos (MAX_STROKE_LENGTH_MM):
       split_long_strokes() los parte en tramos cortos, lo que limita el
       error acumulado y permite pausar/reanudar sin perder mucho.
    3. Los strokes deben ser suaves (sin el "escalon" de pixel del
       skeleton), si no el brazo vibra: smooth_stroke().
    4. Los trazos diminutos (ruido, < MIN_STROKE_LENGTH_MM) solo cuestan
       subidas/bajadas de lapiz: se eliminan.
    5. El orden importa: order_and_join_strokes() usa vecino mas cercano
       (permitiendo invertir strokes) y une strokes cuyos extremos casi se
       tocan, para minimizar desplazamientos con el lapiz arriba.
    6. La conversion px -> mm usa UNA sola escala (no deforma la cara) y
       centra el retrato en el area de dibujo; el eje Y se invierte porque
       en la imagen Y crece hacia abajo y en el robot hacia arriba.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

import config

PointF = Tuple[float, float]
StrokeF = List[PointF]


# ---------------------------------------------------------------------------
# Utilidades geometricas
# ---------------------------------------------------------------------------
def stroke_length(stroke: Sequence[Sequence[float]]) -> float:
    if len(stroke) < 2:
        return 0.0
    pts = np.asarray(stroke, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _as_list(pts: np.ndarray) -> StrokeF:
    return [(float(x), float(y)) for x, y in pts]


# ---------------------------------------------------------------------------
# En pixeles
# ---------------------------------------------------------------------------
def smooth_stroke(stroke: Sequence[Sequence[float]], window: int) -> StrokeF:
    """Media movil sobre los puntos de un stroke (extremos fijos).

    Quita el patron en escalera del skeleton (pixel a pixel) sin mover
    los extremos. En strokes cerrados (extremos juntos) se suaviza de
    forma circular para no dejar un "pico" en el punto de cierre.
    """
    pts = np.asarray(stroke, dtype=np.float64)
    if window < 2 or len(pts) < window + 2:
        return _as_list(pts)

    kernel = np.ones(window) / window
    half = window // 2
    closed = np.linalg.norm(pts[0] - pts[-1]) <= 1.5 and len(pts) > 2 * window
    out = np.empty_like(pts)
    for d in range(2):
        if closed:
            padded = np.concatenate([pts[-half - 1:-1, d], pts[:, d], pts[1:half + 1, d]])
            out[:, d] = np.convolve(padded, kernel, mode="valid")[: len(pts)]
        else:
            padded = np.pad(pts[:, d], half, mode="edge")
            out[:, d] = np.convolve(padded, kernel, mode="valid")[: len(pts)]
    if not closed:
        out[0], out[-1] = pts[0], pts[-1]
    else:
        out[-1] = out[0]
    return _as_list(out)


def order_and_join_strokes(
    strokes: List[StrokeF], join_gap: float, start: PointF = (0.0, 0.0)
) -> List[StrokeF]:
    """Ordena strokes por vecino mas cercano y une los que casi se tocan.

    - En cada paso se elige el stroke cuyo extremo (inicio O fin) este mas
      cerca de la posicion actual del lapiz; si el mas cercano es el fin,
      el stroke se invierte.
    - Si esa distancia es <= join_gap, el stroke se concatena al anterior
      (el lapiz no se levanta). Esto repara cortes pequenos del skeleton
      (cruces, uniones) y reduce el numero de subidas de lapiz.

    Complejidad O(n^2) con NumPy; para los cientos de strokes de un
    retrato tarda milisegundos.
    """
    remaining = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 2]
    if not remaining:
        return []

    starts = np.array([s[0] for s in remaining])
    ends = np.array([s[-1] for s in remaining])
    used = np.zeros(len(remaining), bool)

    ordered: List[np.ndarray] = []
    pos = np.asarray(start, dtype=np.float64)
    for _ in range(len(remaining)):
        d_start = np.linalg.norm(starts - pos, axis=1)
        d_end = np.linalg.norm(ends - pos, axis=1)
        d_start[used] = np.inf
        d_end[used] = np.inf
        i_s, i_e = int(np.argmin(d_start)), int(np.argmin(d_end))
        if d_start[i_s] <= d_end[i_e]:
            idx, dist, s = i_s, d_start[i_s], remaining[i_s]
        else:
            idx, dist, s = i_e, d_end[i_e], remaining[i_e][::-1]
        used[idx] = True

        if ordered and dist <= join_gap:
            ordered[-1] = np.vstack([ordered[-1], s])
        else:
            ordered.append(s)
        pos = s[-1]

    return [_as_list(s) for s in ordered]


def _end_direction(pts: np.ndarray, at_end: bool, k: int = 6) -> np.ndarray:
    """Direccion (unitaria) hacia AFUERA del stroke en uno de sus extremos."""
    k = min(k, len(pts) - 1)
    v = pts[-1] - pts[-1 - k] if at_end else pts[0] - pts[k]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def link_collinear_strokes(
    strokes: List[StrokeF], max_gap: float, max_angle_deg: float
) -> List[StrokeF]:
    """Une strokes "en guiones": si el final de uno apunta hacia el inicio
    de otro (hueco <= max_gap, desvio <= max_angle_deg), se concatenan y el
    hueco se dibuja como un tramo recto.

    El XDoG corta las lineas largas (mechones de pelo, contorno de la cara)
    donde el contraste baja un poco; sin esto cada guion es una subida y
    bajada de lapiz. Se une primero el par mas cercano (greedy) y se evita
    formar ciclos.
    """
    arrs = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 2]
    n = len(arrs)
    if n < 2 or max_gap <= 0:
        return [_as_list(a) for a in arrs]

    # Extremo e = 2*i (inicio del stroke i) o 2*i + 1 (fin del stroke i)
    pos = np.array([a[0] if e % 2 == 0 else a[-1] for a in arrs for e in (0, 1)])
    out = np.array([_end_direction(a, e == 1) for a in arrs for e in (0, 1)])

    diff = pos[None, :, :] - pos[:, None, :]          # vector de i hacia j
    dist = np.linalg.norm(diff, axis=2)
    gap_dir = diff / np.maximum(dist, 1e-9)[:, :, None]
    cos_max = np.cos(np.radians(max_angle_deg))
    cos_i = np.einsum("ijk,ik->ij", gap_dir, out)     # i apunta hacia j
    cos_j = np.einsum("ijk,jk->ij", -gap_dir, out)    # j apunta hacia i
    facing = -(out @ out.T)                           # direcciones opuestas
    close = dist < 2.0
    ok = np.where(close, facing > cos_max, (cos_i > cos_max) & (cos_j > cos_max))
    ok &= dist <= max_gap
    owner = np.repeat(np.arange(n), 2)
    ok &= owner[:, None] != owner[None, :]
    ok = np.triu(ok, 1)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    link = {}  # extremo -> extremo enlazado
    ii, jj = np.nonzero(ok)
    for k in np.argsort(dist[ii, jj], kind="stable"):
        a, b = int(ii[k]), int(jj[k])
        if a in link or b in link:
            continue
        ra, rb = find(a // 2), find(b // 2)
        if ra == rb:
            continue
        parent[ra] = rb
        link[a], link[b] = b, a

    # Reconstruir cadenas empezando por strokes con un extremo libre
    done = [False] * n
    result: List[StrokeF] = []
    for start in range(n):
        if done[start]:
            continue
        if 2 * start in link and 2 * start + 1 in link:
            continue  # esta en medio de una cadena: se visita desde un extremo
        # Orientar para que el extremo libre sea el inicio
        free_start = 2 * start not in link
        idx, entry = start, (2 * start if free_start else 2 * start + 1)
        chain = []
        while True:
            done[idx] = True
            a = arrs[idx] if entry % 2 == 0 else arrs[idx][::-1]
            chain.append(a)
            exit_end = entry ^ 1
            if exit_end not in link:
                break
            nxt = link[exit_end]
            idx, entry = nxt // 2, nxt
        result.append(_as_list(np.vstack(chain)))
    return result


def filter_short_strokes(strokes: List[StrokeF], min_length: float) -> List[StrokeF]:
    return [s for s in strokes if stroke_length(s) >= min_length]


# ---------------------------------------------------------------------------
# Pixeles -> milimetros
# ---------------------------------------------------------------------------
def fit_to_area(
    strokes: List[StrokeF],
    width_px: int,
    height_px: int,
    area_w_mm: float,
    area_h_mm: float,
    margin_mm: float = 0.0,
    origin_mm: PointF = (0.0, 0.0),
    flip_y: bool = True,
) -> Tuple[List[StrokeF], float]:
    """Escala UNIFORME (mm/px) para que la imagen quepa en el area util,
    centrada. Devuelve (strokes_mm, escala_mm_por_px).

    origin_mm es la posicion de la esquina inferior-izquierda del area de
    dibujo en el marco del robot, para que las coordenadas exportadas ya
    esten en ese marco.
    """
    usable_w = area_w_mm - 2 * margin_mm
    usable_h = area_h_mm - 2 * margin_mm
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("El margen es mayor que el area de dibujo.")
    scale = min(usable_w / width_px, usable_h / height_px)
    off_x = origin_mm[0] + (area_w_mm - width_px * scale) / 2
    off_y = origin_mm[1] + (area_h_mm - height_px * scale) / 2

    out: List[StrokeF] = []
    for s in strokes:
        pts = np.asarray(s, dtype=np.float64) * scale
        if flip_y:
            pts[:, 1] = height_px * scale - pts[:, 1]
        pts[:, 0] += off_x
        pts[:, 1] += off_y
        out.append(_as_list(pts))
    return out, scale


# ---------------------------------------------------------------------------
# En milimetros
# ---------------------------------------------------------------------------
def simplify_strokes_mm(strokes: List[StrokeF], epsilon_mm: float) -> List[StrokeF]:
    """Ramer-Douglas-Peucker en mm (acepta coordenadas float)."""
    if epsilon_mm <= 0:
        return strokes
    out = []
    for s in strokes:
        if len(s) < 3:
            out.append(list(s))
            continue
        approx = cv2.approxPolyDP(np.asarray(s, np.float32).reshape(-1, 1, 2), epsilon_mm, False)
        out.append([(float(p[0][0]), float(p[0][1])) for p in approx])
    return out


def split_long_strokes(strokes: List[StrokeF], max_length_mm: float) -> List[StrokeF]:
    """Parte strokes mas largos que max_length_mm en tramos de longitud
    similar (el ultimo punto de un tramo es el primero del siguiente, asi
    que el dibujo queda continuo). max_length_mm <= 0 desactiva."""
    if max_length_mm <= 0:
        return strokes
    out: List[StrokeF] = []
    for s in strokes:
        total = stroke_length(s)
        if total <= max_length_mm:
            out.append(s)
            continue
        n_parts = int(np.ceil(total / max_length_mm))
        target = total / n_parts
        pts = np.asarray(s, dtype=np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cuts = [target * k for k in range(1, n_parts)]
        current = [pts[0]]
        ci = 0
        for i in range(1, len(pts)):
            while ci < len(cuts) and cum[i] >= cuts[ci]:
                # punto exacto del corte, interpolado en el segmento i-1 -> i
                t = (cuts[ci] - cum[i - 1]) / max(seg[i - 1], 1e-12)
                p = pts[i - 1] + t * (pts[i] - pts[i - 1])
                current.append(p)
                out.append(_as_list(np.array(current)))
                current = [p]
                ci += 1
            if not np.allclose(current[-1], pts[i]):
                current.append(pts[i])
        if len(current) >= 2:
            out.append(_as_list(np.array(current)))
    return out


def densify_strokes(strokes: List[StrokeF], max_segment_mm: float) -> List[StrokeF]:
    """Inserta puntos para que ningun segmento mida mas de max_segment_mm.

    Clave para IK: el controlador solo ve los waypoints; con segmentos
    cortos la trayectoria real entre ellos es practicamente recta.
    """
    if max_segment_mm <= 0:
        return strokes
    out = []
    for s in strokes:
        pts = np.asarray(s, dtype=np.float64)
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = int(np.ceil(np.linalg.norm(b - a) / max_segment_mm))
            for k in range(1, max(n, 1) + 1):
                dense.append(a + (b - a) * k / max(n, 1))
        out.append(_as_list(np.array(dense)))
    return out


def path_stats(strokes: List[StrokeF], start: PointF = (0.0, 0.0)) -> dict:
    """Metricas utiles para validar la trayectoria antes de enviarla."""
    draw = sum(stroke_length(s) for s in strokes)
    travel = 0.0
    pos = np.asarray(start, dtype=np.float64)
    max_seg = 0.0
    for s in strokes:
        pts = np.asarray(s, dtype=np.float64)
        travel += float(np.linalg.norm(pts[0] - pos))
        pos = pts[-1]
        if len(pts) > 1:
            max_seg = max(max_seg, float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    n_pts = sum(len(s) for s in strokes)
    lengths = [stroke_length(s) for s in strokes] or [0.0]
    est_time = (
        draw / config.PEN_DRAW_SPEED_MM_S
        + travel / config.PEN_TRAVEL_SPEED_MM_S
        + len(strokes) * config.PEN_LIFT_TIME_S
    )
    all_pts = np.vstack([np.asarray(s) for s in strokes]) if strokes else np.zeros((1, 2))
    return {
        "num_strokes": len(strokes),
        "num_waypoints": n_pts,
        "draw_length_mm": draw,
        "travel_length_mm": travel,
        "max_segment_mm": max_seg,
        "max_stroke_length_mm": max(lengths),
        "mean_stroke_length_mm": float(np.mean(lengths)),
        "bbox_mm": [float(v) for v in (*all_pts.min(axis=0), *all_pts.max(axis=0))],
        "estimated_time_s": est_time,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def save_robot_json(strokes: List[StrokeF], meta: dict, output_path: str | Path) -> None:
    data = dict(meta)
    data["units"] = "mm"
    data["strokes"] = [
        {"id": i, "points": [[round(x, 3), round(y, 3)] for x, y in s]}
        for i, s in enumerate(strokes, start=1)
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def save_waypoints_csv(strokes: List[StrokeF], output_path: str | Path) -> None:
    """CSV plano, una fila por waypoint, en el ORDEN de ejecucion:

        stroke_id, x_mm, y_mm, pen

    pen = 0 -> moverse a este punto con el lapiz ARRIBA (primer punto de
               cada stroke: desplazamiento desde el stroke anterior).
    pen = 1 -> moverse a este punto con el lapiz ABAJO (dibujando).

    Formato pensado para MATLAB: T = readmatrix('robot_waypoints.csv');
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stroke_id", "x_mm", "y_mm", "pen"])
        for sid, s in enumerate(strokes, start=1):
            for k, (x, y) in enumerate(s):
                w.writerow([sid, f"{x:.3f}", f"{y:.3f}", 0 if k == 0 else 1])
