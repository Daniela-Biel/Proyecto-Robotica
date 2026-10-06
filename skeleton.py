"""
image_processing/skeleton.py
=============================
Skeletonization: convierte trazos gruesos en una representacion de una
sola linea central (1 px de ancho).

Por que es util para este proyecto
-----------------------------------
Un robot dibujante no necesita conocer el grosor original del trazo
(cuanto tinta hay), solo la trayectoria central que el lapiz debe seguir.
Conceptualmente:

    ████            █
    ████     ->      █
    ████             █
    ████             █

El skeleton conserva la topologia del trazo (curvas, cruces, extremos)
pero reduce cada linea a su eje central, lo que facilita mucho la
extraccion de una secuencia ordenada de puntos (un "stroke").

Este modulo tambien incluye la logica para recorrer el skeleton (que es
esencialmente un grafo de pixeles) y convertirlo en una lista de
trayectorias (strokes), resolviendo cruces y extremos de forma simple.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from skimage.morphology import skeletonize

import config

Point = Tuple[int, int]


def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    """Aplica skeletonization a una mascara binaria (trazo = 255).

    scikit-image espera valores booleanos, por lo que convertimos antes y
    despues de llamar a skeletonize().
    """
    bool_mask = binary > 0
    skeleton_bool = skeletonize(bool_mask)
    skeleton = (skeleton_bool.astype(np.uint8)) * 255
    return skeleton


def _neighbors(point: Point, shape: Tuple[int, int]) -> List[Point]:
    """Devuelve los vecinos de 8-conectividad de un punto dentro de la imagen."""
    y, x = point
    h, w = shape
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                result.append((ny, nx))
    return result


def _build_pixel_graph(skeleton: np.ndarray) -> Dict[Point, List[Point]]:
    """Construye un grafo (dict de adyacencia) a partir de los pixeles
    activos del skeleton, usando 8-conectividad."""
    ys, xs = np.nonzero(skeleton)
    active = set(zip(ys.tolist(), xs.tolist()))

    graph: Dict[Point, List[Point]] = {}
    for point in active:
        neighs = [n for n in _neighbors(point, skeleton.shape) if n in active]
        graph[point] = neighs
    return graph


def skeleton_to_strokes(skeleton: np.ndarray) -> List[List[Point]]:
    """Convierte una imagen skeleton en una lista de strokes (trayectorias).

    Estrategia (deliberadamente simple para esta V1):
        1. Se construye un grafo de pixeles activos (8-conectividad).
        2. Se identifican "endpoints" (pixeles con 1 solo vecino) y
           "juntions" (pixeles con 3+ vecinos, es decir, cruces de lineas).
        3. Se recorre el grafo empezando por cada endpoint no visitado,
           caminando pixel a pixel hasta llegar a otro endpoint, a una
           juntion, o hasta quedarse sin vecinos no visitados.
        4. Los pixeles restantes (por ejemplo, loops cerrados sin
           endpoints, como un circulo) se recorren igualmente empezando
           por cualquier pixel no visitado.

    NOTA / limitacion conocida: en cruces complejos (mas de 2 lineas que
    se tocan) la forma de "continuar" tras una juntion es ambigua. Esta
    implementacion elige heuristicamente el vecino no visitado mas
    cercano en direccion de avance, lo cual funciona razonablemente bien
    para dibujos simples pero puede fragmentar trazos en cruces muy
    densos. Ver README para más detalle.

    Returns:
        Lista de strokes, cada uno una lista de puntos (x, y) en pixeles.
        Nota: los puntos internamente se manejan como (fila, columna) =
        (y, x) por conveniencia con NumPy, pero esta funcion los devuelve
        ya en convencion imagen estandar (x, y).
    """
    graph = _build_pixel_graph(skeleton)
    if not graph:
        return []

    visited = set()
    strokes: List[List[Point]] = []

    endpoints = [p for p, neighs in graph.items() if len(neighs) == 1]
    # Priorizamos empezar por endpoints (extremos reales de una linea);
    # el resto de puntos no visitados (loops cerrados) se procesan despues.
    start_points = endpoints + [p for p in graph if p not in endpoints]

    for start in start_points:
        if start in visited:
            continue
        path = _walk_path(start, graph, visited)
        if len(path) >= config.MIN_STROKE_POINTS:
            # Convertimos de (y, x) a (x, y)
            strokes.append([(x, y) for (y, x) in path])

    return strokes


def _walk_path(
    start: Point,
    graph: Dict[Point, List[Point]],
    visited: set,
) -> List[Point]:
    """Camina por el grafo desde `start` hasta un extremo natural.

    Se detiene cuando no hay vecinos no visitados, o cuando se alcanza un
    punto que ya tiene 3+ vecinos distintos al de llegada (una juntion),
    para evitar mezclar trazos distintos en un mismo stroke.
    """
    path = [start]
    visited.add(start)
    current = start
    previous = None

    while True:
        neighbors = graph.get(current, [])
        candidates = [n for n in neighbors if n not in visited]

        if not candidates:
            break

        # Si el punto actual es una juntion (cruce), nos detenemos aqui:
        # el siguiente stroke que pase por este punto se iniciara en una
        # nueva pasada (start_points) o quedara como segmento separado.
        if len(neighbors) >= 3 and previous is not None:
            break

        # Elegimos el vecino mas cercano en linea recta a la direccion de
        # avance para mantener el trazo lo mas "recto" posible en cruces.
        if previous is not None and len(candidates) > 1:
            direction = (current[0] - previous[0], current[1] - previous[1])
            candidates.sort(
                key=lambda n: -(
                    (n[0] - current[0]) * direction[0]
                    + (n[1] - current[1]) * direction[1]
                )
            )

        next_point = candidates[0]
        visited.add(next_point)
        path.append(next_point)
        previous = current
        current = next_point

    return path
