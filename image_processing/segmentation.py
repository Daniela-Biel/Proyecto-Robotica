"""
image_processing/segmentation.py
=================================
Etapa 2 del pipeline: segmentacion / limpieza de la mascara binaria.

Tras el threshold, la mascara binaria suele contener:
    - Pequenas manchas de ruido (puntos sueltos, granos de textura del papel).
    - Pequenos huecos dentro de trazos que deberian ser continuos.

Este modulo aplica operaciones morfologicas y filtrado por area de
componentes conexos para "limpiar" la mascara antes de pasarla a la
etapa de deteccion de lineas.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config


def clean_binary_mask(
    binary: np.ndarray,
    kernel_size: Optional[int] = None,
    min_component_area: Optional[int] = None,
) -> np.ndarray:
    """Limpia una mascara binaria (trazo = 255, fondo = 0).

    Pasos:
        1. Closing: cierra pequenos huecos dentro de las lineas.
        2. Opening: elimina puntos de ruido aislados.
        3. Filtrado por area de componentes conexos: elimina manchas
           pequenas que sobrevivieron a las operaciones morfologicas.

    Args:
        binary: mascara binaria de entrada (0/255).
        kernel_size: tamano del kernel morfologico (config.MORPH_KERNEL_SIZE).
        min_component_area: area minima (px) para conservar un componente
            (config.MIN_COMPONENT_AREA).

    Returns:
        Mascara binaria limpia (0/255).
    """
    if kernel_size is None:
        kernel_size = config.MORPH_KERNEL_SIZE
    if min_component_area is None:
        min_component_area = config.MIN_COMPONENT_AREA

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITERATIONS
    )
    opened = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITERATIONS
    )

    cleaned = remove_small_components(opened, min_component_area)
    return cleaned


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Elimina componentes conexos cuya area sea menor a `min_area`.

    Util para descartar ruido residual (granos de textura del papel,
    reflejos, etc.) que no forma parte del dibujo real.
    """
    if min_area <= 0:
        return binary.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):  # 0 es el fondo
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned
