"""
image_processing/preprocess.py
===============================
Etapa 1 del pipeline: preprocesamiento de la fotografia.

Responsabilidades de este modulo:
    - Cargar la imagen desde disco.
    - (Opcional) Corregir perspectiva si se conocen las esquinas del papel.
    - Convertir a escala de grises.
    - Reducir ruido (Gaussian Blur).
    - Ajustar contraste/brillo (opcional).
    - Binarizar (threshold) para obtener una mascara blanco/negro donde
      el trazo del dibujo queda en blanco (255) sobre fondo negro (0).

Cada funcion es independiente y recibe/devuelve arrays de NumPy (imagenes
de OpenCV), de forma que main.py pueda inspeccionar y guardar el resultado
de cada paso por separado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

import config


def load_image(path: str | Path) -> np.ndarray:
    """Carga una imagen desde disco en formato BGR (convencion de OpenCV).

    Lanza FileNotFoundError si la ruta no existe, y ValueError si el
    archivo no pudo ser interpretado como imagen.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro la imagen de entrada: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen (formato no soportado?): {path}")

    return image


def correct_perspective(
    image: np.ndarray,
    corners: Optional[Sequence[Tuple[float, float]]] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Corrige la perspectiva de la foto dado un conjunto de 4 esquinas.

    Args:
        image: imagen BGR original.
        corners: lista de 4 puntos (x, y) en orden
            [superior-izq, superior-der, inferior-der, inferior-izq].
            Si es None, se usa config.PERSPECTIVE_CORNERS. Si tambien es
            None, la imagen se devuelve sin modificar (esta etapa queda
            "preparada" para cuando se implemente deteccion automatica de
            esquinas, pero no es obligatoria en la V1).
        output_size: tamano (width, height) del rectangulo de salida.

    Returns:
        Imagen con perspectiva corregida (o la imagen original si no hay
        esquinas disponibles).
    """
    if corners is None:
        corners = config.PERSPECTIVE_CORNERS

    if corners is None:
        # No se implementa deteccion automatica de esquinas en la V1.
        # Esta funcion queda lista para recibir las 4 esquinas manualmente
        # (por config o en una futura UI) cuando se necesite.
        return image.copy()

    if len(corners) != 4:
        raise ValueError("corners debe contener exactamente 4 puntos (x, y).")

    if output_size is None:
        output_size = config.PERSPECTIVE_OUTPUT_SIZE

    width, height = output_size
    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, transform_matrix, (width, height))
    return warped


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR a escala de grises."""
    if len(image.shape) == 2:
        # Ya esta en escala de grises.
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
    """Aplica Gaussian Blur para reducir ruido de alta frecuencia.

    Un kernel mas grande difumina mas la imagen (menos ruido, pero tambien
    menos detalle). El valor se controla desde config.BLUR_KERNEL_SIZE.
    """
    if kernel_size is None:
        kernel_size = config.BLUR_KERNEL_SIZE

    if kernel_size <= 0:
        return gray.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1  # cv2.GaussianBlur requiere un kernel impar

    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def adjust_contrast(
    gray: np.ndarray,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> np.ndarray:
    """Ajusta contraste/brillo: salida = alpha * entrada + beta.

    alpha > 1 aumenta el contraste; beta > 0 aclara la imagen.
    Los valores por defecto (alpha=1, beta=0) dejan la imagen sin cambios.
    """
    if alpha is None:
        alpha = config.CONTRAST_ALPHA
    if beta is None:
        beta = config.CONTRAST_BETA

    if alpha == 1.0 and beta == 0:
        return gray.copy()

    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)


def binarize(
    gray: np.ndarray,
    method: Optional[str] = None,
    threshold_value: Optional[int] = None,
    invert: Optional[bool] = None,
) -> np.ndarray:
    """Binariza la imagen en escala de grises usando el metodo configurado.

    Args:
        gray: imagen en escala de grises (posiblemente ya suavizada).
        method: "otsu", "adaptive" o "fixed". Por defecto config.THRESHOLD_METHOD.
        threshold_value: solo se usa si method == "fixed".
        invert: si True, el resultado se invierte para que el trazo (mas
            oscuro que el papel) quede en blanco (255) sobre fondo negro (0).

    Returns:
        Imagen binaria (0 / 255) donde 255 representa el trazo del dibujo.
    """
    if method is None:
        method = config.THRESHOLD_METHOD
    if threshold_value is None:
        threshold_value = config.THRESHOLD_VALUE
    if invert is None:
        invert = config.INVERT_BINARY

    method = method.lower()

    if method == "otsu":
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        block_size = config.ADAPTIVE_BLOCK_SIZE
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            config.ADAPTIVE_C,
        )
    elif method == "fixed":
        _, binary = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY
        )
    else:
        raise ValueError(
            f"THRESHOLD_METHOD desconocido: '{method}'. "
            "Usa 'otsu', 'adaptive' o 'fixed'."
        )

    if invert:
        # El papel (fondo) queda tipicamente en blanco tras el threshold
        # directo (porque es la region mas clara), y el trazo en negro.
        # Invertimos para que el trazo (lo que nos interesa) sea 255.
        binary = cv2.bitwise_not(binary)

    return binary


def preprocess_pipeline(
    image: np.ndarray,
    corners: Optional[Sequence[Tuple[float, float]]] = None,
) -> dict:
    """Ejecuta toda la etapa de preprocesamiento y devuelve cada paso.

    Devolver un diccionario con TODOS los resultados intermedios (en lugar
    de solo el resultado final) permite a main.py guardar/inspeccionar cada
    etapa de forma independiente, tal como requiere el proyecto.
    """
    perspective = correct_perspective(image, corners=corners)
    gray = to_grayscale(perspective)
    denoised = denoise(gray)
    contrasted = adjust_contrast(denoised)
    binary = binarize(contrasted)

    return {
        "original": image,
        "perspective": perspective,
        "grayscale": gray,
        "denoised": denoised,
        "contrasted": contrasted,
        "binary": binary,
    }
