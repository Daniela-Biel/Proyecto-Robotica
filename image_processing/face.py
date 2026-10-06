"""
image_processing/face.py
========================
Etapas especificas para RETRATOS (caras de personas).

El pipeline "drawing" (threshold global + skeleton) asume un dibujo de
lineas oscuras sobre papel blanco. En una fotografia de una cara eso no
se cumple: un threshold global convierte la foto en manchas (pelo, ropa,
sombras, fondo) y el skeleton de una mancha es su "eje medio", no una
linea que tenga sentido dibujar.

Este modulo resuelve el problema en 5 pasos, todos con OpenCV clasico
(sin redes neuronales, para que corra en una Raspberry Pi):

    1. detect_face()            -> rectangulo de la cara (Haar cascade).
    2. crop_portrait()          -> recorte cabeza + cuello y tamano fijo.
    3. segment_foreground()     -> mascara persona/fondo (GrabCut), para
                                   no dibujar el fondo sin importar su color.
    4. normalize_illumination() -> elimina gradientes de luz/sombra
                                   (flat-field) y ecualiza contraste (CLAHE).
    5. extract_line_mask()      -> lineas finas tipo "boceto" con XDoG
                                   (diferencia de Gaussianas), que responde
                                   a rasgos (ojos, cejas, nariz, boca, pelo)
                                   y no a regiones grandes oscuras.

La mascara de lineas resultante se pasa al mismo skeleton -> strokes del
resto del pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

import config

Rect = Tuple[int, int, int, int]  # (x, y, w, h)


# ---------------------------------------------------------------------------
# 1. Deteccion de cara
# ---------------------------------------------------------------------------
def _load_cascade(name: str) -> Optional[cv2.CascadeClassifier]:
    base = getattr(getattr(cv2, "data", None), "haarcascades", "")
    cascade = cv2.CascadeClassifier(base + name)
    return None if cascade.empty() else cascade


def detect_face(image: np.ndarray) -> Optional[Rect]:
    """Detecta la cara MAS GRANDE de la imagen. Devuelve (x, y, w, h) o None.

    Se ecualiza el histograma antes de detectar para que funcione tambien
    con fotos oscuras o con contraluz. Se prueban varias cascadas (frontal y
    perfil) en orden; la primera que encuentre algo gana.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.equalizeHist(gray)
    min_side = max(30, int(min(gray.shape[:2]) * config.FACE_MIN_SIZE_RATIO))

    for name in config.FACE_CASCADES:
        cascade = _load_cascade(name)
        if cascade is None:
            continue
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            return int(x), int(y), int(w), int(h)
    return None


# ---------------------------------------------------------------------------
# 2. Recorte del retrato
# ---------------------------------------------------------------------------
def crop_portrait(
    image: np.ndarray, face: Optional[Rect]
) -> Tuple[np.ndarray, Optional[Rect], Rect]:
    """Recorta la region cabeza + cuello alrededor de la cara y la escala
    a una altura fija (config.FACE_WORK_HEIGHT).

    Trabajar siempre al mismo tamano hace que los parametros en px (sigma
    del XDoG, areas minimas, etc.) signifiquen lo mismo para una foto de
    celular de 4000 px que para una webcam de 640 px.

    Returns:
        (recorte, cara_en_coordenadas_del_recorte, rect_del_recorte_en_original)
        Si no hay cara se usa la imagen completa y la cara es None.
    """
    img_h, img_w = image.shape[:2]
    if face is None:
        x0, y0, x1, y1 = 0, 0, img_w, img_h
    else:
        fx, fy, fw, fh = face
        mx_l, mx_r, my_t, my_b = config.FACE_CROP_MARGINS
        x0 = max(0, int(fx - mx_l * fw))
        x1 = min(img_w, int(fx + fw + mx_r * fw))
        y0 = max(0, int(fy - my_t * fh))
        y1 = min(img_h, int(fy + fh + my_b * fh))

    crop = image[y0:y1, x0:x1]
    scale = config.FACE_WORK_HEIGHT / crop.shape[0]
    crop = cv2.resize(
        crop,
        (max(1, int(round(crop.shape[1] * scale))), config.FACE_WORK_HEIGHT),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )

    face_in_crop = None
    if face is not None:
        fx, fy, fw, fh = face
        face_in_crop = (
            int((fx - x0) * scale),
            int((fy - y0) * scale),
            int(fw * scale),
            int(fh * scale),
        )
    return crop, face_in_crop, (x0, y0, x1 - x0, y1 - y0)


# ---------------------------------------------------------------------------
# 3. Segmentacion persona / fondo
# ---------------------------------------------------------------------------
def segment_foreground(image: np.ndarray, face: Optional[Rect]) -> np.ndarray:
    """Mascara (255 = persona) usando GrabCut inicializado con la cara.

    GrabCut modela el color del fondo y de la persona con mezclas de
    Gaussianas, por lo que funciona con fondos de cualquier color siempre
    que se distingan de la persona. Inicializacion:
        - Rectangulo central de la cara        -> seguro persona.
        - Elipse cabeza (cara + pelo) y cuello -> probable persona.
        - Bordes superior/izq/der del recorte  -> seguro fondo.
        - Resto                                -> probable fondo.
    Si no hay cara, o config.FACE_REMOVE_BACKGROUND es False, devuelve una
    mascara completa (no se elimina nada).
    """
    h, w = image.shape[:2]
    full = np.full((h, w), 255, np.uint8)
    if face is None or not config.FACE_REMOVE_BACKGROUND:
        return full

    fx, fy, fw, fh = face
    cx, cy = fx + fw // 2, fy + fh // 2

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    # Cabeza + pelo (elipse algo mas grande que la cara) y cuello/hombros.
    cv2.ellipse(mask, (cx, cy - fh // 8), (int(fw * 0.75), int(fh * 0.85)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.rectangle(mask, (cx - fw // 3, cy), (cx + fw // 3, h - 1), cv2.GC_PR_FGD, -1)
    # Nucleo de la cara: seguro persona.
    cv2.rectangle(
        mask,
        (fx + fw // 5, fy + fh // 5),
        (fx + fw - fw // 5, fy + fh - fh // 10),
        cv2.GC_FGD,
        -1,
    )
    # Bordes (excepto el inferior, donde suelen estar los hombros): fondo.
    border = max(2, w // 40)
    mask[:border, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # GrabCut es la etapa mas lenta: se corre a resolucion reducida y la
    # mascara se reescala (el borde se suaviza despues de todos modos).
    s = config.GRABCUT_SCALE if 0 < config.GRABCUT_SCALE < 1 else 1.0
    small_img = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small_img.shape[1], small_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.setRNGSeed(0)  # GrabCut usa k-means aleatorio: resultado reproducible
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            small_img, small_mask, None, bgd_model, fgd_model,
            config.GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return full

    fg_small = np.where((small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.resize(fg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    fg = np.where(fg > 127, 255, 0).astype(np.uint8)

    # Region plausible de una persona: elipse de la cabeza (con pelo) y,
    # por debajo de la barbilla, cuello/hombros. Lo que GrabCut marque como
    # persona fuera de ahi (una ventana o pared de color parecido al pelo)
    # se descarta, para que no aparezca como silueta en el dibujo.
    allowed = np.zeros((h, w), np.uint8)
    cv2.ellipse(allowed, (cx, cy - fh // 8), (int(fw * 0.80), int(fh * 1.0)), 0, 0, 360, 255, -1)
    allowed[min(h, fy + int(fh * 0.85)):, :] = 255
    fg = cv2.bitwise_and(fg, allowed)

    # Limpieza: quedarnos con el componente que contiene la cara, rellenar
    # huecos (ojos/boca a veces quedan como "fondo") y suavizar el borde.
    # El "opening" con un kernel proporcional a la cara elimina tiras
    # delgadas de fondo pegadas a la cabeza (marcos, ventanas).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    k_open = max(15, int(fw * 0.2)) | 1
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    )
    num, labels = cv2.connectedComponents(fg)
    if num > 1:
        label = labels[min(cy, h - 1), min(cx, w - 1)]
        if label == 0:
            areas = np.bincount(labels.ravel())[1:]
            label = int(np.argmax(areas)) + 1
        fg = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, contours, -1, 255, -1)

    # Salvaguarda: si GrabCut fallo (mascara diminuta), no eliminar nada.
    if filled.sum() / 255 < 0.5 * fw * fh:
        return full
    return filled


# ---------------------------------------------------------------------------
# 4. Normalizacion de iluminacion
# ---------------------------------------------------------------------------
def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """Devuelve una imagen en grises con iluminacion uniforme.

    - Flat-field: se divide la imagen entre una version muy desenfocada de
      si misma. Esto elimina sombras suaves y gradientes de luz (luz
      lateral, contraluz, viñeteo) y conserva los detalles finos.
    - CLAHE: ecualizacion de histograma local, recupera contraste en fotos
      oscuras o lavadas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray_f = gray.astype(np.float32) + 1.0

    sigma = config.ILLUMINATION_SIGMA_RATIO * gray.shape[0]
    background = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    flat = gray_f / background
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    return clahe.apply(flat)


# ---------------------------------------------------------------------------
# 5. Lineas tipo boceto (XDoG)
# ---------------------------------------------------------------------------
def extract_line_mask(
    gray: np.ndarray,
    foreground: Optional[np.ndarray] = None,
    face: Optional[Rect] = None,
) -> np.ndarray:
    """Lineas oscuras finas (255 = linea) mediante Diferencia de Gaussianas.

    DoG = G(sigma) - G(k*sigma) es negativa justo en el lado oscuro de un
    borde o sobre una linea oscura delgada (pestañas, contorno de labios,
    fosas nasales). A diferencia de Canny no produce bordes dobles, y a
    diferencia del threshold global no produce manchas.

    El umbral se fija como percentil de la respuesta DENTRO de la persona,
    asi que se adapta solo a fotos con poco o mucho contraste. Si se conoce
    la cara, la zona de rasgos (ojos-nariz-boca) usa su PROPIO umbral con
    un percentil mayor (FACE_FEATURE_BOOST): si no, el pelo y la ropa, que
    tienen mucha textura, se "comen" el presupuesto de lineas y los rasgos
    -lo que hace reconocible un retrato- quedan incompletos.
    """
    g = gray.astype(np.float32) / 255.0
    s = config.XDOG_SIGMA
    g1 = cv2.GaussianBlur(g, (0, 0), s)
    g2 = cv2.GaussianBlur(g, (0, 0), s * config.XDOG_K)
    dog = g1 - config.XDOG_TAU * g2  # < 0 en lineas oscuras

    region = foreground > 0 if foreground is not None else np.ones(g.shape, bool)

    features = np.zeros(g.shape, bool)
    if face is not None:
        fx, fy, fw, fh = face
        ellipse = np.zeros(g.shape, np.uint8)
        cv2.ellipse(
            ellipse, (fx + fw // 2, fy + int(fh * 0.58)),
            (int(fw * 0.40), int(fh * 0.40)), 0, 0, 360, 255, -1,
        )
        features = (ellipse > 0) & region
    rest = region & ~features

    lines = np.zeros(g.shape, bool)
    for zone, pct in (
        (rest, config.LINE_PERCENTILE),
        (features, min(50.0, config.LINE_PERCENTILE * config.FACE_FEATURE_BOOST)),
    ):
        values = dog[zone]
        if values.size == 0:
            continue
        thr = min(np.percentile(values, pct), -config.XDOG_MIN_RESPONSE)
        lines |= (dog < thr) & zone

    lines = lines.astype(np.uint8) * 255
    # Unir guiones cortos de una misma linea (pestañas, cejas) antes del
    # skeleton: menos strokes y menos subidas de lapiz.
    if config.LINE_CLOSE_KERNEL > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.LINE_CLOSE_KERNEL, config.LINE_CLOSE_KERNEL)
        )
        lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, k)
    return lines


def foreground_outline(foreground: np.ndarray) -> np.ndarray:
    """Silueta de la persona como linea de 1 px (para dibujar el contorno
    de cabeza/hombros aunque el contraste con el fondo sea bajo)."""
    outline = np.zeros_like(foreground)
    if foreground.min() == 255:  # no hubo segmentacion
        return outline
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(outline, contours, -1, 255, 1)
    # No dibujar el "corte" del recorte (borde inferior/lateral de la imagen).
    outline[-3:, :] = 0
    outline[:, :3] = 0
    outline[:, -3:] = 0
    return outline
