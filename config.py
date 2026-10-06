"""
config.py
=========
Configuracion centralizada del pipeline de procesamiento de imagen.

Toda constante que afecte el comportamiento del sistema debe vivir aqui,
en lugar de estar "hardcodeada" dentro de las funciones de cada modulo.
Esto permite iterar rapidamente (ajustar threshold, epsilon, metodo de
extraccion de lineas, etc.) sin tocar el codigo de los algoritmos.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
# Kernel del Gaussian Blur. Debe ser impar (3, 5, 7, ...).
BLUR_KERNEL_SIZE = 5

# Ajuste opcional de contraste/brillo: nueva_img = alpha * img + beta
CONTRAST_ALPHA = 1.0     # 1.0 = sin cambio. >1 aumenta contraste.
CONTRAST_BETA = 0        # 0 = sin cambio. Positivo aclara la imagen.

# Metodo de binarizacion: "otsu", "adaptive" o "fixed"
THRESHOLD_METHOD = "otsu"

# Solo se usa si THRESHOLD_METHOD == "fixed"
THRESHOLD_VALUE = 127

# Solo se usa si THRESHOLD_METHOD == "adaptive"
ADAPTIVE_BLOCK_SIZE = 35   # debe ser impar
ADAPTIVE_C = 10

# Tras binarizar, queremos que el TRAZO (lineas del dibujo) quede en blanco
# (255) sobre fondo negro (0), que es la convencion usada en el resto del
# pipeline. Como el papel es claro y el trazo oscuro, se necesita invertir.
INVERT_BINARY = True

# ---------------------------------------------------------------------------
# Correccion de perspectiva
# ---------------------------------------------------------------------------
# Si se conocen las 4 esquinas de la hoja en la foto original, se pueden
# especificar aqui como lista de 4 tuplas (x, y) en orden:
# [superior-izq, superior-der, inferior-der, inferior-izq]
# Si es None, no se aplica correccion de perspectiva (se usa la imagen tal cual).
PERSPECTIVE_CORNERS = None

# Tamano de salida (en px) del rectangulo corregido cuando SI se aplica
# correccion de perspectiva.
PERSPECTIVE_OUTPUT_SIZE = (1000, 700)  # (width, height)

# ---------------------------------------------------------------------------
# Segmentacion / limpieza morfologica
# ---------------------------------------------------------------------------
# Tamano del kernel usado para operaciones morfologicas (close/open) que
# eliminan ruido pequeno y cierran pequenos huecos en las lineas.
MORPH_KERNEL_SIZE = 3
MORPH_CLOSE_ITERATIONS = 1
MORPH_OPEN_ITERATIONS = 1

# Componentes conexos con area (en px) menor a este valor se consideran
# ruido y se eliminan.
MIN_COMPONENT_AREA = 25

# ---------------------------------------------------------------------------
# Deteccion / extraccion de lineas
# ---------------------------------------------------------------------------
# Estrategia por defecto. Puede sobreescribirse por CLI (--method).
# Opciones: "contours", "edges", "skeleton", "all"
LINE_EXTRACTION_METHOD = "skeleton"

# Parametros para Canny (usados si el metodo es "edges")
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Contours: se ignoran contornos cuya longitud de arco (perimetro, en px)
# sea menor a este valor, para descartar ruido.
MIN_CONTOUR_LENGTH = 15.0

# Strokes: numero minimo de puntos para que un stroke se considere valido.
MIN_STROKE_POINTS = 2

# Skeleton -> strokes: distancia maxima (px) para unir un segmento de
# skeleton suelto con otro cercano al trazar caminos. Ayuda con pequenas
# discontinuidades del skeleton.
SKELETON_GAP_TOLERANCE = 3

# ---------------------------------------------------------------------------
# Simplificacion de trayectorias (Ramer-Douglas-Peucker via cv2.approxPolyDP)
# ---------------------------------------------------------------------------
SIMPLIFICATION_EPSILON = 2.5

# ---------------------------------------------------------------------------
# Coordenadas fisicas
# ---------------------------------------------------------------------------
# Area fisica (en mm) que representa la hoja/imagen procesada. Se usa para
# convertir coordenadas de pixeles a milimetros en la etapa de coordenadas.
PHYSICAL_WIDTH_MM = 200.0
PHYSICAL_HEIGHT_MM = 150.0

# ---------------------------------------------------------------------------
# Visualizacion / salida
# ---------------------------------------------------------------------------
SAVE_INTERMEDIATE_IMAGES = True

# Color (BGR) usado para dibujar strokes sobre la imagen original.
STROKE_COLORS = [
    (0, 0, 255),    # rojo
    (0, 255, 0),    # verde
    (255, 0, 0),    # azul
    (0, 255, 255),  # amarillo
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # naranja
    (128, 0, 255),  # violeta
]

STROKE_LINE_THICKNESS = 2
SIMPLIFIED_POINT_RADIUS = 4
