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

# ===========================================================================
# MODO RETRATO ("face")
# ===========================================================================
# Modo por defecto del pipeline: "face" (fotos de personas) o "drawing"
# (dibujos de lineas sobre papel, comportamiento de la V1).
PIPELINE_MODE = "face"

# --- Deteccion de cara -----------------------------------------------------
# Detector principal: YuNet (OpenCV FaceDetectorYN, modelo de 230 KB en
# models/). Tolera cabeza inclinada y cara parcialmente tapada, y da la
# posicion de ojos, nariz y boca. Si no esta disponible se usa Haar.
YUNET_MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_SCORE_THRESHOLD = 0.7
# Si no se detecta ninguna cara: True = error (no dibujar la escena entera);
# False = usar la imagen completa (se puede forzar con --allow-no-face).
FACE_REQUIRED = True
# Respaldo: cascadas Haar incluidas en opencv-python.
FACE_CASCADES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
]
# Tamano minimo de cara, como fraccion del lado menor de la imagen.
FACE_MIN_SIZE_RATIO = 0.08

# Margenes del recorte respecto al tamano de la cara:
# (izquierda, derecha, arriba, abajo) en multiplos de ancho/alto de cara.
FACE_CROP_MARGINS = (0.55, 0.55, 0.65, 0.55)

# Altura (px) a la que se reescala el recorte. Todos los parametros en px
# del modo retrato estan pensados para esta altura.
FACE_WORK_HEIGHT = 600

# --- Fondo -----------------------------------------------------------------
FACE_REMOVE_BACKGROUND = True
GRABCUT_ITERATIONS = 5
# Dibujar la silueta (contorno cabeza/hombros) obtenida de la segmentacion.
FACE_DRAW_OUTLINE = True

# --- Iluminacion -----------------------------------------------------------
# Sigma del blur de "flat-field" como fraccion de la altura de trabajo.
ILLUMINATION_SIGMA_RATIO = 0.06
CLAHE_CLIP_LIMIT = 2.0

# --- Lineas (XDoG) ---------------------------------------------------------
XDOG_SIGMA = 1.6          # escala de los rasgos (px a FACE_WORK_HEIGHT)
XDOG_K = 1.6              # razon entre las dos Gaussianas
XDOG_TAU = 0.98           # <1 favorece lineas oscuras
# Porcentaje de pixeles de la persona que se marcan como linea. Subir =
# mas detalle (y mas tiempo de dibujo); bajar = retrato mas minimalista.
LINE_PERCENTILE = 7.0
# Respuesta minima para considerar linea (evita dibujar ruido en zonas planas).
XDOG_MIN_RESPONSE = 0.01
# Multiplicador de LINE_PERCENTILE dentro de la zona de rasgos (ojos, nariz,
# boca). >1 = mas detalle en la cara que en pelo/ropa.
FACE_FEATURE_BOOST = 1.8
# Igual, para una zona chica alrededor de cada ojo (requiere YuNet). Mas alto
# porque con lentes el armazon opaca al ojo.
FACE_EYE_BOOST = 3.0
# Kernel (px) del "closing" que une guiones cortos de una misma linea. 0 = off.
LINE_CLOSE_KERNEL = 5
# Huecos dentro de las lineas con area (px) <= esto se rellenan (reflejos
# en ojos/lentes que el skeleton convertiria en "burbujas"). 0 = off.
FACE_FILL_HOLES_AREA = 40
# Componentes de linea con area (px) menor a esto se descartan (motas).
FACE_MIN_LINE_AREA = 15
# GrabCut se ejecuta a esta fraccion de la resolucion de trabajo (mas rapido
# en Raspberry Pi; la mascara se reescala despues).
GRABCUT_SCALE = 0.5

# ===========================================================================
# TRAYECTORIA PARA EL ROBOT (etapa robot_path, comun a ambos modos)
# ===========================================================================
# Area de dibujo = PHYSICAL_WIDTH_MM x PHYSICAL_HEIGHT_MM. La imagen se
# escala de forma UNIFORME (sin deformar) y se centra dentro de ella.
DRAWING_MARGIN_MM = 5.0
# Posicion (mm) de la esquina inferior-izquierda del area de dibujo en el
# marco del robot. Ajustar para que las coordenadas salgan ya en ese marco.
DRAWING_ORIGIN_MM = (0.0, 0.0)
# En la imagen Y crece hacia abajo; en el robot normalmente hacia arriba.
FLIP_Y = True

# Suavizado del stroke en px (media movil). 0/1 desactiva.
SMOOTH_WINDOW_PX = 5
# Strokes cuyos extremos esten a <= esta distancia (px) se unen en uno solo.
JOIN_GAP_PX = 3.0
# Union de lineas "en guiones": si el final de un stroke apunta al inicio de
# otro (hueco <= LINK_GAP_PX y desvio <= LINK_MAX_ANGLE_DEG) se unen,
# dibujando el hueco. Menos subidas de lapiz. LINK_GAP_PX = 0 desactiva.
LINK_GAP_PX = 12.0
LINK_MAX_ANGLE_DEG = 30.0
# Strokes mas cortos que esto (mm) se descartan (ruido / puntos sueltos).
MIN_STROKE_LENGTH_MM = 1.5
# (modo face) Fuera de la zona de rasgos (pelo, mejillas, ropa) los trazos
# sueltos cortos son casi siempre textura: se exige una longitud mayor.
MIN_STROKE_LENGTH_OUTSIDE_MM = 4.0
# Tolerancia RDP en mm (se aplica antes de densificar).
ROBOT_SIMPLIFY_EPSILON_MM = 0.15
# Longitud maxima de un stroke (mm). Strokes mas largos se parten. 0 = sin limite.
MAX_STROKE_LENGTH_MM = 30.0
# Distancia maxima (mm) entre dos waypoints consecutivos (para IK). 0 = no densificar.
MAX_SEGMENT_MM = 1.0

# Solo para estimar el tiempo de dibujo en el resumen.
PEN_DRAW_SPEED_MM_S = 20.0
PEN_TRAVEL_SPEED_MM_S = 50.0
PEN_LIFT_TIME_S = 0.6
# Resolucion de las imagenes de previsualizacion en mm.
PREVIEW_PX_PER_MM = 4
