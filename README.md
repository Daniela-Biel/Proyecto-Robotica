# Robot Drawing — Sistema de procesamiento de imágenes (V1)

Sistema de visión por computadora que convierte la fotografía de un dibujo
hecho a mano en trayectorias 2D (`strokes`) que, en una fase futura, un
robot podría usar para reproducir el dibujo con un lápiz.

**Esta V1 NO controla ningún robot.** Su único objetivo es resolver:

> "¿Cómo convierto las líneas de esta imagen en trayectorias 2D?"

No intenta responder "¿qué objeto representa este dibujo?" — el sistema es
puramente geométrico, no semántico.

---

## 1. Objetivo del proyecto

Dada una fotografía de un dibujo simple (figuras geométricas, casas,
estrellas, bocetos con líneas) sobre una hoja blanca, el sistema:

1. Preprocesa la imagen (grises, blur, contraste, binarización).
2. Segmenta y limpia el trazo del ruido de fondo.
3. Detecta líneas usando una de tres estrategias intercambiables
   (contornos, bordes o esqueletización).
4. Extrae trayectorias continuas (`strokes`).
5. Simplifica cada trayectoria a un número reducido de puntos relevantes.
6. Convierte a coordenadas 2D (píxeles y, opcionalmente, milímetros).
7. Exporta el resultado a JSON/CSV y genera imágenes de cada etapa.

---

## 2. Arquitectura

```text
robot_drawing/
│
├── main.py                      # CLI: orquesta el pipeline paso a paso
├── config.py                    # TODOS los parámetros configurables
├── requirements.txt
├── README.md
│
├── image_processing/
│   ├── preprocess.py            # carga, grises, blur, contraste, threshold, perspectiva
│   ├── segmentation.py          # limpieza morfológica de la máscara binaria
│   ├── skeleton.py              # skeletonization + skeleton -> strokes
│   ├── strokes.py               # despachador de estrategias (contours/edges/skeleton)
│   ├── simplification.py        # Ramer-Douglas-Peucker (cv2.approxPolyDP)
│   └── coordinates.py           # px -> mm, export JSON/CSV
│
├── visualization/
│   └── visualize.py             # guarda imágenes de cada etapa
│
├── input/                       # imágenes de entrada
└── output/                      # resultados generados (imágenes, JSON, CSV)
```

Cada módulo es independiente: recibe y devuelve arrays de NumPy o
estructuras de datos simples (listas de puntos, diccionarios), sin
depender de los demás módulos más que lo estrictamente necesario. Esto
permite sustituir una estrategia (por ejemplo, cambiar cómo se hace la
simplificación) sin reescribir el resto del proyecto.

`main.py` **no** oculta el pipeline detrás de una única función
`process_image()`: cada etapa se invoca explícitamente para poder
inspeccionar o depurar cualquier paso de forma aislada.

---

## 3. Instalación

Requiere Python 3.9+.

```bash
cd robot_drawing
python3 -m venv venv
source venv/bin/activate       # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dependencias

- `opencv-python-headless` — procesamiento de imagen (blur, threshold,
  contornos, Canny, approxPolyDP, morfología).
- `numpy` — manejo de arrays.
- `scikit-image` — skeletonization (`skimage.morphology.skeletonize`).
- `matplotlib` — disponible para análisis/visualización adicional (no es
  estrictamente necesaria para el pipeline actual, pero se incluye para
  facilitar exploración interactiva de resultados).

---

## 5. Cómo ejecutar

Coloca tu imagen en `input/` (o usa cualquier ruta) y ejecuta:

```bash
python main.py --input input/dibujo.png
```

Elegir estrategia de extracción de líneas:

```bash
python main.py --input input/dibujo.png --method skeleton
python main.py --input input/dibujo.png --method contours
python main.py --input input/dibujo.png --method edges
```

Comparar las tres estrategias en una sola ejecución (genera
`output/contours/`, `output/edges/`, `output/skeleton/`):

```bash
python main.py --input input/dibujo.png --method all
```

Ajustar simplificación y threshold:

```bash
python main.py --input input/dibujo.png --epsilon 3.0 --threshold 140 --threshold-method fixed
```

Otras opciones útiles:

```text
--output-dir DIR             Directorio de salida (default: output/)
--physical-width-mm N        Ancho físico representado por la imagen (mm)
--physical-height-mm N       Alto físico representado por la imagen (mm)
--no-intermediate            No guardar imágenes intermedias (solo el resultado final)
```

### Ejemplo real (incluido en este repo)

Se incluye una imagen sintética de prueba en `input/dibujo_casa.png`
(una casa con puerta, ventana circular y una estrella) para validar la
instalación:

```bash
python main.py --input input/dibujo_casa.png --method all
```

Deberías obtener una salida de consola similar a:

```text
====================================
RESULTADO
====================================

Imagen: dibujo_casa.png

Dimensiones:
1000 x 700 px

Método:
contours

Strokes detectados:
13

Puntos antes de simplificación:
5040

Puntos después de simplificación:
102

Reducción:
98.0%

Longitud total aproximada de trayectorias:
5421.8 px

Archivos de salida en: output/contours
====================================
```

---

## 6. Estructura de carpetas de salida

Por cada ejecución (o por cada método, en modo `--method all`) se genera:

```text
output/
├── 01_original.png            # imagen de entrada, sin modificar
├── 02_perspective.png         # tras corrección de perspectiva (o igual a 01 si no aplica)
├── 03_grayscale.png           # escala de grises
├── 04_threshold.png           # máscara binaria (trazo = blanco)
├── 05_cleaned.png             # máscara tras limpieza morfológica
├── 06_edges.png                # solo si method == "edges"
├── 07_skeleton.png             # solo si method == "skeleton"
├── 08_strokes.png             # strokes crudos (antes de simplificar) sobre la imagen original
├── 09_simplified.png          # puntos tras simplificación, sobre la imagen original
├── 10_final_trajectory.png    # imagen final: original + trayectorias
├── trajectories.json          # trayectorias en píxeles
├── trajectories_mm.json       # trayectorias convertidas a milímetros
└── trajectories.csv           # stroke_id, point_id, x, y (en píxeles)
```

---

## 7. Qué hace cada etapa

| Etapa | Módulo | Descripción |
|---|---|---|
| Preprocesamiento | `image_processing/preprocess.py` | Carga la imagen, corrige perspectiva (si hay esquinas configuradas), convierte a grises, reduce ruido (Gaussian Blur), ajusta contraste y binariza. |
| Segmentación | `image_processing/segmentation.py` | Cierra pequeños huecos y elimina ruido puntual mediante operaciones morfológicas y filtrado de componentes conexos pequeños. |
| Detección de líneas + extracción de strokes | `image_processing/strokes.py`, `image_processing/skeleton.py` | Convierte la máscara limpia en una lista de trayectorias (`strokes`), usando una de tres estrategias intercambiables. |
| Simplificación | `image_processing/simplification.py` | Reduce el número de puntos de cada stroke con Ramer-Douglas-Peucker (`cv2.approxPolyDP`), controlado por `SIMPLIFICATION_EPSILON`. |
| Coordenadas | `image_processing/coordinates.py` | Mantiene las coordenadas en píxeles y provee una conversión explícita e independiente a milímetros. Exporta JSON/CSV. |
| Visualización | `visualization/visualize.py` | Guarda una imagen por cada etapa relevante del pipeline. |

### Las tres estrategias de detección de líneas

- **`contours`** (`cv2.findContours`): rápido y robusto para formas
  cerradas o con relleno de línea gruesa. Tiende a generar un contorno
  "doble" (por ambos lados del trazo) en líneas abiertas dibujadas a
  mano, ya que trabaja sobre regiones, no sobre líneas de 1 px.
- **`edges`** (`cv2.Canny` + contornos sobre el mapa de bordes): útil
  cuando el contraste es irregular. Suele generar más strokes y más
  puntos que `contours`, porque Canny puede producir bordes discontinuos.
- **`skeleton`** (`skimage.morphology.skeletonize`): reduce cada trazo a
  su eje central de 1 px de ancho — evita la duplicación de línea que
  ocurre con `contours`, pero es más sensible a fragmentarse en cruces
  de líneas (ver limitaciones).

No hay una estrategia "correcta" universal: el diseño permite ejecutar
`--method all` y comparar visualmente los resultados en `output/*/`.

---

## 8. Cómo cambiar de estrategia

Dos formas:

1. **Por CLI** (recomendado para experimentar): `--method contours|edges|skeleton|all`.
2. **Por defecto en `config.py`**: cambiar `LINE_EXTRACTION_METHOD`.

Para agregar una nueva estrategia en el futuro, basta con:
1. Implementar una función `extract_strokes_from_X(...)` en `strokes.py`
   (o un módulo nuevo) que devuelva `List[List[Tuple[int, int]]]`.
2. Agregar el nuevo caso al `if/elif` de `extract_strokes()` en
   `strokes.py`.

No es necesario tocar `main.py`, `simplification.py` ni `coordinates.py`.

---

## 9. Cómo interpretar los archivos de salida

- **Imágenes `0X_*.png`**: revisar en orden. Si el resultado final se ve
  mal, revisa las imágenes intermedias en orden para ubicar en qué etapa
  se degrada (por ejemplo: si `04_threshold.png` ya se ve mal, el
  problema está en el threshold/iluminación, no en la detección de
  líneas).
- **`trajectories.json`**: estructura principal, en píxeles. Cada
  `stroke` es una trayectoria continua (el robot "baja el lápiz" al
  inicio y lo "sube" al final). Preparada para agregar en el futuro `z`,
  `pen_down`/`pen_up`, velocidad, etc.
- **`trajectories_mm.json`**: mismas trayectorias convertidas a
  milímetros usando `PHYSICAL_WIDTH_MM`/`PHYSICAL_HEIGHT_MM` (o los
  argumentos `--physical-width-mm`/`--physical-height-mm`).
- **`trajectories.csv`**: mismos datos en formato tabular
  (`stroke_id,point_id,x,y`), útil para inspección rápida en Excel/Sheets
  o para depuración con `pandas`.

---

## 10. Limitaciones actuales (V1)

- **Sin detección automática de esquinas del papel**: la corrección de
  perspectiva solo se aplica si se proveen manualmente las 4 esquinas en
  `config.PERSPECTIVE_CORNERS`. Si no se proveen, se omite este paso.
- **Cruces de líneas**: ninguna de las tres estrategias resuelve
  perfectamente una "X" o cruces densos de líneas. El resultado puede
  fragmentarse en más strokes de los "estéticamente ideales", aunque
  sigue siendo geométricamente válido.
- **Orden de los strokes**: es simplemente el orden en que se descubren
  durante el recorrido/escaneo. No hay optimización de ruta (tipo
  "traveling salesman") para minimizar desplazamientos del lápiz — esto
  quedará para una fase posterior.
- **Conversión px→mm simplificada**: asume una escala lineal
  independiente en X e Y a partir del tamaño total de la imagen; no
  corrige distorsiones de lente ni variaciones de escala dentro de la
  misma imagen.
- **No hay reconocimiento semántico**: el sistema no sabe que un conjunto
  de líneas forma una "casa"; solo extrae geometría. Esto es
  intencional (ver principio fundamental del proyecto).
- **Solo trazos oscuros sobre fondo claro**: el pipeline asume alto
  contraste trazo/papel. Colores complejos, acuarelas o sombreados
  extensos no están soportados en esta V1.

---

## 11. Próximos pasos recomendados

1. **Detección automática de las esquinas del papel** (por ejemplo,
   buscando el contorno de mayor área con 4 vértices) para automatizar
   `correct_perspective`.
2. **Mejor manejo de cruces en `skeleton.py`**: usar un análisis de
   grafo más sofisticado (por ejemplo, `networkx`) para decidir cómo
   continuar un stroke a través de una unión, en vez de la heurística de
   "vecino más alineado con la dirección de avance".
3. **Optimización de orden de strokes** (nearest-neighbor o similar)
   para minimizar el recorrido total del lápiz entre trayectorias.
4. **Calibración de cámara** más precisa para la conversión píxeles→mm
   (usar un patrón de calibración conocido en vez de asumir el tamaño de
   la hoja).
5. **Pruebas con fotografías reales** (no solo imágenes sintéticas) para
   ajustar `BLUR_KERNEL_SIZE`, `THRESHOLD_METHOD`, `MIN_COMPONENT_AREA` y
   `SIMPLIFICATION_EPSILON` a condiciones reales de iluminación y cámara.
6. **Fase 2 del proyecto** (fuera de alcance de esta V1): integrar
   control real del robot (motores, G-code o comunicación serial),
   cinemática, y el manejo de `pen_down`/`pen_up` ya previsto en la
   estructura del JSON.
