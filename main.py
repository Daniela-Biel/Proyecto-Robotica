"""
main.py
=======
Punto de entrada del sistema. Orquesta el pipeline completo, etapa por
etapa, guardando resultados intermedios y el resultado final.

Deliberadamente NO se encapsula todo en una funcion tipo process_image()
que oculte los pasos intermedios: cada etapa se invoca explicitamente
aqui para poder inspeccionar/depurar el pipeline con facilidad.

Dos modos:
    - face    (por defecto): fotografia de una persona -> retrato de lineas.
    - drawing: fotografia de un dibujo de lineas sobre papel (V1).

Ambos terminan en la misma etapa "robot_path", que produce la trayectoria
en mm lista para la cinematica inversa (robot_waypoints.csv).

Uso:
    python main.py --input input/foto.jpg
    python main.py --input input/foto.jpg --detail 9 --max-stroke-mm 20
    python main.py --input input/dibujo.png --mode drawing --method skeleton
    python main.py --input input/dibujo.png --mode drawing --method all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from image_processing import (
    coordinates,
    face,
    preprocess,
    robot_path,
    segmentation,
    simplification,
    strokes,
)
from image_processing import skeleton as skeleton_module
from visualization import visualize


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte una fotografia (retrato o dibujo) en trayectorias "
        "2D en mm para un robot dibujante."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Ruta a la imagen de entrada."
    )
    parser.add_argument(
        "--mode",
        choices=["face", "drawing"],
        default=config.PIPELINE_MODE,
        help="'face' para fotos de personas, 'drawing' para dibujos de lineas.",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["contours", "edges", "skeleton", "all"],
        default=config.LINE_EXTRACTION_METHOD,
        help="(modo drawing) Estrategia de extraccion de lineas. 'all' compara las tres.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config.SIMPLIFICATION_EPSILON,
        help="(modo drawing) Tolerancia de simplificacion en px (Ramer-Douglas-Peucker).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config.THRESHOLD_VALUE,
        help="(modo drawing) Valor de threshold fijo (solo si --threshold-method fixed).",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["otsu", "adaptive", "fixed"],
        default=config.THRESHOLD_METHOD,
        help="(modo drawing) Metodo de binarizacion.",
    )
    parser.add_argument(
        "--detail",
        type=float,
        default=config.LINE_PERCENTILE,
        help="(modo face) %% de pixeles de la persona marcados como linea. "
        "Mas alto = mas detalle y mas tiempo de dibujo (tipico 4-12).",
    )
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="(modo face) No eliminar el fondo.",
    )
    parser.add_argument(
        "--allow-no-face",
        action="store_true",
        help="(modo face) Si no se detecta cara, dibujar la imagen completa en vez de dar error.",
    )
    parser.add_argument(
        "--no-outline",
        action="store_true",
        help="(modo face) No dibujar la silueta cabeza/hombros.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(config.OUTPUT_DIR),
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--physical-width-mm",
        type=float,
        default=config.PHYSICAL_WIDTH_MM,
        help="Ancho (mm) del area de dibujo.",
    )
    parser.add_argument(
        "--physical-height-mm",
        type=float,
        default=config.PHYSICAL_HEIGHT_MM,
        help="Alto (mm) del area de dibujo.",
    )
    parser.add_argument(
        "--max-stroke-mm",
        type=float,
        default=config.MAX_STROKE_LENGTH_MM,
        help="Longitud maxima de un stroke en mm (0 = sin limite).",
    )
    parser.add_argument(
        "--max-segment-mm",
        type=float,
        default=config.MAX_SEGMENT_MM,
        help="Distancia maxima entre waypoints consecutivos en mm (0 = no densificar).",
    )
    parser.add_argument(
        "--min-stroke-mm",
        type=float,
        default=config.MIN_STROKE_LENGTH_MM,
        help="Strokes mas cortos que esto (mm) se descartan.",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="No guardar imagenes intermedias, solo el resultado final.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Modo "face": foto de una persona -> strokes en px
# ---------------------------------------------------------------------------
def run_face_lines(image_path: Path, output_dir: Path, args, save_intermediate: bool) -> dict:
    """Etapas 1-4 del modo retrato. Devuelve strokes en px del RECORTE."""
    original = preprocess.load_image(image_path)
    visualize.save_step_image(original, output_dir / "01_original.png")

    # 1. Deteccion de cara + recorte a tamano de trabajo fijo
    detected = face.detect_face(original)
    crop, face_in_crop, crop_rect = face.crop_portrait(original, detected)
    if save_intermediate:
        visualize.save_step_image(
            visualize.draw_face_detection(original, detected, crop_rect),
            output_dir / "02_face_detection.png",
        )
    if detected is None:
        if config.FACE_REQUIRED and not args.allow_no_face:
            # Sin cara, el resultado seria la escena completa (lamparas,
            # muebles...): mejor no generar una trayectoria que no sirve.
            raise ValueError(
                f"No se detecto ninguna cara en {image_path.name}. Revisa "
                "02_face_detection.png; usa --allow-no-face para dibujar la imagen completa."
            )
        print("AVISO: no se detecto ninguna cara; se usa la imagen completa "
              "y no se elimina el fondo.", file=sys.stderr)
    if save_intermediate:
        visualize.save_step_image(
            visualize.draw_face_zones(crop, face_in_crop), output_dir / "03_crop.png"
        )

    # 2. Persona vs fondo
    if args.keep_background:
        config.FACE_REMOVE_BACKGROUND = False
    foreground = face.segment_foreground(crop, face_in_crop)
    if save_intermediate:
        visualize.save_step_image(
            cv2.bitwise_and(crop, crop, mask=foreground), output_dir / "04_foreground.png"
        )

    # 3. Iluminacion uniforme
    normalized = face.normalize_illumination(crop)
    if save_intermediate:
        visualize.save_step_image(normalized, output_dir / "05_illumination.png")

    # 4. Lineas (XDoG) + silueta + limpieza
    config.LINE_PERCENTILE = args.detail
    lines = face.extract_line_mask(normalized, foreground, face_in_crop)
    if config.FACE_DRAW_OUTLINE and not args.no_outline:
        lines = cv2.bitwise_or(lines, face.foreground_outline(foreground))
    lines = face.fill_small_holes(lines, config.FACE_FILL_HOLES_AREA)
    lines = segmentation.remove_small_components(lines, config.FACE_MIN_LINE_AREA)
    if save_intermediate:
        visualize.save_step_image(lines, output_dir / "06_lines.png")

    skel = skeleton_module.skeletonize_image(lines)
    raw_strokes = skeleton_module.skeleton_to_strokes(skel)
    if save_intermediate:
        visualize.save_step_image(skel, output_dir / "07_skeleton.png")
        visualize.save_step_image(
            visualize.draw_strokes(crop, raw_strokes), output_dir / "08_strokes.png"
        )

    features, eyes = face.feature_zones(face_in_crop, crop.shape[:2])
    detail_zone = (features | eyes) if face_in_crop is not None else None

    return {
        "strokes": raw_strokes,
        "base_image": crop,
        "face_detected": detected is not None,
        "detail_zone": detail_zone,
    }


# ---------------------------------------------------------------------------
# Modo "drawing": foto de un dibujo de lineas -> strokes en px (V1)
# ---------------------------------------------------------------------------
def run_drawing_lines(
    image_path: Path, method: str, output_dir: Path, args, save_intermediate: bool
) -> dict:
    original = preprocess.load_image(image_path)

    pre = preprocess.preprocess_pipeline(original)
    visualize.save_step_image(pre["original"], output_dir / "01_original.png")
    if save_intermediate:
        visualize.save_step_image(pre["perspective"], output_dir / "02_perspective.png")
        visualize.save_step_image(pre["grayscale"], output_dir / "03_grayscale.png")

    binary = preprocess.binarize(
        pre["contrasted"], method=args.threshold_method, threshold_value=args.threshold
    )
    if save_intermediate:
        visualize.save_step_image(binary, output_dir / "04_threshold.png")

    cleaned = segmentation.clean_binary_mask(binary)
    if save_intermediate:
        visualize.save_step_image(cleaned, output_dir / "05_cleaned.png")

    result = strokes.extract_strokes(cleaned, pre["grayscale"], method=method)
    raw_strokes = result["strokes"]
    extra = result["extra"]
    if save_intermediate and extra is not None:
        if method == "edges":
            visualize.save_step_image(extra, output_dir / "06_edges.png")
        elif method == "skeleton":
            visualize.save_step_image(extra, output_dir / "07_skeleton.png")
    if save_intermediate:
        visualize.save_step_image(
            visualize.draw_strokes(pre["original"], raw_strokes), output_dir / "08_strokes.png"
        )

    # Salidas en px de la V1 (se conservan por compatibilidad).
    simplified = simplification.simplify_strokes(raw_strokes, epsilon=args.epsilon)
    if save_intermediate:
        visualize.save_step_image(
            visualize.draw_simplified_points(pre["original"], simplified),
            output_dir / "09_simplified.png",
        )
    h, w = original.shape[:2]
    coordinates.save_json(
        coordinates.strokes_to_dict(simplified, image_path.name, w, h),
        output_dir / "trajectories.json",
    )
    coordinates.save_csv(simplified, output_dir / "trajectories.csv")

    return {"strokes": raw_strokes, "base_image": pre["perspective"], "face_detected": None}


# ---------------------------------------------------------------------------
# Etapa comun: strokes en px -> trayectoria del robot en mm
# ---------------------------------------------------------------------------
def build_robot_path(raw_strokes, width_px: int, height_px: int, args, detail_zone=None) -> dict:
    """detail_zone: mascara (px) de ojos/nariz/boca. Si se da, fuera de ella
    se exige MIN_STROKE_LENGTH_OUTSIDE_MM en vez de --min-stroke-mm."""
    # En px: unir guiones, ordenar/unir extremos, suavizar
    linked = robot_path.link_collinear_strokes(
        raw_strokes, config.LINK_GAP_PX, config.LINK_MAX_ANGLE_DEG
    )
    joined = robot_path.order_and_join_strokes(linked, config.JOIN_GAP_PX)
    joined = [robot_path.smooth_stroke(s, config.SMOOTH_WINDOW_PX) for s in joined]

    in_detail = [True] * len(joined)
    if detail_zone is not None:
        h, w = detail_zone.shape
        in_detail = []
        for s in joined:
            pts = np.clip(np.round(np.asarray(s)).astype(int), 0, [w - 1, h - 1])
            in_detail.append(detail_zone[pts[:, 1], pts[:, 0]].mean() > 0.5)

    # px -> mm (escala uniforme, centrado, Y hacia arriba)
    area = (args.physical_width_mm, args.physical_height_mm)
    mm, scale = robot_path.fit_to_area(
        joined, width_px, height_px, area[0], area[1],
        margin_mm=config.DRAWING_MARGIN_MM,
        origin_mm=config.DRAWING_ORIGIN_MM,
        flip_y=config.FLIP_Y,
    )
    min_outside = max(args.min_stroke_mm, config.MIN_STROKE_LENGTH_OUTSIDE_MM)
    mm = [
        s for s, inside in zip(mm, in_detail)
        if robot_path.stroke_length(s) >= (args.min_stroke_mm if inside else min_outside)
    ]
    mm = robot_path.simplify_strokes_mm(mm, config.ROBOT_SIMPLIFY_EPSILON_MM)
    # Reordenar despues de filtrar (sin unir: los cortes de abajo son intencionales)
    mm = robot_path.order_and_join_strokes(mm, join_gap=0.0, start=config.DRAWING_ORIGIN_MM)
    mm = robot_path.split_long_strokes(mm, args.max_stroke_mm)
    mm = robot_path.densify_strokes(mm, args.max_segment_mm)
    stats = robot_path.path_stats(mm, start=config.DRAWING_ORIGIN_MM)
    stats["mm_per_px"] = scale
    return {"strokes_mm": mm, "stats": stats, "area": area}


def run(image_path: Path, mode: str, method: str, output_dir: Path, args) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_intermediate = not args.no_intermediate

    if mode == "face":
        lines = run_face_lines(image_path, output_dir, args, save_intermediate)
    else:
        lines = run_drawing_lines(image_path, method, output_dir, args, save_intermediate)

    base = lines["base_image"]
    height_px, width_px = base.shape[:2]
    robot = build_robot_path(
        lines["strokes"], width_px, height_px, args, lines.get("detail_zone")
    )
    strokes_mm, stats = robot["strokes_mm"], robot["stats"]

    meta = {
        "image": image_path.name,
        "mode": mode,
        "drawing_area_mm": list(robot["area"]),
        "origin_mm": list(config.DRAWING_ORIGIN_MM),
        "y_axis": "up" if config.FLIP_Y else "down",
        "max_segment_mm": args.max_segment_mm,
        "max_stroke_length_mm": args.max_stroke_mm,
        "stats": stats,
    }
    robot_path.save_robot_json(strokes_mm, meta, output_dir / "trajectories_mm.json")
    robot_path.save_waypoints_csv(strokes_mm, output_dir / "robot_waypoints.csv")

    origin = config.DRAWING_ORIGIN_MM
    if save_intermediate:
        visualize.save_step_image(
            visualize.draw_robot_path(strokes_mm, robot["area"], origin),
            output_dir / "09_robot_path.png",
        )
    visualize.save_step_image(
        visualize.draw_robot_path(
            strokes_mm, robot["area"], origin, show_travel=False, color_strokes=False
        ),
        output_dir / "10_final_trajectory.png",
    )

    stats.update(
        {
            "mode": mode if mode == "face" else f"drawing/{method}",
            "raw_strokes": len(lines["strokes"]),
            "face_detected": lines["face_detected"],
            "output_dir": str(output_dir),
        }
    )
    return stats


def print_summary(image_name: str, m: dict) -> None:
    print("=" * 44)
    print(f"RESULTADO  ({image_name}, modo {m['mode']})")
    print("=" * 44)
    if m["face_detected"] is not None:
        print(f"Cara detectada:            {'si' if m['face_detected'] else 'NO'}")
    print(f"Strokes crudos (skeleton): {m['raw_strokes']}")
    print(f"Strokes finales:           {m['num_strokes']}")
    print(f"Waypoints:                 {m['num_waypoints']}")
    print(f"Longitud dibujando:        {m['draw_length_mm']:.0f} mm")
    print(f"Desplazamiento lapiz arriba: {m['travel_length_mm']:.0f} mm")
    print(f"Stroke mas largo:          {m['max_stroke_length_mm']:.1f} mm "
          f"(promedio {m['mean_stroke_length_mm']:.1f} mm)")
    print(f"Segmento mas largo:        {m['max_segment_mm']:.2f} mm")
    x0, y0, x1, y1 = m["bbox_mm"]
    print(f"Caja envolvente:           x [{x0:.1f}, {x1:.1f}]  y [{y0:.1f}, {y1:.1f}] mm")
    print(f"Escala:                    {m['mm_per_px']:.3f} mm/px")
    print(f"Tiempo estimado:           {m['estimated_time_s'] / 60:.1f} min")
    print(f"Archivos de salida en:     {m['output_dir']}")
    print("=" * 44)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    image_path = Path(args.input)
    base_output_dir = Path(args.output_dir)

    if args.mode == "drawing" and args.method == "all":
        # Modo comparacion: una subcarpeta por estrategia.
        jobs = [(m, base_output_dir / m) for m in ("contours", "edges", "skeleton")]
    else:
        jobs = [(args.method, base_output_dir)]

    try:
        for method, out_dir in jobs:
            start = time.time()
            metrics = run(image_path, args.mode, method, out_dir, args)
            print_summary(image_path.name, metrics)
            print(f"(tiempo: {time.time() - start:.2f}s)\n")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
