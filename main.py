"""
main.py
=======
Punto de entrada del sistema. Orquesta el pipeline completo, etapa por
etapa, guardando resultados intermedios y el resultado final.

Deliberadamente NO se encapsula todo en una funcion tipo process_image()
que oculte los pasos intermedios: cada etapa se invoca explicitamente
aqui para poder inspeccionar/depurar el pipeline con facilidad.

Uso:
    python main.py --input input/dibujo.png
    python main.py --input input/dibujo.png --method skeleton
    python main.py --input input/dibujo.png --method contours
    python main.py --input input/dibujo.png --method edges
    python main.py --input input/dibujo.png --method all
    python main.py --input input/dibujo.png --epsilon 3.0 --threshold 140
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import config
from image_processing import coordinates, preprocess, segmentation, simplification, strokes
from visualization import visualize


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte una fotografia de un dibujo en trayectorias 2D "
        "(sistema de vision para robot dibujante - V1, sin control de robot)."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Ruta a la imagen de entrada."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["contours", "edges", "skeleton", "all"],
        default=config.LINE_EXTRACTION_METHOD,
        help="Estrategia de extraccion de lineas. 'all' ejecuta y compara las tres.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config.SIMPLIFICATION_EPSILON,
        help="Tolerancia de simplificacion (Ramer-Douglas-Peucker).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config.THRESHOLD_VALUE,
        help="Valor de threshold fijo (solo aplica si THRESHOLD_METHOD='fixed').",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["otsu", "adaptive", "fixed"],
        default=config.THRESHOLD_METHOD,
        help="Metodo de binarizacion.",
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
        help="Ancho fisico (mm) representado por la imagen.",
    )
    parser.add_argument(
        "--physical-height-mm",
        type=float,
        default=config.PHYSICAL_HEIGHT_MM,
        help="Alto fisico (mm) representado por la imagen.",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="No guardar imagenes intermedias, solo el resultado final.",
    )
    return parser.parse_args(argv)


def run_pipeline_for_method(
    image_path: Path,
    method: str,
    output_dir: Path,
    epsilon: float,
    threshold_method: str,
    threshold_value: int,
    physical_width_mm: float,
    physical_height_mm: float,
    save_intermediate: bool,
) -> dict:
    """Ejecuta el pipeline completo para UNA estrategia de extraccion de
    lineas y guarda todos los archivos de salida correspondientes.

    Devuelve un diccionario con metricas para poder imprimir el resumen.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Carga + preprocesamiento
    # -------------------------------------------------------------
    original = preprocess.load_image(image_path)
    height_px, width_px = original.shape[:2]

    pre = preprocess.preprocess_pipeline(original)
    visualize.save_step_image(pre["original"], output_dir / "01_original.png")
    if save_intermediate:
        visualize.save_step_image(pre["perspective"], output_dir / "02_perspective.png")
        visualize.save_step_image(pre["grayscale"], output_dir / "03_grayscale.png")

    binary = preprocess.binarize(
        pre["contrasted"], method=threshold_method, threshold_value=threshold_value
    )
    if save_intermediate:
        visualize.save_step_image(binary, output_dir / "04_threshold.png")

    # -------------------------------------------------------------
    # 2. Segmentacion (limpieza morfologica)
    # -------------------------------------------------------------
    cleaned = segmentation.clean_binary_mask(binary)
    if save_intermediate:
        visualize.save_step_image(cleaned, output_dir / "05_cleaned.png")

    # -------------------------------------------------------------
    # 3 y 4. Deteccion de lineas + extraccion de strokes
    # -------------------------------------------------------------
    result = strokes.extract_strokes(cleaned, pre["grayscale"], method=method)
    raw_strokes = result["strokes"]
    extra = result["extra"]

    if save_intermediate and extra is not None:
        if method == "edges":
            visualize.save_step_image(extra, output_dir / "06_edges.png")
        elif method == "skeleton":
            visualize.save_step_image(extra, output_dir / "07_skeleton.png")

    if save_intermediate:
        strokes_preview = visualize.draw_strokes(pre["original"], raw_strokes)
        visualize.save_step_image(strokes_preview, output_dir / "08_strokes.png")

    points_before = strokes.total_points(raw_strokes)

    # -------------------------------------------------------------
    # 5. Simplificacion
    # -------------------------------------------------------------
    simplified_strokes = simplification.simplify_strokes(raw_strokes, epsilon=epsilon)
    points_after = strokes.total_points(simplified_strokes)

    if save_intermediate:
        simplified_preview = visualize.draw_simplified_points(
            pre["original"], simplified_strokes
        )
        visualize.save_step_image(simplified_preview, output_dir / "09_simplified.png")

    # -------------------------------------------------------------
    # 6. Coordenadas 2D (px) + conversion a mm + export
    # -------------------------------------------------------------
    data = coordinates.strokes_to_dict(
        simplified_strokes, image_name=image_path.name, width_px=width_px, height_px=height_px
    )
    coordinates.save_json(data, output_dir / "trajectories.json")
    coordinates.save_csv(simplified_strokes, output_dir / "trajectories.csv")

    mm_strokes = coordinates.pixel_strokes_to_mm(
        simplified_strokes, width_px, height_px, physical_width_mm, physical_height_mm
    )
    mm_data = coordinates.strokes_to_dict(
        [[(x, y) for x, y in s] for s in mm_strokes],
        image_name=image_path.name,
        width_px=width_px,
        height_px=height_px,
    )
    coordinates.save_json(mm_data, output_dir / "trajectories_mm.json")

    # -------------------------------------------------------------
    # 7. Visualizacion final
    # -------------------------------------------------------------
    final_image = visualize.draw_final_trajectory(pre["original"], simplified_strokes)
    visualize.save_step_image(final_image, output_dir / "10_final_trajectory.png")

    total_length_px = strokes.total_length(simplified_strokes)
    reduction_pct = (
        0.0 if points_before == 0 else (1 - points_after / points_before) * 100
    )

    return {
        "method": method,
        "width_px": width_px,
        "height_px": height_px,
        "num_strokes": len(simplified_strokes),
        "points_before": points_before,
        "points_after": points_after,
        "reduction_pct": reduction_pct,
        "total_length_px": total_length_px,
        "output_dir": str(output_dir),
    }


def print_summary(image_name: str, metrics: dict) -> None:
    print("=" * 44)
    print("RESULTADO")
    print("=" * 44)
    print()
    print(f"Imagen: {image_name}")
    print()
    print("Dimensiones:")
    print(f"{metrics['width_px']} x {metrics['height_px']} px")
    print()
    print("Metodo:")
    print(metrics["method"])
    print()
    print("Strokes detectados:")
    print(metrics["num_strokes"])
    print()
    print("Puntos antes de simplificacion:")
    print(metrics["points_before"])
    print()
    print("Puntos despues de simplificacion:")
    print(metrics["points_after"])
    print()
    print("Reduccion:")
    print(f"{metrics['reduction_pct']:.1f}%")
    print()
    print("Longitud total aproximada de trayectorias:")
    print(f"{metrics['total_length_px']:.1f} px")
    print()
    print(f"Archivos de salida en: {metrics['output_dir']}")
    print("=" * 44)
    print()


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    image_path = Path(args.input)
    base_output_dir = Path(args.output_dir)
    save_intermediate = not args.no_intermediate

    try:
        if args.method == "all":
            # Modo comparacion: corre las 3 estrategias en subcarpetas
            # separadas (output/contours, output/edges, output/skeleton)
            # para poder comparar visualmente cual funciona mejor.
            for method in ("contours", "edges", "skeleton"):
                start = time.time()
                method_output_dir = base_output_dir / method
                metrics = run_pipeline_for_method(
                    image_path=image_path,
                    method=method,
                    output_dir=method_output_dir,
                    epsilon=args.epsilon,
                    threshold_method=args.threshold_method,
                    threshold_value=args.threshold,
                    physical_width_mm=args.physical_width_mm,
                    physical_height_mm=args.physical_height_mm,
                    save_intermediate=save_intermediate,
                )
                elapsed = time.time() - start
                print_summary(image_path.name, metrics)
                print(f"(tiempo: {elapsed:.2f}s)\n")
        else:
            start = time.time()
            metrics = run_pipeline_for_method(
                image_path=image_path,
                method=args.method,
                output_dir=base_output_dir,
                epsilon=args.epsilon,
                threshold_method=args.threshold_method,
                threshold_value=args.threshold,
                physical_width_mm=args.physical_width_mm,
                physical_height_mm=args.physical_height_mm,
                save_intermediate=save_intermediate,
            )
            elapsed = time.time() - start
            print_summary(image_path.name, metrics)
            print(f"(tiempo: {elapsed:.2f}s)\n")

    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
