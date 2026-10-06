"""
validate.py
===========
Validacion por lotes del modo retrato: corre el pipeline sobre TODAS las
imagenes de una carpeta y genera

    <output>/contact_sheet.jpg   original | lineas | lo que dibujara el robot
    <output>/summary.csv         metricas por imagen (strokes, tiempo, etc.)
    <output>/<imagen>/...        salida completa de cada imagen

Uso:
    python validate.py --input-dir fotos_prueba/ --output-dir output/validacion
    python validate.py --input-dir fotos_prueba/ --detail 9 --max-stroke-mm 20

Acepta los mismos parametros que main.py (excepto --input/--output-dir).
Pensado para probar con fotos REALES de la camara que se usara (buena y
mala iluminacion, fondos de distintos colores) antes de dibujar.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import main

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TILE_H = 320


def _tile(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        img = np.full((TILE_H, TILE_H, 3), 255, np.uint8)
    return cv2.resize(img, (max(1, int(img.shape[1] * TILE_H / img.shape[0])), TILE_H),
                      interpolation=cv2.INTER_AREA)


def main_validate(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="output/validacion")
    own, rest = parser.parse_known_args(argv)

    images = sorted(p for p in Path(own.input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"ERROR: no hay imagenes en {own.input_dir}", file=sys.stderr)
        return 1

    out_root = Path(own.output_dir)
    rows, sheet_rows = [], []
    for img_path in images:
        args = main.parse_args(["--input", str(img_path), *rest])
        out_dir = out_root / img_path.stem
        start = time.time()
        try:
            m = main.run(img_path, args.mode, args.method, out_dir, args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{img_path.name}] ERROR: {exc}", file=sys.stderr)
            continue
        m["seconds"] = time.time() - start
        rows.append({
            "image": img_path.name,
            **{k: round(v, 3) if isinstance(v, float) else v
               for k, v in m.items() if k not in ("bbox_mm", "output_dir")},
        })
        print(f"[{img_path.name}] cara={m['face_detected']} strokes={m['num_strokes']} "
              f"waypoints={m['num_waypoints']} tiempo_dibujo={m['estimated_time_s'] / 60:.1f}min "
              f"({m['seconds']:.1f}s)")

        lines_name = "06_lines.png" if args.mode == "face" else "05_cleaned.png"
        tiles = [_tile(out_dir / "01_original.png"), _tile(out_dir / lines_name),
                 _tile(out_dir / "10_final_trajectory.png")]
        row = np.hstack(tiles)
        cv2.putText(row, img_path.name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        sheet_rows.append(row)

    if not rows:
        return 1
    width = max(r.shape[1] for r in sheet_rows)
    sheet = np.vstack([
        cv2.copyMakeBorder(r, 0, 6, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(128, 128, 128))
        for r in sheet_rows
    ])
    cv2.imwrite(str(out_root / "contact_sheet.jpg"), sheet)

    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    no_face = [r["image"] for r in rows if r.get("face_detected") is False]
    print(f"\n{len(rows)} imagenes procesadas. Hoja de contacto: {out_root / 'contact_sheet.jpg'}")
    if no_face:
        print(f"SIN CARA DETECTADA en: {', '.join(no_face)}")
    return 0


if __name__ == "__main__":
    sys.exit(main_validate())
