"""
generate_ocr_dataset.py
───────────────────────
Script principal para generar el dataset piloto de documentos OCR con PII anotado.

Genera N documentos de cada tipo (formulario, factura, id_card), los guarda como PNG
y escribe un archivo JSONL con todas las anotaciones (cajas delimitadoras de PII).

Uso:
    python generate_ocr_dataset.py
    python generate_ocr_dataset.py --n 50 --out data/ocr_dataset

Salida:
    data/ocr_dataset/images/   → imágenes PNG
    data/ocr_dataset/annotations.jsonl → una línea JSON por documento
    data/ocr_dataset/preview/  → imágenes con cajas dibujadas (para verificación visual)
"""

import argparse
import json
import sys
from pathlib import Path

# ── Asegurar que el paquete raíz esté en el path ───────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_generator import DOC_TYPES, generate_document

# ── Preview: dibuja cajas sobre la imagen para verificación ───────────────────

def _draw_preview(annotation, img_dir: Path, preview_dir: Path) -> None:
    """Crea una copia de la imagen con las cajas PII dibujadas encima."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    img_path = Path(annotation.image_path)
    if not img_path.exists():
        return

    img = Image.open(img_path).copy()
    draw = ImageDraw.Draw(img)

    COLORS = {
        "NOMBRE":   "#FF4444",
        "DNI":      "#FF8800",
        "TELEFONO": "#9933FF",
        "EMAIL":    "#0088FF",
        "DIRECCION": "#00AA44",
    }

    for bb in annotation.bounding_boxes:
        color = COLORS.get(bb.label, "#FFCC00")
        x1, y1, x2, y2 = bb.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 14)), bb.label, fill=color)

    preview_dir.mkdir(parents=True, exist_ok=True)
    out_path = preview_dir / img_path.name
    img.save(out_path)


# ── Función principal ──────────────────────────────────────────────────────────

def main(n_per_type: int = 10, out_dir: str = "data/ocr_dataset",
         preview: bool = True) -> None:

    out_path = Path(out_dir)
    images_dir  = out_path / "images"
    preview_dir = out_path / "preview"
    annot_file  = out_path / "annotations.jsonl"

    images_dir.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    total = len(DOC_TYPES) * n_per_type
    generated = 0

    print(f"\n{'─'*55}")
    print(f"  Dataset OCR de Documentos Sintéticos con PII")
    print(f"{'─'*55}")
    print(f"  Tipos    : {', '.join(DOC_TYPES)}")
    print(f"  Por tipo : {n_per_type}")
    print(f"  Total    : {total} documentos")
    print(f"  Salida   : {out_path.resolve()}")
    print(f"{'─'*55}\n")

    with open(annot_file, "w", encoding="utf-8") as f_annot:
        for doc_type in DOC_TYPES:
            print(f"  Generando {n_per_type} × '{doc_type}' ...", end="", flush=True)
            for i in range(n_per_type):
                img_name = f"{doc_type}_{i:04d}.png"
                img_path = images_dir / img_name

                annotation = generate_document(
                    doc_type=doc_type,
                    output_path=img_path,
                )

                # Guardar anotación como línea JSONL
                f_annot.write(annotation.model_dump_json() + "\n")

                # Preview con cajas dibujadas
                if preview:
                    _draw_preview(annotation, images_dir, preview_dir)

                generated += 1

            n_boxes = sum(
                1 for line in open(annot_file, encoding="utf-8")
                if json.loads(line)["doc_type"] == doc_type
            )
            print(f" ✓  ({n_boxes} docs anotados hasta ahora)")

    # ── Resumen ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  ✅ Generación completada")
    print(f"{'─'*55}")

    all_annotations = [
        json.loads(line)
        for line in open(annot_file, encoding="utf-8")
    ]

    total_boxes = sum(len(a["bounding_boxes"]) for a in all_annotations)
    from collections import Counter
    label_counts: Counter = Counter()
    for ann in all_annotations:
        for bb in ann["bounding_boxes"]:
            label_counts[bb["label"]] += 1

    print(f"  Documentos generados : {generated}")
    print(f"  Cajas PII totales    : {total_boxes}")
    print(f"\n  Distribución por etiqueta:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // max(1, total_boxes // 30))
        print(f"    {label:<12} {count:>4}  {bar}")

    print(f"\n  Archivos guardados en:")
    print(f"    {images_dir.resolve()}")
    print(f"    {annot_file.resolve()}")
    if preview:
        print(f"    {preview_dir.resolve()}  (previews con cajas)")
    print(f"{'─'*55}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset OCR de documentos sintéticos con PII anotado"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Número de documentos por tipo (default: 10)"
    )
    parser.add_argument(
        "--out", type=str, default="data/ocr_dataset",
        help="Directorio de salida (default: data/ocr_dataset)"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="No generar imágenes de preview con cajas dibujadas"
    )
    args = parser.parse_args()
    main(n_per_type=args.n, out_dir=args.out, preview=not args.no_preview)
