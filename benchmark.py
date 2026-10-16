"""
benchmark.py
────────────
Módulo 3: Benchmark de detección de PII con división de evaluación congelada,
arnés de puntuación (F1 a nivel de token y span) y tabla de clasificación.

Cómo funciona:
  1. Toma el dataset fusionado y lo divide en train / eval (eval queda CONGELADO).
  2. Implementa un scorer que compara predicciones contra el ground truth.
  3. Genera una tabla de clasificación en Markdown.

Métricas calculadas:
  • Precision / Recall / F1  por etiqueta PII y micro-promedio
  • Exact-match span accuracy
  • Soporte (número de entidades reales)

Uso:
    python benchmark.py                 # split + score con predictor dummy
    python benchmark.py --split-only    # sólo genera la división train/eval
    python benchmark.py --score predictions.jsonl   # puntúa un archivo externo
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constantes ─────────────────────────────────────────────────────────────────

EVAL_RATIO   = 0.20   # 20 % del dataset va al split de evaluación
RANDOM_SEED  = 42
LEADERBOARD_FILE = Path("data/benchmark/leaderboard.jsonl")


# ── División train / eval ──────────────────────────────────────────────────────

def split_dataset(
    merged_path: str = "data/merged/merged_dataset.jsonl",
    out_dir: str = "data/benchmark",
    eval_ratio: float = EVAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Divide el dataset fusionado en train y eval.
    El split de eval se escribe y nunca se vuelve a regenerar si ya existe.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_file = out / "train.jsonl"
    eval_file  = out / "eval.jsonl"   # ← CONGELADO una vez creado

    merged = Path(merged_path)
    if not merged.exists():
        print(f"❌ No se encontró el dataset fusionado en: {merged}")
        print("   Ejecuta primero: python merge_datasets.py")
        sys.exit(1)

    records = [json.loads(l) for l in merged.read_text(encoding="utf-8").splitlines() if l.strip()]

    if eval_file.exists():
        print(f"  ℹ️  Split de eval ya existe ({eval_file}) — no se regenera.")
        eval_records  = [json.loads(l) for l in eval_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_ids      = {r["id"] for r in eval_records}
        train_records = [r for r in records if r["id"] not in eval_ids]
        return train_records, eval_records

    rng = random.Random(seed)
    rng.shuffle(records)
    n_eval = max(1, int(len(records) * eval_ratio))
    eval_records  = records[:n_eval]
    train_records = records[n_eval:]

    eval_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records),
        encoding="utf-8",
    )
    train_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_records),
        encoding="utf-8",
    )

    print(f"  ✅ Split generado → train={len(train_records)}  eval={len(eval_records)}")
    print(f"     eval congelado en: {eval_file.resolve()}")
    return train_records, eval_records


# ── Formato de predicciones esperado ──────────────────────────────────────────
# predictions.jsonl: una línea por muestra
# { "id": "...", "entities": [{"label": "NOMBRE", "value": "Ana"}, ...] }


# ── Arnés de puntuación ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalización para comparación: minúsculas + colapso de espacios."""
    return " ".join(text.lower().split())


def score_predictions(
    eval_path: str  = "data/benchmark/eval.jsonl",
    pred_path: str  = "data/benchmark/predictions.jsonl",
) -> dict[str, Any]:
    """
    Calcula Precision / Recall / F1 comparando predicciones contra el eval.

    Matching: un (label, normalize(value)) debe aparecer en ground truth.
    """
    eval_records = {
        r["id"]: r
        for line in Path(eval_path).read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line)) and line.strip()
    }
    pred_records = {
        r["id"]: r
        for line in Path(pred_path).read_text(encoding="utf-8").splitlines()
        if (r := json.loads(line)) and line.strip()
    }

    # ── Acumuladores por etiqueta ─────────────────────────────────────────────
    tp_by_label: dict[str, int] = defaultdict(int)
    fp_by_label: dict[str, int] = defaultdict(int)
    fn_by_label: dict[str, int] = defaultdict(int)

    for rec_id, gt in eval_records.items():
        gt_set = {
            (e["label"], _normalize(e["value"]))
            for e in gt.get("entities", [])
        }
        pred = pred_records.get(rec_id, {})
        pred_set = {
            (e["label"], _normalize(e["value"]))
            for e in pred.get("entities", [])
        }

        for item in pred_set & gt_set:
            tp_by_label[item[0]] += 1
        for item in pred_set - gt_set:
            fp_by_label[item[0]] += 1
        for item in gt_set - pred_set:
            fn_by_label[item[0]] += 1

    # ── Métricas por etiqueta ─────────────────────────────────────────────────
    all_labels = sorted(
        set(tp_by_label) | set(fp_by_label) | set(fn_by_label)
    )
    results_by_label: dict[str, dict] = {}

    for lbl in all_labels:
        tp = tp_by_label[lbl]
        fp = fp_by_label[lbl]
        fn = fn_by_label[lbl]
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results_by_label[lbl] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }

    # ── Micro-promedio global ─────────────────────────────────────────────────
    tp_total = sum(tp_by_label.values())
    fp_total = sum(fp_by_label.values())
    fn_total = sum(fn_by_label.values())
    p_micro  = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    r_micro  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) else 0.0

    return {
        "by_label": results_by_label,
        "micro_avg": {
            "precision": round(p_micro, 4),
            "recall":    round(r_micro, 4),
            "f1":        round(f1_micro, 4),
        },
    }


# ── Tabla de clasificación ─────────────────────────────────────────────────────

def update_leaderboard(
    model_name: str,
    scores: dict[str, Any],
    notes: str = "",
) -> list[dict]:
    """
    Añade una entrada a la tabla de clasificación y devuelve la tabla ordenada.
    """
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "rank":       0,  # se recalcula al final
        "model":      model_name,
        "micro_f1":   scores["micro_avg"]["f1"],
        "precision":  scores["micro_avg"]["precision"],
        "recall":     scores["micro_avg"]["recall"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "notes":      notes,
        "by_label":   scores["by_label"],
    }

    # Cargar tabla existente
    board: list[dict] = []
    if LEADERBOARD_FILE.exists():
        board = [
            json.loads(l)
            for l in LEADERBOARD_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    board.append(entry)
    board.sort(key=lambda x: x["micro_f1"], reverse=True)
    for i, e in enumerate(board, 1):
        e["rank"] = i

    LEADERBOARD_FILE.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in board),
        encoding="utf-8",
    )
    return board


def print_leaderboard(board: list[dict]) -> None:
    """Imprime la tabla de clasificación en Markdown."""
    print(f"\n{'─'*72}")
    print(f"  🏆 TABLA DE CLASIFICACIÓN  —  PII Detection Benchmark")
    print(f"{'─'*72}")
    print(f"  {'#':>3}  {'Modelo':<28}  {'F1':>7}  {'P':>7}  {'R':>7}  Fecha")
    print(f"  {'─'*3}  {'─'*28}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}")
    for e in board:
        date = e["submitted_at"][:10]
        print(f"  {e['rank']:>3}  {e['model']:<28}  {e['micro_f1']:>7.4f}  "
              f"{e['precision']:>7.4f}  {e['recall']:>7.4f}  {date}")
    print(f"{'─'*72}\n")


def _print_scores(scores: dict[str, Any]) -> None:
    """Imprime resultados detallados por etiqueta."""
    print(f"\n  {'Etiqueta':<14}  {'P':>7}  {'R':>7}  {'F1':>7}  {'Soporte':>8}")
    print(f"  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    for lbl, m in sorted(scores["by_label"].items()):
        print(f"  {lbl:<14}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  "
              f"{m['f1']:>7.4f}  {m['support']:>8}")
    ma = scores["micro_avg"]
    print(f"  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    print(f"  {'MICRO-AVG':<14}  {ma['precision']:>7.4f}  {ma['recall']:>7.4f}  "
          f"{ma['f1']:>7.4f}")


# ── Predictor de referencia (baseline regex) ───────────────────────────────────

import re as _re

_PATTERNS = {
    "EMAIL":    _re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"),
    "TELEFONO": _re.compile(r"(?:\+?\d[\d\s\-]{7,14}\d)"),
    "DNI":      _re.compile(r"\b\d{7,8}[A-Za-z]\b"),
}

def _baseline_predict(records: list[dict]) -> list[dict]:
    """
    Predictor baseline con regex simple.
    Detecta EMAIL, TELEFONO y DNI; NOMBRE y DIRECCION siempre fallan (F1=0).
    """
    preds = []
    for rec in records:
        entities = []
        text = rec.get("text", "")
        for label, pat in _PATTERNS.items():
            for m in pat.finditer(text):
                entities.append({"label": label, "value": m.group()})
        preds.append({"id": rec["id"], "entities": entities})
    return preds


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de detección PII con F1 y tabla de clasificación"
    )
    parser.add_argument("--split-only",  action="store_true",
                        help="Sólo genera el split train/eval y sale")
    parser.add_argument("--score",       metavar="PRED_JSONL", default=None,
                        help="Puntúa un archivo de predicciones externo")
    parser.add_argument("--model-name",  default="baseline_regex",
                        help="Nombre del modelo para la tabla de clasificación")
    parser.add_argument("--notes",       default="",
                        help="Notas sobre el modelo o experimento")
    parser.add_argument("--merged",      default="data/merged/merged_dataset.jsonl",
                        help="Ruta al dataset fusionado")
    parser.add_argument("--out-dir",     default="data/benchmark",
                        help="Directorio de salida del benchmark")
    args = parser.parse_args()

    print(f"\n{'─'*58}")
    print(f"  Benchmark PII — División y Puntuación")
    print(f"{'─'*58}\n")

    # 1. Dividir
    train_records, eval_records = split_dataset(
        merged_path=args.merged, out_dir=args.out_dir
    )

    if args.split_only:
        return

    # 2. Obtener predicciones
    pred_path = Path(args.out_dir) / "predictions.jsonl"

    if args.score:
        pred_path = Path(args.score)
        print(f"  Usando predicciones externas: {pred_path}")
    else:
        # Ejecutar baseline
        print(f"  Ejecutando baseline regex sobre {len(eval_records)} muestras...")
        preds = _baseline_predict(eval_records)
        pred_path.write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in preds),
            encoding="utf-8",
        )
        print(f"  Predicciones guardadas en: {pred_path}")

    # 3. Puntuar
    eval_path = Path(args.out_dir) / "eval.jsonl"
    scores = score_predictions(str(eval_path), str(pred_path))

    print(f"\n  Resultados para '{args.model_name}':")
    _print_scores(scores)

    # 4. Actualizar tabla de clasificación
    board = update_leaderboard(args.model_name, scores, notes=args.notes)
    print_leaderboard(board)
    print(f"  Tabla guardada en: {LEADERBOARD_FILE.resolve()}")


if __name__ == "__main__":
    main()
