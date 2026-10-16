"""
merge_datasets.py
─────────────────
Módulo 2: Fusión y deduplicación de datasets de texto PII.

Lee todos los archivos JSONL de texto (pilot_metadata, datasets externos, etc.),
los normaliza al mismo esquema, elimina duplicados por hash de texto,
aplica controles de calidad y escribe un dataset unificado con procedencia clara.

Salida:
    data/merged/merged_dataset.jsonl   → dataset limpio y deduplicado
    data/merged/provenance.jsonl       → registro de origen de cada muestra
    data/merged/merge_report.json      → estadísticas de la fusión
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constantes de calidad ──────────────────────────────────────────────────────

MIN_TEXT_LEN    = 15    # caracteres mínimos en el texto
MAX_TEXT_LEN    = 2000  # caracteres máximos
MIN_PII_FIELDS  = 1     # al menos 1 entidad PII por muestra
VALID_LABELS    = {"NOMBRE", "DNI", "TELEFONO", "EMAIL", "DIRECCION",
                   "FECHA", "EMPRESA", "TARJETA", "CUENTA", "PASAPORTE"}


# ── Normalización de esquema ───────────────────────────────────────────────────

def _normalize_text_record(raw: dict, source_name: str) -> dict | None:
    """
    Convierte cualquier registro crudo al esquema canónico:
    {
      id, text, entities: [{label, value, start_char, end_char}],
      locale, domain, source, text_hash
    }
    Devuelve None si el registro no cumple el control de calidad.
    """
    # ── Extraer texto ─────────────────────────────────────────────────────────
    text = (
        raw.get("text") or
        raw.get("sentence") or
        raw.get("content") or ""
    ).strip()

    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return None

    # ── Extraer entidades PII ─────────────────────────────────────────────────
    entities: list[dict] = []

    # Formato propio del proyecto: campo "pii_annotations" o "entities"
    raw_entities = (
        raw.get("entities") or
        raw.get("pii_annotations") or
        raw.get("spans") or
        []
    )

    for ent in raw_entities:
        label = str(ent.get("label", "")).upper().strip()
        value = str(ent.get("value") or ent.get("text_segment") or ent.get("text") or "").strip()
        if not label or not value:
            continue
        # Aceptar cualquier label; normalizar los conocidos
        entities.append({
            "label":      label,
            "value":      value,
            "start_char": int(ent.get("start_char", -1)),
            "end_char":   int(ent.get("end_char",   -1)),
        })

    if len(entities) < MIN_PII_FIELDS:
        return None

    # ── Control de calidad: verificar spans válidos ───────────────────────────
    verified_entities = []
    for e in entities:
        s, end = e["start_char"], e["end_char"]
        if s >= 0 and end > s:
            if text[s:end] == e["value"]:          # span exacto
                verified_entities.append({**e, "span_verified": True})
            else:
                verified_entities.append({**e, "span_verified": False})
        else:
            verified_entities.append({**e, "span_verified": False})

    # ── Hash del texto (para deduplicación) ──────────────────────────────────
    text_hash = hashlib.sha256(
        re.sub(r"\s+", " ", text.lower()).encode()
    ).hexdigest()[:16]

    # ── Construir registro canónico ───────────────────────────────────────────
    sample_id = (
        raw.get("id") or
        raw.get("sample_id") or
        f"{source_name}_{text_hash}"
    )

    return {
        "id":       sample_id,
        "text":     text,
        "entities": verified_entities,
        "locale":   raw.get("locale", "es_ES"),
        "domain":   raw.get("domain", "general"),
        "source":   source_name,
        "text_hash": text_hash,
    }


# ── Lectura de fuentes ─────────────────────────────────────────────────────────

def _iter_jsonl(path: Path):
    """Itera líneas JSONL saltando errores de parse."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  ⚠️  Parse error en {path.name}:{i} → {exc}")


def _load_pilot_metadata(path: Path) -> list[dict]:
    """Lee el formato pilot_metadata.jsonl (tiene pii_annotations con timestamps)."""
    records = []
    for raw in _iter_jsonl(path):
        # Construir entities a partir de pii_annotations del piloto de voz
        anns = raw.get("pii_annotations", [])
        text = raw.get("text", "").strip()
        entities = []
        for ann in anns:
            val = ann.get("text_segment", "")
            if val and val in text:
                idx = text.find(val)
                entities.append({
                    "label":      ann.get("label", ""),
                    "value":      val,
                    "start_char": idx,
                    "end_char":   idx + len(val),
                })
        if entities:
            records.append({**raw, "entities": entities})
    return records


# ── Pipeline principal ─────────────────────────────────────────────────────────

def merge_datasets(
    sources: list[dict[str, Any]] | None = None,
    out_dir: str = "data/merged",
) -> dict:
    """
    Fusiona múltiples fuentes JSONL en un dataset unificado y deduplicado.

    Args:
        sources: Lista de dicts con claves 'path' y 'loader' (opcional).
                 Si es None, usa las fuentes por defecto del proyecto.
        out_dir: Directorio de salida.

    Returns:
        Diccionario con estadísticas del proceso.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged_file    = out_path / "merged_dataset.jsonl"
    provenance_file = out_path / "provenance.jsonl"
    report_file    = out_path / "merge_report.json"

    # ── Fuentes por defecto ───────────────────────────────────────────────────
    if sources is None:
        sources = []
        defaults = [
            ("data/pilot_dataset/pilot_metadata.jsonl", "pilot_voice"),
            ("data/ocr_dataset/annotations.jsonl",      "ocr_documents"),
        ]
        for rel_path, name in defaults:
            p = ROOT / rel_path
            if p.exists():
                sources.append({"path": str(p), "name": name})
            else:
                print(f"  ℹ️  Fuente no encontrada (se omite): {rel_path}")

    print(f"\n{'─'*58}")
    print(f"  Fusión y Deduplicación de Datasets PII")
    print(f"{'─'*58}")
    print(f"  Fuentes : {len(sources)}")
    print(f"  Salida  : {out_path.resolve()}")
    print(f"{'─'*58}\n")

    seen_hashes: set[str] = set()
    all_records:  list[dict] = []
    stats: dict[str, Any] = defaultdict(int)
    label_counts: Counter  = Counter()
    source_counts: Counter = Counter()

    for src in sources:
        src_path = Path(src["path"])
        src_name = src.get("name", src_path.stem)
        loader   = src.get("loader")  # función personalizada opcional

        print(f"  📂 Cargando '{src_name}' ...", end=" ", flush=True)

        if not src_path.exists():
            print(f"❌ no encontrado")
            continue

        raw_records = (
            loader(src_path) if loader
            else (list(_iter_jsonl(src_path))
                  if "ocr" not in src_name
                  else list(_iter_jsonl(src_path)))
        )

        # Para pilot_metadata usar el loader especial
        if "pilot" in src_name:
            raw_records = _load_pilot_metadata(src_path)

        loaded = dup = bad = 0

        for raw in raw_records:
            # Los registros OCR no tienen texto pero tienen bounding_boxes
            # Los convertimos a muestras de texto con la lista de valores PII
            if raw.get("doc_type"):
                bbs = raw.get("bounding_boxes", [])
                if not bbs:
                    bad += 1
                    continue
                text_parts = [f"{bb['label']}: {bb['value']}" for bb in bbs]
                raw_text = " | ".join(text_parts)
                raw = {
                    "id":       raw.get("sample_id", ""),
                    "text":     raw_text,
                    "entities": [
                        {"label": bb["label"], "value": bb["value"],
                         "start_char": -1, "end_char": -1}
                        for bb in bbs
                    ],
                    "locale": "es_ES",
                    "domain": f"ocr_{raw['doc_type']}",
                }

            record = _normalize_text_record(raw, src_name)
            if record is None:
                bad += 1
                continue

            h = record["text_hash"]
            if h in seen_hashes:
                dup += 1
                continue

            seen_hashes.add(h)
            all_records.append(record)
            for ent in record["entities"]:
                label_counts[ent["label"]] += 1
            source_counts[src_name] += 1
            loaded += 1

        print(f"✓  cargados={loaded}  duplicados={dup}  descartados={bad}")
        stats["total_loaded"]    += loaded
        stats["total_duplicates"] += dup
        stats["total_discarded"]  += bad

    # ── Escribir salidas ──────────────────────────────────────────────────────
    with (
        open(merged_file,     "w", encoding="utf-8") as f_merged,
        open(provenance_file, "w", encoding="utf-8") as f_prov,
    ):
        for rec in all_records:
            f_merged.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_prov.write(json.dumps({
                "id":        rec["id"],
                "source":    rec["source"],
                "domain":    rec["domain"],
                "text_hash": rec["text_hash"],
                "n_entities": len(rec["entities"]),
                "span_verified": sum(1 for e in rec["entities"] if e.get("span_verified")),
            }, ensure_ascii=False) + "\n")

    # ── Reporte ───────────────────────────────────────────────────────────────
    report = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "total_records":   len(all_records),
        "total_entities":  sum(label_counts.values()),
        "duplicates_removed": stats["total_duplicates"],
        "discarded_qc":    stats["total_discarded"],
        "by_source":       dict(source_counts),
        "by_label":        dict(label_counts.most_common()),
        "output_files": {
            "dataset":    str(merged_file),
            "provenance": str(provenance_file),
            "report":     str(report_file),
        },
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  ✅ Fusión completada")
    print(f"{'─'*58}")
    print(f"  Total registros     : {len(all_records)}")
    print(f"  Total entidades PII : {sum(label_counts.values())}")
    print(f"  Duplicados removidos: {stats['total_duplicates']}")
    print(f"  Descartados (QC)    : {stats['total_discarded']}")
    print(f"\n  Por fuente:")
    for src, cnt in source_counts.items():
        print(f"    {src:<20} {cnt:>4} muestras")
    print(f"\n  Por etiqueta PII:")
    for lbl, cnt in label_counts.most_common():
        bar = "█" * (cnt // max(1, sum(label_counts.values()) // 20))
        print(f"    {lbl:<14} {cnt:>4}  {bar}")
    print(f"\n  Archivos en: {out_path.resolve()}")
    print(f"{'─'*58}\n")

    return report


if __name__ == "__main__":
    merge_datasets()
