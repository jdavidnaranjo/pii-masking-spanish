"""
tests/test_benchmark_harness.py
────────────────────────────────
Suite de pruebas unitarias para el arnés de puntuación del benchmark PII.

Cubre:
  1. División train/eval — inmutabilidad y ausencia de fuga de datos
  2. Scorer aritmético   — TP/FP/FN y F1 micro-avg
  3. Casos límite        — división por cero, predictor vacío, muestra ausente
  4. Esquemas OCR        — coordenadas válidas dentro de dimensiones de imagen
  5. Esquemas audio      — intervalos temporales positivos y dentro de duración
  6. CLI --score         — no sobreescribe leaderboard, no permite duplicados

Ejecutar:
    pytest tests/test_benchmark_harness.py -v
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

import pytest

# ── Importar módulos del proyecto ─────────────────────────────────────────────
import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import (
    _normalize,
    score_predictions,
    split_dataset,
    update_leaderboard,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


def _make_merged(n: int = 20) -> list[dict]:
    """Genera N registros sintéticos con esquema canónico."""
    return [
        {
            "id": f"sample_{i:03d}",
            "text": f"El cliente {i} llama con DNI 1234567{i}A y email test{i}@ex.com.",
            "entities": [
                {"label": "DNI",   "value": f"1234567{i}A",    "start_char": 0, "end_char": 9, "span_verified": True},
                {"label": "EMAIL", "value": f"test{i}@ex.com", "start_char": 0, "end_char": 14,"span_verified": True},
            ],
            "locale": "es_ES",
            "domain": "test",
            "source": "test_fixture",
            "text_hash": hashlib.sha256(f"sample_{i}".encode()).hexdigest()[:16],
        }
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DIVISIÓN TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitDataset:

    def test_no_id_overlap(self, tmp_path):
        """Ningún ID debe aparecer en ambos splits."""
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, _make_merged(20))

        train, eval_ = split_dataset(str(merged), out_dir=str(tmp_path))

        train_ids = {r["id"] for r in train}
        eval_ids  = {r["id"] for r in eval_}
        assert train_ids.isdisjoint(eval_ids), \
            f"FUGA DE DATOS: IDs compartidos → {train_ids & eval_ids}"

    def test_no_hash_overlap(self, tmp_path):
        """Ningún text_hash debe aparecer en ambos splits."""
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, _make_merged(20))

        train, eval_ = split_dataset(str(merged), out_dir=str(tmp_path))

        train_hashes = {r["text_hash"] for r in train}
        eval_hashes  = {r["text_hash"] for r in eval_}
        assert train_hashes.isdisjoint(eval_hashes), \
            f"HASH LEAK: hashes compartidos → {train_hashes & eval_hashes}"

    def test_total_coverage(self, tmp_path):
        """Train + eval debe cubrir exactamente todas las muestras."""
        n = 25
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, _make_merged(n))

        train, eval_ = split_dataset(str(merged), out_dir=str(tmp_path))
        assert len(train) + len(eval_) == n

    def test_eval_frozen_after_first_run(self, tmp_path):
        """El eval no se regenera si ya existe el archivo."""
        merged = tmp_path / "merged.jsonl"
        records = _make_merged(20)
        _write_jsonl(merged, records)

        # Primera ejecución
        _, eval_v1 = split_dataset(str(merged), out_dir=str(tmp_path))
        eval_ids_v1 = {r["id"] for r in eval_v1}

        # Añadir registros nuevos al merged
        records += _make_merged(5)   # 5 registros extra con IDs distintos
        _write_jsonl(merged, records)

        # Segunda ejecución → eval debe ser idéntico
        _, eval_v2 = split_dataset(str(merged), out_dir=str(tmp_path))
        eval_ids_v2 = {r["id"] for r in eval_v2}

        assert eval_ids_v1 == eval_ids_v2, \
            "El split de eval cambió entre ejecuciones — no está congelado"

    def test_eval_size_is_20_percent(self, tmp_path):
        """El split de eval debe ser aprox. 20 % del total."""
        n = 30
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, _make_merged(n))

        train, eval_ = split_dataset(str(merged), out_dir=str(tmp_path))
        ratio = len(eval_) / n
        assert 0.15 <= ratio <= 0.25, f"Ratio eval inesperado: {ratio:.2f}"

    def test_seed_determinism(self, tmp_path):
        """Misma seed debe producir el mismo split en dos tmp_path distintos."""
        records = _make_merged(20)

        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"

        m_a = dir_a / "merged.jsonl"
        m_b = dir_b / "merged.jsonl"
        m_a.parent.mkdir()
        m_b.parent.mkdir()
        _write_jsonl(m_a, records)
        _write_jsonl(m_b, records)

        _, eval_a = split_dataset(str(m_a), out_dir=str(dir_a), seed=42)
        _, eval_b = split_dataset(str(m_b), out_dir=str(dir_b), seed=42)

        assert {r["id"] for r in eval_a} == {r["id"] for r in eval_b}, \
            "El split no es determinista con la misma semilla"


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCORER ARITMÉTICO
# ══════════════════════════════════════════════════════════════════════════════

class TestScorer:

    def _score(self, tmp_path, gt_entities, pred_entities):
        """Utilidad: construye eval y predictions y devuelve scores."""
        eval_rec  = [{"id": "s1", "text": "test", "entities": gt_entities}]
        pred_rec  = [{"id": "s1", "entities": pred_entities}]
        eval_path = tmp_path / "eval.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        _write_jsonl(eval_path, eval_rec)
        _write_jsonl(pred_path, pred_rec)
        return score_predictions(str(eval_path), str(pred_path))

    def test_perfect_prediction(self, tmp_path):
        """Predicción perfecta → P=R=F1=1.0."""
        gt   = [{"label": "DNI",   "value": "12345678A"},
                {"label": "EMAIL", "value": "a@b.com"}]
        pred = [{"label": "DNI",   "value": "12345678A"},
                {"label": "EMAIL", "value": "a@b.com"}]
        scores = self._score(tmp_path, gt, pred)
        assert scores["micro_avg"]["f1"] == pytest.approx(1.0)
        assert scores["micro_avg"]["precision"] == pytest.approx(1.0)
        assert scores["micro_avg"]["recall"] == pytest.approx(1.0)

    def test_empty_prediction(self, tmp_path):
        """Sin predicciones → P=undefined(0), R=0, F1=0."""
        gt   = [{"label": "NOMBRE", "value": "Ana"}]
        pred = []
        scores = self._score(tmp_path, gt, pred)
        assert scores["micro_avg"]["f1"] == pytest.approx(0.0)
        assert scores["micro_avg"]["recall"] == pytest.approx(0.0)

    def test_all_false_positives(self, tmp_path):
        """Solo FP: predicciones que no están en GT → P=0, R=0, F1=0."""
        gt   = [{"label": "NOMBRE", "value": "Ana"}]
        pred = [{"label": "NOMBRE", "value": "Carlos"},
                {"label": "DNI",    "value": "11111111A"}]
        scores = self._score(tmp_path, gt, pred)
        assert scores["micro_avg"]["precision"] == pytest.approx(0.0)

    def test_no_division_by_zero_when_p_r_zero(self, tmp_path):
        """Cuando P=0 y R=0, F1 debe ser 0.0 sin excepción."""
        gt   = [{"label": "NOMBRE",   "value": "Ana"},
                {"label": "DIRECCION","value": "Calle 1"}]
        pred = []
        scores = self._score(tmp_path, gt, pred)
        for lbl, m in scores["by_label"].items():
            assert m["f1"] == pytest.approx(0.0), \
                f"F1 de {lbl} debería ser 0.0, obtuvo {m['f1']}"

    def test_micro_f1_arithmetic(self, tmp_path):
        """Verifica aritmética: F1 = 2·P·R/(P+R) con fixture controlada.

        Fixture: 7 muestras, cada una con DNI+EMAIL (predichos) + NOMBRE (no predicho)
          TP = 7×2 = 14  (DNI y EMAIL correctos)
          FP = 0
          FN = 7×1 = 7   (NOMBRE nunca predicho)
          P  = 14/14 = 1.0
          R  = 14/21 ≈ 0.6667
          F1 = 2·1·(2/3)/(1+2/3) = 0.8
        """
        tp, fp, fn = 14, 0, 7
        p_exp  = tp / (tp + fp)        # 1.0
        r_exp  = tp / (tp + fn)        # 14/21 ≈ 0.6667
        f1_exp = 2 * p_exp * r_exp / (p_exp + r_exp)   # 0.8

        gt_records = [
            {"id": f"s{i}", "text": "x", "entities": [
                {"label": "DNI",    "value": f"dni{i}"},
                {"label": "EMAIL",  "value": f"e{i}@x.com"},
                {"label": "NOMBRE", "value": f"Nombre {i}"},
            ]}
            for i in range(7)
        ]
        pred_records = [
            {"id": f"s{i}", "entities": [
                {"label": "DNI",   "value": f"dni{i}"},
                {"label": "EMAIL", "value": f"e{i}@x.com"},
                # NOMBRE intencionalmente omitido → FN
            ]}
            for i in range(7)
        ]
        eval_path = tmp_path / "eval.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        _write_jsonl(eval_path, gt_records)
        _write_jsonl(pred_path, pred_records)

        scores = score_predictions(str(eval_path), str(pred_path))
        assert scores["micro_avg"]["precision"] == pytest.approx(p_exp,  abs=1e-4)
        assert scores["micro_avg"]["recall"]    == pytest.approx(r_exp,  abs=1e-4)
        assert scores["micro_avg"]["f1"]        == pytest.approx(f1_exp, abs=1e-4)

    def test_case_insensitive_matching(self, tmp_path):
        """El matching debe ser insensible a mayúsculas y espacios extra."""
        gt   = [{"label": "NOMBRE", "value": "María López"}]
        pred = [{"label": "NOMBRE", "value": "MARÍA  LÓPEZ"}]
        scores = self._score(tmp_path, gt, pred)
        assert scores["micro_avg"]["f1"] == pytest.approx(1.0), \
            "Matching debe ser case-insensitive y normalizar espacios"

    def test_missing_sample_in_predictions(self, tmp_path):
        """Si una muestra no aparece en predicciones, sus entidades son FN."""
        gt   = [
            {"id": "s1", "text": "a", "entities": [{"label": "DNI", "value": "123X"}]},
            {"id": "s2", "text": "b", "entities": [{"label": "DNI", "value": "456Y"}]},
        ]
        pred = [{"id": "s1", "entities": [{"label": "DNI", "value": "123X"}]}]
        # s2 ausente → FN += 1
        eval_path = tmp_path / "eval.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        _write_jsonl(eval_path, gt)
        _write_jsonl(pred_path, pred)

        scores = score_predictions(str(eval_path), str(pred_path))
        assert scores["micro_avg"]["recall"] == pytest.approx(0.5, abs=1e-3)

    def test_support_count_matches_gt_entities(self, tmp_path):
        """El campo 'support' debe igual al número de entidades GT por label."""
        gt   = [
            {"id": "a", "text": "x", "entities": [
                {"label": "DNI", "value": "111X"},
                {"label": "DNI", "value": "222Y"},
                {"label": "EMAIL", "value": "a@b.com"},
            ]}
        ]
        pred = [{"id": "a", "entities": []}]
        eval_path = tmp_path / "eval.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        _write_jsonl(eval_path, gt)
        _write_jsonl(pred_path, pred)

        scores = score_predictions(str(eval_path), str(pred_path))
        assert scores["by_label"]["DNI"]["support"] == 2
        assert scores["by_label"]["EMAIL"]["support"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. ESQUEMAS OCR
# ══════════════════════════════════════════════════════════════════════════════

class TestOCRSchema:

    OCR_FILE = ROOT / "data/ocr_dataset/annotations.jsonl"

    @pytest.mark.skipif(not OCR_FILE.exists(), reason="Dataset OCR no generado")
    def test_bounding_boxes_non_negative(self):
        """Ninguna caja debe tener coordenadas negativas."""
        records = [json.loads(l) for l in self.OCR_FILE.read_text().splitlines() if l.strip()]
        errors = []
        for rec in records:
            for bb in rec.get("bounding_boxes", []):
                if bb["x"] < 0 or bb["y"] < 0 or bb["width"] <= 0 or bb["height"] <= 0:
                    errors.append(f"{rec['sample_id']}: {bb}")
        assert not errors, f"Bounding boxes con coords inválidas:\n" + "\n".join(errors)

    @pytest.mark.skipif(not OCR_FILE.exists(), reason="Dataset OCR no generado")
    def test_bounding_boxes_within_image(self):
        """Las cajas no deben superar las dimensiones de la imagen."""
        records = [json.loads(l) for l in self.OCR_FILE.read_text().splitlines() if l.strip()]
        errors = []
        for rec in records:
            W, H = rec["width_px"], rec["height_px"]
            for bb in rec.get("bounding_boxes", []):
                if bb["x"] + bb["width"] > W or bb["y"] + bb["height"] > H:
                    errors.append(f"{rec['sample_id']}: {bb} en {W}x{H}")
        assert not errors, f"Cajas fuera de imagen:\n" + "\n".join(errors)

    @pytest.mark.skipif(not OCR_FILE.exists(), reason="Dataset OCR no generado")
    def test_all_records_have_bounding_boxes(self):
        """Todo registro OCR debe tener al menos una caja PII."""
        records = [json.loads(l) for l in self.OCR_FILE.read_text().splitlines() if l.strip()]
        empty = [r["sample_id"] for r in records if not r.get("bounding_boxes")]
        assert not empty, f"Registros sin bounding boxes: {empty}"

    @pytest.mark.skipif(not OCR_FILE.exists(), reason="Dataset OCR no generado")
    def test_doc_types_are_valid(self):
        """Solo se permiten doc_types registrados."""
        valid = {"formulario", "factura", "id_card"}
        records = [json.loads(l) for l in self.OCR_FILE.read_text().splitlines() if l.strip()]
        invalid = [r["doc_type"] for r in records if r["doc_type"] not in valid]
        assert not invalid, f"doc_types desconocidos: {set(invalid)}"

    @pytest.mark.skipif(not OCR_FILE.exists(), reason="Dataset OCR no generado")
    def test_pii_labels_are_valid(self):
        """Todas las etiquetas PII deben pertenecer al vocabulario definido."""
        valid_labels = {"NOMBRE", "DNI", "TELEFONO", "EMAIL", "DIRECCION",
                        "FECHA", "EMPRESA", "TARJETA", "CUENTA", "PASAPORTE"}
        records = [json.loads(l) for l in self.OCR_FILE.read_text().splitlines() if l.strip()]
        bad = []
        for rec in records:
            for bb in rec.get("bounding_boxes", []):
                if bb["label"] not in valid_labels:
                    bad.append(bb["label"])
        assert not bad, f"Etiquetas inválidas: {set(bad)}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. ESQUEMAS AUDIO
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioSchema:

    AUDIO_FILE = ROOT / "data/pilot_dataset/pilot_metadata.jsonl"

    @pytest.mark.skipif(not AUDIO_FILE.exists(), reason="Dataset de audio no generado")
    def test_time_intervals_positive(self):
        """start_time y end_time deben ser mayores a 0."""
        records = [json.loads(l) for l in self.AUDIO_FILE.read_text().splitlines() if l.strip()]
        errors = []
        for rec in records:
            for ann in rec.get("pii_annotations", []):
                if ann["start_time"] < 0 or ann["end_time"] <= 0:
                    errors.append(f"{rec['sample_id']}: {ann}")
        assert not errors, f"Intervalos de tiempo inválidos:\n" + "\n".join(errors)

    @pytest.mark.skipif(not AUDIO_FILE.exists(), reason="Dataset de audio no generado")
    def test_end_time_within_duration(self):
        """end_time no debe superar la duración total del audio."""
        records = [json.loads(l) for l in self.AUDIO_FILE.read_text().splitlines() if l.strip()]
        errors = []
        for rec in records:
            dur = rec.get("audio_duration_seconds", 0)
            for ann in rec.get("pii_annotations", []):
                if ann["end_time"] > dur + 0.05:   # +50ms tolerancia
                    errors.append(f"{rec['sample_id']}: end={ann['end_time']} > dur={dur}")
        assert not errors, f"Intervalos fuera de duración:\n" + "\n".join(errors)

    @pytest.mark.skipif(not AUDIO_FILE.exists(), reason="Dataset de audio no generado")
    def test_start_before_end(self):
        """start_time debe ser estrictamente menor a end_time."""
        records = [json.loads(l) for l in self.AUDIO_FILE.read_text().splitlines() if l.strip()]
        errors = []
        for rec in records:
            for ann in rec.get("pii_annotations", []):
                if ann["start_time"] >= ann["end_time"]:
                    errors.append(f"{rec['sample_id']}: start={ann['start_time']} >= end={ann['end_time']}")
        assert not errors, f"Intervalos invertidos:\n" + "\n".join(errors)

    @pytest.mark.skipif(not AUDIO_FILE.exists(), reason="Dataset de audio no generado")
    def test_audio_files_exist(self):
        """Los archivos WAV referenciados deben existir en disco."""
        records = [json.loads(l) for l in self.AUDIO_FILE.read_text().splitlines() if l.strip()]
        missing = []
        for rec in records:
            for key in ("original_audio_path", "censored_audio_path"):
                p = Path(rec.get(key, ""))
                if not p.exists():
                    missing.append(str(p))
        assert not missing, f"Archivos WAV faltantes:\n" + "\n".join(missing)


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEADERBOARD — SEGURIDAD Y PERSISTENCIA
# ══════════════════════════════════════════════════════════════════════════════

class TestLeaderboard:

    def _fake_scores(self, f1=0.8, p=0.9, r=0.72):
        return {
            "micro_avg": {"f1": f1, "precision": p, "recall": r},
            "by_label":  {"NOMBRE": {"precision": p, "recall": r, "f1": f1, "support": 10}},
        }

    def test_new_entry_appended(self, tmp_path):
        """Una nueva entrada debe añadirse sin borrar las existentes."""
        import benchmark as bm
        original_lb = bm.LEADERBOARD_FILE
        bm.LEADERBOARD_FILE = tmp_path / "lb.jsonl"

        update_leaderboard("modelo_A", self._fake_scores(0.8))
        update_leaderboard("modelo_B", self._fake_scores(0.9))

        entries = [json.loads(l) for l in bm.LEADERBOARD_FILE.read_text().splitlines() if l.strip()]
        assert len(entries) == 2
        bm.LEADERBOARD_FILE = original_lb

    def test_sorted_by_f1_desc(self, tmp_path):
        """El leaderboard debe estar ordenado de mayor a menor F1."""
        import benchmark as bm
        original_lb = bm.LEADERBOARD_FILE
        bm.LEADERBOARD_FILE = tmp_path / "lb.jsonl"

        update_leaderboard("bajo",  self._fake_scores(0.5))
        update_leaderboard("alto",  self._fake_scores(0.95))
        update_leaderboard("medio", self._fake_scores(0.75))

        entries = [json.loads(l) for l in bm.LEADERBOARD_FILE.read_text().splitlines() if l.strip()]
        f1s = [e["micro_f1"] for e in entries]
        assert f1s == sorted(f1s, reverse=True), f"Orden incorrecto: {f1s}"
        bm.LEADERBOARD_FILE = original_lb

    def test_rank_is_sequential(self, tmp_path):
        """Los rangos deben ser 1, 2, 3 ... sin saltos."""
        import benchmark as bm
        original_lb = bm.LEADERBOARD_FILE
        bm.LEADERBOARD_FILE = tmp_path / "lb.jsonl"

        for i, f1 in enumerate([0.9, 0.7, 0.5]):
            update_leaderboard(f"model_{i}", self._fake_scores(f1))

        entries = [json.loads(l) for l in bm.LEADERBOARD_FILE.read_text().splitlines() if l.strip()]
        ranks = [e["rank"] for e in entries]
        assert ranks == list(range(1, len(entries) + 1))
        bm.LEADERBOARD_FILE = original_lb

    def test_duplicate_model_allowed_appended(self, tmp_path):
        """Registrar el mismo modelo dos veces crea 2 entradas (historial)."""
        import benchmark as bm
        original_lb = bm.LEADERBOARD_FILE
        bm.LEADERBOARD_FILE = tmp_path / "lb.jsonl"

        update_leaderboard("modelo_v1", self._fake_scores(0.7))
        update_leaderboard("modelo_v1", self._fake_scores(0.85))  # versión mejorada

        entries = [json.loads(l) for l in bm.LEADERBOARD_FILE.read_text().splitlines() if l.strip()]
        assert len(entries) == 2, "Debe haber 2 entradas (historial de versiones)"
        bm.LEADERBOARD_FILE = original_lb


# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALIZACIÓN DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalize:

    @pytest.mark.parametrize("text, expected", [
        ("María López",    "maría lópez"),
        ("JUAN  GARCÍA",   "juan garcía"),
        ("  test  ",       "test"),
        ("a@B.COM",        "a@b.com"),
        ("12345678A",      "12345678a"),
        ("",               ""),
    ])
    def test_normalize(self, text, expected):
        assert _normalize(text) == expected
