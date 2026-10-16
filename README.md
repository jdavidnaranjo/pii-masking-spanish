# 🔒 PII Masking Dataset Pipeline

Pipeline reproducible de generación, censura y anotación de datos de **Información de Identificación Personal (PII)** en español, para entrenamiento de modelos de IA que preserven la privacidad.

---

## 📋 Descripción del proyecto

Este repositorio implementa tres módulos complementarios:

| Módulo | Script | Descripción |
|---|---|---|
| 🎤 **Voz** | `generate_pilot_dataset.py` | Audio TTS con censura por bip y alineación a nivel de palabra |
| 📄 **OCR** | `generate_ocr_dataset.py` | Documentos sintéticos (formularios, facturas, ID) con cajas PII |
| 🔀 **Fusión** | `merge_datasets.py` | Unificación, deduplicación y control de calidad |
| 🏆 **Benchmark** | `benchmark.py` | División eval congelada, F1 por etiqueta y tabla de clasificación |

---

## ⚙️ Instalación

### Requisitos
- Python ≥ 3.10
- `arecord` (para grabación de micrófono, solo Linux): `sudo apt install alsa-utils`

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd "Proyecto 1"

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno (opcional, solo para LLM)
cp .env.example .env
# Editar .env con tus claves API
```

---

## 🚀 Ejecución rápida (pipeline completo)

```bash
# Paso 1: Generar dataset de voz (5 muestras)
python generate_pilot_dataset.py

# Paso 2: Generar dataset OCR (10 docs por tipo = 30 total)
python generate_ocr_dataset.py --n 10

# Paso 3: Fusionar y deduplicar todos los datasets
python merge_datasets.py

# Paso 4: Correr el benchmark (split + puntuación baseline)
python benchmark.py
```

---

## 📁 Estructura del proyecto

```
Proyecto 1/
│
├── schemas/
│   └── pii_models.py          ← Modelos Pydantic compartidos
│       ├── PIISpan            # Entidad en texto (start_char, end_char)
│       ├── SyntheticTextSample
│       ├── TimeInterval       # Intervalo de audio (start_time, end_time)
│       ├── AudioPIIAnnotation
│       ├── BoundingBox        # Caja PII en imagen (x, y, width, height)
│       └── DocumentPIIAnnotation
│
├── src/
│   ├── data_generator.py      ← Generador de texto sintético con Faker
│   ├── audio_censor.py        ← Censura de audio con tono bip (DSP)
│   ├── forced_aligner.py      ← Alineación forzada PyTorch MMS_FA
│   ├── llm_generator.py       ← Generación con LLM (OpenAI / Gemini)
│   └── document_generator.py  ← Generador de imágenes PNG con Pillow
│
├── generate_pilot_dataset.py  ← Pipeline de voz
├── generate_ocr_dataset.py    ← Pipeline de documentos OCR
├── merge_datasets.py          ← Fusión y deduplicación
├── benchmark.py               ← Benchmark con F1 y leaderboard
├── probar_con_mi_voz.py       ← Demo interactivo con micrófono
│
├── data/
│   ├── pilot_dataset/
│   │   ├── original_wavs/     ← Audios originales (.wav)
│   │   ├── censored_wavs/     ← Audios censurados con bip (.wav)
│   │   └── pilot_metadata.jsonl
│   ├── ocr_dataset/
│   │   ├── images/            ← Imágenes PNG limpias
│   │   ├── preview/           ← PNG con cajas dibujadas (QA visual)
│   │   └── annotations.jsonl
│   ├── merged/
│   │   ├── merged_dataset.jsonl
│   │   ├── provenance.jsonl
│   │   └── merge_report.json
│   └── benchmark/
│       ├── train.jsonl
│       ├── eval.jsonl         ← ⚠️ CONGELADO — no regenerar
│       ├── predictions.jsonl
│       └── leaderboard.jsonl
│
├── tests/
│   ├── test_spans.py
│   └── test_audio_censor.py
│
├── requirements.txt
└── .env.example
```

---

## 📊 Esquemas de datos

### Texto (`merged_dataset.jsonl`)
```json
{
  "id": "pilot_voice_a1b2c3d4",
  "text": "Hola, soy María López y mi DNI es 12345678A.",
  "entities": [
    {"label": "NOMBRE", "value": "María López",  "start_char": 10, "end_char": 21, "span_verified": true},
    {"label": "DNI",    "value": "12345678A",     "start_char": 35, "end_char": 44, "span_verified": true}
  ],
  "locale": "es_ES",
  "domain": "atencion_cliente",
  "source": "pilot_voice",
  "text_hash": "a1b2c3d4e5f6a7b8"
}
```

### Audio (`pilot_metadata.jsonl`)
```json
{
  "sample_id": "pilot_sample_001",
  "text": "Mi nombre es Laura García y resido en Madrid.",
  "audio_duration_seconds": 3.456,
  "original_audio_path": "data/pilot_dataset/original_wavs/pilot_sample_001.wav",
  "censored_audio_path":  "data/pilot_dataset/censored_wavs/pilot_sample_001_censored.wav",
  "pii_annotations": [
    {"label": "NOMBRE", "text_segment": "Laura García", "start_time": 0.82, "end_time": 1.54}
  ]
}
```

### Documento OCR (`annotations.jsonl`)
```json
{
  "sample_id": "uuid-...",
  "image_path": "data/ocr_dataset/images/formulario_0001.png",
  "doc_type": "formulario",
  "width_px": 794, "height_px": 1123,
  "bounding_boxes": [
    {"label": "NOMBRE",   "value": "Ana Ruiz", "x": 56, "y": 104, "width": 80, "height": 12},
    {"label": "DNI",      "value": "87654321B", "x": 56, "y": 154, "width": 86, "height": 12},
    {"label": "TELEFONO", "value": "+34 600 123 456", "x": 411, "y": 104, "width": 127, "height": 12}
  ]
}
```

---

## 🏆 Benchmark y tabla de clasificación

### Añadir un modelo

1. Genera un archivo `predictions.jsonl` con el formato:
   ```json
   {"id": "sample_001", "entities": [{"label": "NOMBRE", "value": "Ana Ruiz"}]}
   ```

2. Puntúalo:
   ```bash
   python benchmark.py --score predictions.jsonl --model-name "mi_modelo_v1" --notes "BERT fine-tuned"
   ```

3. La tabla de clasificación se actualiza automáticamente en `data/benchmark/leaderboard.jsonl`.

### Métricas
| Métrica | Descripción |
|---|---|
| **Precision** | De lo que el modelo predijo, ¿cuánto era correcto? |
| **Recall** | De todo lo que había, ¿cuánto encontró el modelo? |
| **F1 micro-avg** | Media armónica global (métrica principal del ranking) |

> ⚠️ El split `eval.jsonl` está **congelado** — jamás debe regenerarse para garantizar comparaciones justas.

---

## 🧪 Pruebas

```bash
python -m pytest tests/ -v
```

---

## 🏷️ Etiquetas PII soportadas

| Etiqueta | Ejemplo |
|---|---|
| `NOMBRE` | "María López" |
| `DNI` | "12345678A" |
| `TELEFONO` | "+34 600 123 456" |
| `EMAIL` | "usuario@ejemplo.com" |
| `DIRECCION` | "Calle Mayor 5, Madrid" |
| `FECHA` | "15/10/1985" |
| `EMPRESA` | "Acme S.L." |
| `TARJETA` | "4111 1111 1111 1111" |
| `CUENTA` | "ES12 3456 7890 1234 5678" |
| `PASAPORTE` | "AAA123456" |

---

## 📜 Licencia

Datos sintéticos generados con [Faker](https://faker.readthedocs.io/). Ningún dato real de personas fue utilizado.
