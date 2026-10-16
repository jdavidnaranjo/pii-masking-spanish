"""
Generador del Conjunto de Datos Piloto de Voz para Enmascaramiento de PII.
Cumple con la especificación técnica de la vacante:
- Genera audio a partir de texto sintético en español (TTS).
- Realiza alineación forzada a nivel de palabra con PyTorch (TorchAudio MMS_FA).
- Exporta archivos WAV (originales y censurados con bip) y un catálogo de metadatos en JSONL.
"""

import os
import io
import json
import soundfile as sf
from gtts import gTTS

from src.data_generator import PIIDataGenerator
from src.forced_aligner import MMSForcedAligner
from src.audio_censor import AudioPIICensor
from schemas.pii_models import TimeInterval


def build_voice_pilot_dataset(num_samples: int = 5, output_dir: str = "data/pilot_dataset"):
    print("=" * 75)
    print("🚀 GENERADOR DE CONJUNTO DE DATOS PILOTO (PII VOICE DATASET - PYTORCH MMS_FA)")
    print("=" * 75)

    orig_dir = os.path.join(output_dir, "original_wavs")
    cens_dir = os.path.join(output_dir, "censored_wavs")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(cens_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "pilot_metadata.jsonl")

    # Inicializar generadores y modelos
    generator = PIIDataGenerator(locale="es_ES", seed=100)
    aligner = MMSForcedAligner()
    censor = AudioPIICensor(beep_frequency=1000.0, beep_volume=0.35, fade_duration_ms=6.0)

    # Plantillas variadas de alta claridad acústica para el piloto
    templates = [
        "Hola, buenas tardes. Mi nombre es {{NOMBRE}} y resido en la ciudad de {{DIRECCION}}.",
        "Le atiende el asesor {{NOMBRE}}. Para validar su solicitud indique su documento {{DNI}}.",
        "Confirmo la entrega a nombre de {{NOMBRE}} en el domicilio {{DIRECCION}}.",
        "Buenas noches, soy {{NOMBRE}} y me comunico para reportar una incidencia.",
        "Estimado doctor, solicito la consulta médica para el paciente {{NOMBRE}}."
    ]

    records = []

    for i in range(num_samples):
        sample_id = f"pilot_sample_{i+1:03d}"
        tpl = templates[i % len(templates)]
        sample = generator.generate_sample(template=tpl)

        print(f"\n[{i+1}/{num_samples}] Procesando {sample_id}...")
        print(f"  📝 Texto: \"{sample.text}\"")

        # 1. Síntesis TTS
        tts = gTTS(text=sample.text, lang="es", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_data, sr = sf.read(fp)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        duration = len(audio_data) / sr

        # 2. Guardar WAV Original
        orig_file = os.path.join(orig_dir, f"{sample_id}.wav")
        censor.save_wav(orig_file, audio_data, sr)

        # 3. Alineación Forzada con PyTorch (Nivel de palabra y PII)
        aligned_words = aligner.align_words(audio_data, sr, sample.text)
        pii_intervals = aligner.align_pii_entities(audio_data, sr, sample)

        # 4. Censura con Bip
        censored_audio = censor.censor_audio(audio_data, sr, pii_intervals, mode="beep")
        cens_file = os.path.join(cens_dir, f"{sample_id}_censored.wav")
        censor.save_wav(cens_file, censored_audio, sr)

        # 5. Estructurar registro de metadatos
        record = {
            "sample_id": sample_id,
            "text": sample.text,
            "audio_duration_seconds": round(duration, 3),
            "original_audio_path": orig_file,
            "censored_audio_path": cens_file,
            "pii_annotations": [
                {
                    "label": ti.label,
                    "text_segment": ti.text_segment,
                    "start_time": ti.start_time,
                    "end_time": ti.end_time
                }
                for ti in pii_intervals
            ],
            "word_level_timestamps": [
                {
                    "word": w.word,
                    "start_char": w.start_char,
                    "end_char": w.end_char,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence_score": w.confidence_score
                }
                for w in aligned_words
            ]
        }
        records.append(record)
        print(f"  ✅ Alineadas {len(aligned_words)} palabras y {len(pii_intervals)} entidades PII con éxito.")

    # Guardar archivo JSONL de metadatos
    with open(metadata_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 75)
    print("✨ CONJUNTO DE DATOS PILOTO GENERADO CON ÉXITO")
    print(f"📁 Directorio de salida: '{output_dir}/'")
    print(f"  • Metadatos y marcas de tiempo: {metadata_path}")
    print(f"  • Audios originales ({num_samples}): {orig_dir}/")
    print(f"  • Audios censurados ({num_samples}): {cens_dir}/")
    print("=" * 75)


if __name__ == "__main__":
    build_voice_pilot_dataset(num_samples=5)
