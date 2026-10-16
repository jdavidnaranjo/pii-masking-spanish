"""
Demostración de Pipeline Completo con LLM (Ollama / Llama 3.2 Local) + PyTorch MMS_FA:
1. Generación de guion conversacional realista con Ollama (Llama 3.2 local, 100% gratis y privado).
2. Extracción y validación automática de PII con Pydantic.
3. Síntesis de voz humana (TTS) en archivo WAV.
4. Alineación Forzada a nivel de palabra con PyTorch TorchAudio MMS_FA.
5. Censura acústica con tono 'BIP' (1000 Hz) y visualización en tiempo real.
"""

import os
import io
import sys
import argparse
import soundfile as sf
from gtts import gTTS

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.llm_generator import LLMPIIDataGenerator
from src.forced_aligner import MMSForcedAligner
from src.audio_censor import AudioPIICensor


def run_llm_demo(provider: str = "ollama", domain: str = "banca"):
    if provider == "ollama":
        provider_name = "Ollama (Llama 3.2 Local - 100% Privado y Gratis)"
    elif provider == "gemini":
        provider_name = "Google Gemini"
    else:
        provider_name = "OpenAI (GPT-4o mini)"

    print("=" * 75)
    print(f"🤖 PIPELINE AVANZADO: {provider_name.upper()} + PYTORCH MMS_FA")
    print("=" * 75)

    # 1. Generar el guion con el LLM
    print(f"\n[Paso 1] Pidiéndole a {provider_name} que redacte una conversación real de '{domain}'...")
    generator = LLMPIIDataGenerator(provider=provider)
    sample = generator.generate_dialogue(domain=domain)

    print(f"\n📝 Diálogo generado por {provider_name}:")
    print(f"\"{sample.text}\"")

    print(f"\n🔍 Datos personales (PII) detectados por el modelo:")
    for ent in sample.entities:
        print(f"  • {ent.label:<15} [{ent.start_char:3d}..{ent.end_char:3d}] -> \"{ent.value}\"")

    # 2. Sintetizar la voz con TTS
    print("\n[Paso 2] Sintetizando la voz humana del guion con TTS...")
    tts = gTTS(text=sample.text, lang="es", slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_data, sr = sf.read(fp)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration = len(audio_data) / sr
    print(f"  🔊 Audio generado con éxito. Duración: {duration:.2f} segundos.")

    os.makedirs("data", exist_ok=True)
    censor = AudioPIICensor(beep_frequency=1000.0, beep_volume=0.35, fade_duration_ms=6.0)
    orig_wav = "data/llm_voz_original.wav"
    censor.save_wav(orig_wav, audio_data, sr)
    print(f"  💾 Audio ORIGINAL guardado en: {orig_wav}")

    # 3. Alineación Forzada con PyTorch MMS_FA
    print("\n[Paso 3] Ejecutando Alineación Forzada palabra por palabra con PyTorch MMS_FA...")
    aligner = MMSForcedAligner()
    words = aligner.align_words(audio_data, sr, sample.text)

    print(f"  🎯 Palabras detectadas: {len(words)}")
    for w in words[:8]:
        print(f"     • {w.word:<15} [{w.start_time:.2f}s -> {w.end_time:.2f}s] conf: {w.confidence_score:.2f}")
    if len(words) > 8:
        print(f"     ... y {len(words)-8} palabras más.")

    # Mapear los secretos a intervalos de tiempo exactos
    intervals = aligner.align_pii_entities(audio_data, sr, sample)
    print("\n  🔒 Marcas de tiempo de los datos privados a censurar:")
    for ti in intervals:
        print(f"     • {ti.label:<12} [{ti.start_time:.2f}s -> {ti.end_time:.2f}s] ('{ti.text_segment}')")

    # 4. Censura con BIP
    print("\n[Paso 4] Aplicando el BIP de censura (1000 Hz) sobre el audio...")
    censored_audio = censor.censor_audio(audio_data, sr, intervals, mode="beep")
    cens_wav = "data/llm_voz_censurada_con_bip.wav"
    censor.save_wav(cens_wav, censored_audio, sr)
    print(f"  💾 Audio CENSURADO guardado en: {cens_wav}")

    # Línea de tiempo visual
    print("\n[Paso 5] Línea de tiempo de la censura:")
    timeline_items = [(ti.start_time, ti.end_time, ti.label) for ti in intervals]
    print(censor.render_ascii_timeline(duration, timeline_items))

    print("\n" + "=" * 75)
    print("✨ ¡PIPELINE LLM OLLAMA + PYTORCH COMPLETADO CON ÉXITO!")
    print("Puedes escuchar los audios generados con:")
    print(f"   Original  : aplay {orig_wav}")
    print(f"   Censurado : aplay {cens_wav}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de generación con LLM + PyTorch MMS_FA.")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "gemini", "openai"], help="Proveedor de LLM")
    parser.add_argument("--domain", type=str, default="banca", help="Dominio de la conversación (banca, salud, atencion_cliente)")

    args = parser.parse_args()
    run_llm_demo(provider=args.provider, domain=args.domain)


if __name__ == "__main__":
    main()
