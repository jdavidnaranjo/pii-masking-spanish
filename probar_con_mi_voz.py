"""
Script interactivo para probar el pipeline con TU PROPIA VOZ.
Permite censurar MÚLTIPLES SECRETOS (nombre, ciudad, teléfono, correo, etc.)
en una sola grabación de voz.
"""

import os
import sys
import argparse
import subprocess
from typing import List
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.forced_aligner import MMSForcedAligner
from src.audio_censor import AudioPIICensor
from schemas.pii_models import PIISpan, SyntheticTextSample, TimeInterval


def grabar_microfono(output_path: str, duracion_segundos: int = 10):
    """Graba audio desde el micrófono usando arecord."""
    print(f"\n🎤 Preparando grabación de {duracion_segundos} segundos...")
    input("👉 Presiona ENTER en la terminal cuando estés listo para empezar a hablar: ")
    print(f"🔴 ¡HABLA AHORA! (Tienes {duracion_segundos} segundos)...")
    
    cmd = [
        "arecord",
        "-d", str(duracion_segundos),
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print("⏹️ Grabación finalizada.")


def procesar_voz(audio_path: str, texto_dicho: str, secretos_str: str):
    if not os.path.exists(audio_path):
        print(f"❌ Error: El archivo '{audio_path}' no existe.")
        return

    # Extraer la lista de secretos separados por coma
    lista_secretos = [s.strip() for s in secretos_str.split(",") if s.strip()]
    if not lista_secretos:
        print("❌ Error: No indicaste ningún secreto para censurar.")
        return

    # Buscar cada secreto en el texto y construir los PIISpan
    spans: List[PIISpan] = []
    for sec in lista_secretos:
        idx = texto_dicho.lower().find(sec.lower())
        if idx == -1:
            print(f"⚠️ Aviso: La palabra '{sec}' no se encontró en el texto. Se ignorará.")
            continue
        valor_real = texto_dicho[idx:idx + len(sec)]
        spans.append(PIISpan(
            label="SECRETO",
            value=valor_real,
            start_char=idx,
            end_char=idx + len(sec)
        ))

    if not spans:
        print("❌ Ninguno de los secretos indicados coincide con el texto.")
        return

    print("\n" + "=" * 75)
    print("🎧 PROCESANDO TU VOZ CON INTELIGENCIA ARTIFICIAL (PYTORCH MMS_FA)")
    print("=" * 75)
    print(f"  📝 Frase completa : \"{texto_dicho}\"")
    print(f"  🔒 Secretos a censurar ({len(spans)}): {[s.value for s in spans]}")

    # 2. Cargar audio
    audio_data, sample_rate = sf.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duracion = len(audio_data) / sample_rate
    print(f"  🔊 Duración del audio : {duracion:.2f} segundos")

    sample = SyntheticTextSample(
        text=texto_dicho,
        entities=spans
    )

    # 3. Alineación Forzada con PyTorch MMS_FA
    print("\n[Paso 1] La IA está escuchando tu voz y cronometrando todas las palabras...")
    aligner = MMSForcedAligner()
    palabras = aligner.align_words(audio_data, sample_rate, texto_dicho)

    print(f"  🎯 Total de palabras reconocidas en tu voz: {len(palabras)}")
    for p in palabras:
        print(f"     • {p.word:<18} [{p.start_time:.2f}s -> {p.end_time:.2f}s] (conf: {p.confidence_score:.2f})")

    # Mapear los segundos exactos de cada secreto
    intervalos = aligner.align_pii_entities(audio_data, sample_rate, sample)
    if not intervalos:
        print("⚠️ No se pudieron alinear los secretos con suficiente precisión.")
        return

    print(f"\n  🔒 MARCAS DE TIEMPO EXACTAS DE CADA SECRETO:")
    for ti in intervalos:
        print(f"     • '{ti.text_segment}' suena entre: {ti.start_time:.2f}s y {ti.end_time:.2f}s")

    # 4. Aplicar censura con BIP en cada secreto
    print(f"\n[Paso 2] Aplicando {len(intervalos)} BIPS de censura sobre tu voz...")
    censor = AudioPIICensor(beep_frequency=1000.0, beep_volume=0.35, fade_duration_ms=6.0)
    audio_censurado = censor.censor_audio(audio_data, sample_rate, intervalos, mode="beep")

    output_censurado = "data/mi_voz_censurada.wav"
    censor.save_wav(output_censurado, audio_censurado, sample_rate)
    print(f"  💾 Audio con todos los BIPS guardado en: {output_censurado}")

    # Línea de tiempo ASCII
    print("\n[Paso 3] Línea de tiempo visual de tu audio:")
    items = [(ti.start_time, ti.end_time, ti.label) for ti in intervalos]
    print(censor.render_ascii_timeline(duracion, items))

    print("\n" + "=" * 75)
    print("✨ ¡LISTO! Para escuchar tu voz con todos los BIPS tapando tus datos, ejecuta:")
    print(f"   aplay {output_censurado}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Prueba con tu voz censurando múltiples secretos.")
    parser.add_argument("--audio", type=str, default="", help="Ruta al archivo WAV (opcional)")
    parser.add_argument("--texto", type=str, default="", help="Texto exacto que dijiste")
    parser.add_argument("--secretos", type=str, default="", help="Secretos a tapar separados por coma (ej: 'David, Quito')")
    parser.add_argument("--duracion", type=int, default=12, help="Segundos de grabación (por defecto 12)")

    args = parser.parse_args()

    audio_file = args.audio
    texto = args.texto
    secretos = args.secretos
    duracion = args.duracion

    os.makedirs("data", exist_ok=True)

    if not audio_file:
        print("=" * 75)
        print("🎙️ PROBADOR DE VOZ: CENSURA MÚLTIPLES SECRETOS EN UN SOLO AUDIO")
        print("=" * 75)

        frase_sugerida = "Hola, mi nombre es David, resido en Quito y mi correo es david arroba gmail punto com."
        secretos_sugeridos = "David, Quito, david arroba gmail punto com"

        print(f"\n💡 Frase de ejemplo sugerida con múltiples datos:")
        print(f"   \"{frase_sugerida}\"")

        texto_in = input(f"\n1. Escribe la frase completa que vas a decir [ENTER para la sugerida]: ").strip()
        texto = texto_in if texto_in else frase_sugerida

        print(f"\n2. Escribe qué palabras quieres tapar con BIP separadas por comas (ej: David, Quito):")
        secretos_in = input(f"   Secretos [ENTER para: '{secretos_sugeridos}']: ").strip()
        secretos = secretos_in if secretos_in else secretos_sugeridos

        duracion_in = input(f"\n3. ¿Cuántos segundos quieres para hablar? [ENTER para 12 segundos]: ").strip()
        if duracion_in.isdigit():
            duracion = int(duracion_in)

        audio_file = "data/mi_voz.wav"
        grabar_microfono(audio_file, duracion_segundos=duracion)

    procesar_voz(audio_file, texto, secretos)


if __name__ == "__main__":
    main()
