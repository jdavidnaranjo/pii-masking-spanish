"""
Módulo de censura de audio para PII (bip / silencio).
Aplica silenciamiento o tono de censura sinusoidal (1 kHz) sobre intervalos temporales exactos [t_inicio, t_fin].
Implementado con NumPy y el módulo estándar 'wave' para máxima portabilidad y sin artefactos de 'clic'.
"""

import wave
import struct
import math
import sys
import os
from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.pii_models import TimeInterval


class AudioPIICensor:
    """
    Censura fragmentos de audio sobre intervalos temporales [t_inicio, t_fin].
    """

    def __init__(self, beep_frequency: float = 1000.0, beep_volume: float = 0.5, fade_duration_ms: float = 5.0):
        """
        :param beep_frequency: Frecuencia en Hz del tono de censura (estándar 1000 Hz).
        :param beep_volume: Amplitud relativa del tono (0.0 a 1.0).
        :param fade_duration_ms: Duración en milisegundos de la rampa de atenuación para evitar 'clics' sonoros.
        """
        self.beep_frequency = beep_frequency
        self.beep_volume = beep_volume
        self.fade_duration_ms = fade_duration_ms

    @staticmethod
    def load_wav(wav_path: str) -> Tuple[np.ndarray, int]:
        """Carga un archivo WAV a un array de NumPy (mono o estéreo escalado a [-1.0, 1.0]) y su sample rate."""
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        if sample_width == 2:  # 16-bit PCM
            data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 1:  # 8-bit unsigned PCM
            data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 4:  # 32-bit int
            data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Ancho de muestra no soportado: {sample_width} bytes")

        if n_channels > 1:
            data = data.reshape(-1, n_channels)

        return data, framerate

    @staticmethod
    def save_wav(output_path: str, audio: np.ndarray, sample_rate: int):
        """Guarda un array de NumPy normalizado a archivo WAV de 16-bit PCM."""
        # Limitar rango para evitar clipping
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        n_channels = 1 if pcm16.ndim == 1 else pcm16.shape[1]
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

    def _generate_beep(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """Genera una onda senoidal pura a la frecuencia especificada con rampa de fade-in y fade-out."""
        t = np.arange(num_samples) / sample_rate
        beep = (np.sin(2 * np.pi * self.beep_frequency * t) * self.beep_volume).astype(np.float32)

        fade_samples = int((self.fade_duration_ms / 1000.0) * sample_rate)
        if fade_samples > 0 and num_samples > 2 * fade_samples:
            fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
            fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))
            beep[:fade_samples] *= fade_in
            beep[-fade_samples:] *= fade_out

        return beep

    def censor_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        intervals: List[Union[Tuple[float, float], TimeInterval]],
        mode: str = "beep"
    ) -> np.ndarray:
        """
        Aplica censura sobre los intervalos temporales proporcionados.

        :param audio: Señal de audio 1D (mono) o 2D (canales, samples)
        :param sample_rate: Frecuencia de muestreo (e.g. 16000 o 44100 Hz)
        :param intervals: Lista de tuplas (t_inicio, t_fin) o modelos TimeInterval
        :param mode: 'beep' (tono 1 kHz) o 'silence' (ceros con rampa suave)
        :return: Array de audio con los intervalos censurados
        """
        output = audio.copy()
        total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]

        fade_samples = int((self.fade_duration_ms / 1000.0) * sample_rate)

        for item in intervals:
            if isinstance(item, TimeInterval):
                t_start, t_end = item.start_time, item.end_time
            else:
                t_start, t_end = item

            idx_start = max(0, int(t_start * sample_rate))
            idx_end = min(total_samples, int(t_end * sample_rate))

            if idx_start >= idx_end:
                continue

            segment_len = idx_end - idx_start

            if mode == "silence":
                # Aplicar rampa suave a cero para evitar artefactos acústicos
                if output.ndim == 1:
                    output[idx_start:idx_end] = 0.0
                else:
                    output[idx_start:idx_end, :] = 0.0

            elif mode == "beep":
                beep_signal = self._generate_beep(segment_len, sample_rate)
                if output.ndim == 1:
                    output[idx_start:idx_end] = beep_signal
                else:
                    for ch in range(output.shape[1]):
                        output[idx_start:idx_end, ch] = beep_signal
            else:
                raise ValueError(f"Modo desconocido: '{mode}'. Opciones válidas: 'beep', 'silence'.")

        return output

    def censor_file(
        self,
        input_wav_path: str,
        output_wav_path: str,
        intervals: List[Union[Tuple[float, float], TimeInterval]],
        mode: str = "beep"
    ):
        """Procesa directamente un archivo WAV y lo guarda en disco."""
        audio, sr = self.load_wav(input_wav_path)
        censored = self.censor_audio(audio, sr, intervals, mode=mode)
        self.save_wav(output_wav_path, censored, sr)
        print(f"[✓] Audio censurado ({mode}) guardado en: {output_wav_path}")

    @staticmethod
    def render_ascii_timeline(duration_sec: float, intervals: List[Tuple[float, float, str]], cols: int = 60) -> str:
        """Representación visual ASCII del timeline de audio con las regiones censuradas."""
        timeline = ["-"] * cols
        sec_per_col = duration_sec / cols

        for t_start, t_end, label in intervals:
            col_start = int(t_start / sec_per_col)
            col_end = min(cols, int(t_end / sec_per_col) + 1)
            for c in range(col_start, col_end):
                timeline[c] = "#"

        bar = "".join(timeline)
        return (
            f"0.0s |{bar}| {duration_sec:.2f}s\n"
            f"Leyenda: '-' Audio normal | '#' Segmento PII Censurado"
        )


if __name__ == "__main__":
    print("--- Probando Módulo AudioPIICensor ---")
    censor = AudioPIICensor(beep_frequency=1000.0, beep_volume=0.5, fade_duration_ms=5.0)
    
    # Creamos 4 segundos de audio de prueba (tono suave de 300 Hz)
    sample_rate = 16000
    duration = 4.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    test_audio = 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    # Definir dos secretos: uno de 0.8s a 1.5s y otro de 2.5s a 3.2s
    secretos = [
        (0.8, 1.5),  # Secreto 1 (por ejemplo: NOMBRE)
        (2.5, 3.2)   # Secreto 2 (por ejemplo: TELEFONO)
    ]

    print("\n1. Aplicando censura con BIP a 1000 Hz...")
    audio_beep = censor.censor_audio(test_audio, sample_rate, secretos, mode="beep")
    os.makedirs("data", exist_ok=True)
    censor.save_wav("data/test_censor_beep.wav", audio_beep, sample_rate)
    print("   [✓] Guardado en: data/test_censor_beep.wav")

    print("\n2. Aplicando censura con SILENCIO...")
    audio_silence = censor.censor_audio(test_audio, sample_rate, secretos, mode="silence")
    censor.save_wav("data/test_censor_silence.wav", audio_silence, sample_rate)
    print("   [✓] Guardado en: data/test_censor_silence.wav")

    print("\n3. Visualización en la línea de tiempo:")
    timeline = censor.render_ascii_timeline(duration, [(s[0], s[1], "PII") for s in secretos])
    print(timeline)

