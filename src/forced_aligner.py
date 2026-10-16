"""
Módulo de Alineación Forzada (Forced Alignment) con PyTorch y TorchAudio MMS_FA.
Permite calcular marcas de tiempo exactas a nivel de palabra [t_inicio, t_fin]
para datos sensibles (PII) a partir del audio generado por TTS.
"""

import os
import re
import sys
import unicodedata
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schemas.pii_models import PIISpan, SyntheticTextSample, TimeInterval


@dataclass
class WordTimestamp:
    """Representa una palabra alineada en el audio con sus marcas de tiempo."""
    word: str
    start_char: int
    end_char: int
    start_time: float
    end_time: float
    confidence_score: float


class MMSForcedAligner:
    """
    Alineador forzado basado en el modelo MMS_FA de Meta / TorchAudio.
    Mapea texto en español y audio para obtener marcas de tiempo a nivel de palabra.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = MMS_FA
        self.sample_rate = self.bundle.sample_rate
        self._model = None
        self._tokenizer = None
        self._aligner = None

    def _load_model(self):
        """Carga perezosa del modelo para optimizar el inicio."""
        if self._model is None:
            self._model = self.bundle.get_model().to(self.device)
            self._model.eval()
            self._tokenizer = self.bundle.get_tokenizer()
            self._aligner = self.bundle.get_aligner()

    @staticmethod
    def normalize_word(word: str) -> str:
        """
        Normaliza palabras eliminando tildes y caracteres no alfabéticos
        para coincidir con el vocabulario del alineador MMS_FA (a-z).
        """
        nfkd = unicodedata.normalize("NFKD", word)
        ascii_word = "".join(c for c in nfkd if not unicodedata.combining(c))
        return re.sub(r"[^a-z]", "", ascii_word.lower())

    def align_words(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        text: str
    ) -> List[WordTimestamp]:
        """
        Alinea cada palabra del texto sobre la señal de audio.
        Retorna la lista de palabras con sus segundos exactos y límites de caracteres.
        """
        self._load_model()

        # Preparar tensor de audio (mono, 16kHz)
        waveform = torch.from_numpy(audio_array).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.t().mean(dim=0, keepdim=True)
        elif waveform.ndim == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)

        # Extraer palabras y sus offsets de caracteres originales en el texto
        raw_words = []
        for match in re.finditer(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9]+", text):
            cleaned = self.normalize_word(match.group())
            if cleaned:  # Solo palabras con letras representables
                raw_words.append({
                    "raw": match.group(),
                    "cleaned": cleaned,
                    "start_char": match.start(),
                    "end_char": match.end(),
                })

        if not raw_words:
            return []

        cleaned_tokens = [w["cleaned"] for w in raw_words]

        # Inferencia con TorchAudio MMS_FA
        with torch.inference_mode():
            emission, _ = self._model(waveform)

        tokens = self._tokenizer(cleaned_tokens)
        spans = self._aligner(emission[0], tokens)

        # Conversión de frames del modelo a segundos
        num_frames = emission.size(1)
        ratio = waveform.size(1) / num_frames / self.sample_rate

        aligned_words: List[WordTimestamp] = []
        for word_dict, span in zip(raw_words, spans):
            if not span:
                continue
            t_start = round(float(span[0].start * ratio), 3)
            t_end = round(float(span[-1].end * ratio), 3)
            score = round(float(span[0].score), 3)

            aligned_words.append(WordTimestamp(
                word=word_dict["raw"],
                start_char=word_dict["start_char"],
                end_char=word_dict["end_char"],
                start_time=t_start,
                end_time=t_end,
                confidence_score=score
            ))

        return aligned_words

    def align_pii_entities(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        sample: SyntheticTextSample
    ) -> List[TimeInterval]:
        """
        Alinea entidades PII etiquetadas con el audio a nivel de palabra.
        Calcula el intervalo temporal [t_start, t_end] de cada entidad PII.
        """
        aligned_words = self.align_words(audio_array, sample_rate, sample.text)
        if not aligned_words:
            return []

        time_intervals: List[TimeInterval] = []

        for entity in sample.entities:
            # Buscar las palabras que caen dentro o intersectan el span del PII
            matching_words = [
                w for w in aligned_words
                if not (w.end_char <= entity.start_char or w.start_char >= entity.end_char)
            ]

            if matching_words:
                t_start = matching_words[0].start_time
                t_end = matching_words[-1].end_time

                # Margen de seguridad leve (30ms) para una censura acústica impecable
                duration = len(audio_array) / sample_rate
                t_start = max(0.0, round(t_start - 0.03, 3))
                t_end = min(round(duration, 3), round(t_end + 0.03, 3))

                time_intervals.append(TimeInterval(
                    start_time=t_start,
                    end_time=t_end,
                    label=entity.label,
                    text_segment=entity.value
                ))

        return time_intervals
