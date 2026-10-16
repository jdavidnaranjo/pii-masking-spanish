import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_censor import AudioPIICensor
from schemas.pii_models import TimeInterval

class TestAudioCensor(unittest.TestCase):
    def setUp(self):
        self.censor = AudioPIICensor(beep_frequency=1000.0, beep_volume=0.5)
        self.sample_rate = 16000
        # 3 segundos de audio sintético (tono de 440 Hz)
        t = np.arange(self.sample_rate * 3) / self.sample_rate
        self.dummy_audio = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    def test_silence_mode(self):
        """Verifica que el modo silence ponga a 0.0 el intervalo especificado."""
        interval = (1.0, 2.0)
        censored = self.censor.censor_audio(self.dummy_audio, self.sample_rate, [interval], mode="silence")

        start_idx = int(1.0 * self.sample_rate)
        end_idx = int(2.0 * self.sample_rate)

        # En el centro del intervalo debe ser exactamente 0.0
        mid_idx = (start_idx + end_idx) // 2
        self.assertEqual(censored[mid_idx], 0.0)
        # Fuera del intervalo el audio original debe conservarse
        self.assertAlmostEqual(censored[100], self.dummy_audio[100], places=5)

    def test_beep_mode(self):
        """Verifica que el modo beep reemplace la señal por el tono."""
        interval = TimeInterval(start_time=0.5, end_time=1.5, label="NOMBRE", text_segment="Carlos")
        censored = self.censor.censor_audio(self.dummy_audio, self.sample_rate, [interval], mode="beep")

        start_idx = int(0.5 * self.sample_rate)
        end_idx = int(1.5 * self.sample_rate)
        # El segmento reemplazado no debe coincidir con el audio original de 440Hz
        self.assertNotEqual(censored[start_idx + 100], self.dummy_audio[start_idx + 100])

if __name__ == "__main__":
    unittest.main()
