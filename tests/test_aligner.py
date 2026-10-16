import unittest
import numpy as np
from src.forced_aligner import MMSForcedAligner
from schemas.pii_models import SyntheticTextSample, PIISpan


class TestMMSForcedAligner(unittest.TestCase):

    def setUp(self):
        self.aligner = MMSForcedAligner()

    def test_normalize_word(self):
        # Comprobar eliminación de acentos y caracteres raros
        self.assertEqual(self.aligner.normalize_word("Ramón"), "ramon")
        self.assertEqual(self.aligner.normalize_word("Díaz,"), "diaz")
        self.assertEqual(self.aligner.normalize_word("¡Hola!"), "hola")
        self.assertEqual(self.aligner.normalize_word("Pérez-Gómez"), "perezgomez")

    def test_align_empty_text(self):
        audio = np.zeros(16000, dtype=np.float32)
        words = self.aligner.align_words(audio, 16000, "")
        self.assertEqual(words, [])

    def test_align_pii_entities_empty(self):
        audio = np.zeros(16000, dtype=np.float32)
        sample = SyntheticTextSample(text="", entities=[])
        intervals = self.aligner.align_pii_entities(audio, 16000, sample)
        self.assertEqual(intervals, [])


if __name__ == "__main__":
    unittest.main()
