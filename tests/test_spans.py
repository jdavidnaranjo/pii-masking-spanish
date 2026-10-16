import unittest
import sys
import os

# Asegurar que la ruta raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import PIIDataGenerator

class TestPIISpans(unittest.TestCase):
    def setUp(self):
        self.generator = PIIDataGenerator(seed=123)

    def test_single_sample_span_alignment(self):
        """Verifica que el slicing directo de start_char:end_char coincida exactamente con la entidad."""
        sample = self.generator.generate_sample()
        self.assertGreater(len(sample.entities), 0)
        for ent in sample.entities:
            extracted = sample.text[ent.start_char:ent.end_char]
            self.assertEqual(
                extracted, ent.value,
                f"Desfase detectado en {ent.label}: esperado '{ent.value}', obtenido '{extracted}'"
            )

    def test_custom_template_with_multiple_pii(self):
        """Verifica plantillas con múltiples ocurrencias y entidades contiguas."""
        custom_template = "Hola {{NOMBRE}}, su DNI es {{DNI}} y su email es {{EMAIL}}."
        sample = self.generator.generate_sample(template=custom_template)
        
        self.assertEqual(len(sample.entities), 3)
        for ent in sample.entities:
            self.assertTrue(ent.validate_against_text(sample.text))

    def test_batch_generation_integrity(self):
        """Verifica la integridad de un lote de 20 muestras generadas."""
        batch = self.generator.generate_batch(num_samples=20)
        self.assertEqual(len(batch), 20)
        for sample in batch:
            for ent in sample.entities:
                self.assertEqual(sample.text[ent.start_char:ent.end_char], ent.value)

if __name__ == "__main__":
    unittest.main()
