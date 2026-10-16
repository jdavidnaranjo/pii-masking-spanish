"""
Generador sintético de texto en español con entidades PII y spans exactos de caracteres.
Diseñado para alimentar pipelines de Audio TTS y Forced Alignment (MMS_FA).
"""

import re
import json
import random
import sys
import os
from typing import List, Dict, Optional, Callable
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path para importar schemas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from faker import Faker
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False

from schemas.pii_models import PIISpan, SyntheticTextSample


class PIIDataGenerator:
    """
    Genera oraciones sintéticas en español inyectando entidades PII y calculando
    de forma determinista los límites de caracteres (start_char, end_char).
    """

    DEFAULT_TEMPLATES = [
        # Atención al cliente / Soporte
        "Hola, muy buenos días. Mi nombre es {{NOMBRE}} y me comunico para reportar un problema con mi servicio.",
        "Buenas tardes, le habla {{NOMBRE}}. Desearía verificar el estado de mi solicitud asociada al correo {{EMAIL}}.",
        "Para proceder con el trámite, indico mi documento de identidad {{DNI}} y mi número móvil {{TELEFONO}}.",
        "Por favor confirmar si la factura puede ser enviada a la dirección {{DIRECCION}} a nombre de {{NOMBRE}}.",
        
        # Banca y finanzas
        "Quisiera bloquear mi tarjeta de crédito {{TARJETA}} por un cobro sospechoso realizado hoy a nombre de {{NOMBRE}}.",
        "Confirmo la transferencia bancaria con destino a la cuenta {{IBAN}} perteneciente a {{NOMBRE}} con DNI {{DNI}}.",
        "Mi nombre es {{NOMBRE}}, nací el {{FECHA_NACIMIENTO}} y mi número telefónico de contacto es {{TELEFONO}}.",
        "Necesito asociar mi nuevo correo electrónico {{EMAIL}} a la cuenta bancaria terminada en {{IBAN}}.",
        
        # Salud y citas médicas
        "Buen día, solicito una cita médica para el paciente {{NOMBRE}}, identificado con documento {{DNI}}.",
        "El doctor puede contactarme al teléfono {{TELEFONO}} o enviarme los resultados clínicos al correo {{EMAIL}}.",
        "Por cambio de residencia, solicito actualizar mi domicilio a {{DIRECCION}}. El titular de la póliza es {{NOMBRE}}.",
        
        # Servicios generales y verificación
        "Estimado asesor, le dicto mis datos de validación: {{NOMBRE}}, identificación {{DNI}} y teléfono {{TELEFONO}}.",
        "He recibido una notificación en mi email {{EMAIL}} sobre un movimiento de mi tarjeta {{TARJETA}}."
    ]

    def __init__(self, locale: str = "es_ES", seed: Optional[int] = None):
        self.locale = locale
        if seed is not None:
            random.seed(seed)
            if _HAS_FAKER:
                Faker.seed(seed)

        if _HAS_FAKER:
            self.faker = Faker(locale)
        else:
            self.faker = None

        # Mapeo de generadores por tipo de entidad
        self.entity_generators: Dict[str, Callable[[], str]] = {
            "NOMBRE": self._generate_name,
            "DNI": self._generate_dni,
            "TELEFONO": self._generate_phone,
            "EMAIL": self._generate_email,
            "DIRECCION": self._generate_address,
            "TARJETA": self._generate_credit_card,
            "FECHA_NACIMIENTO": self._generate_date,
            "IBAN": self._generate_iban,
        }

    # ==========================================
    # Generadores de entidades PII
    # ==========================================
    def _generate_name(self) -> str:
        if self.faker:
            return self.faker.name()
        nombres = ["Carlos Mendoza", "Sofía Morales", "Alejandro Pérez", "María Elena Rodríguez", "Javier Gómez"]
        return random.choice(nombres)

    def _generate_dni(self) -> str:
        if self.faker:
            # En Faker es_ES: dni() o nie()
            try:
                return self.faker.dni()
            except AttributeError:
                pass
        num = random.randint(10000000, 99999999)
        letras = "TRWAGMYFPDXBNJZSQVHLCKE"
        return f"{num}{letras[num % 23]}"

    def _generate_phone(self) -> str:
        if self.faker:
            phone = self.faker.phone_number()
            # Limpiar formatos exóticos para que sea natural en TTS
            return re.sub(r"[^\d\s\+]", "", phone).strip()
        prefijo = random.choice(["6", "7", "9"])
        resto = "".join(str(random.randint(0, 9)) for _ in range(8))
        return f"+34 {prefijo}{resto[:2]} {resto[2:5]} {resto[5:]}"

    def _generate_email(self) -> str:
        if self.faker:
            return self.faker.email()
        usuario = random.choice(["carlos.mendoza", "sofia.m", "mrodriguez24", "javier_gomez"])
        dominio = random.choice(["gmail.com", "outlook.es", "empresa.com", "protonmail.com"])
        return f"{usuario}@{dominio}"

    def _generate_address(self) -> str:
        if self.faker:
            # Simplificar a una sola línea sin saltos para texto fluido
            return self.faker.street_address()
        calles = ["Calle Mayor 45", "Avenida de la Constitución 12, 3ºB", "Paseo de la Castellana 120", "Gran Vía 18"]
        return random.choice(calles)

    def _generate_credit_card(self) -> str:
        if self.faker:
            return self.faker.credit_card_number(card_type=None)
        # 16 dígitos separados en grupos de 4 para lectura TTS
        digits = "".join(str(random.randint(0, 9)) for _ in range(16))
        return f"{digits[0:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"

    def _generate_date(self) -> str:
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                 "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        dia = random.randint(1, 28)
        mes = random.choice(meses)
        anio = random.randint(1965, 2004)
        return f"{dia} de {mes} de {anio}"

    def _generate_iban(self) -> str:
        if self.faker:
            try:
                return self.faker.iban()
            except AttributeError:
                pass
        codigo = "".join(str(random.randint(0, 9)) for _ in range(20))
        return f"ES{random.randint(10, 99)} {codigo[:4]} {codigo[4:8]} {codigo[8:12]} {codigo[12:16]} {codigo[16:20]}"

    # ==========================================
    # Motor de construcción de spans exactos
    # ==========================================
    def generate_sample(self, template: Optional[str] = None, domain: str = "general") -> SyntheticTextSample:
        """
        Interpola las entidades en la plantilla y calcula los spans sin desalineación.
        Utiliza un escaneo secuencial basado en regex para garantizar que duplicados
        o caracteres especiales no alteren los índices.
        """
        raw_template = template or random.choice(self.DEFAULT_TEMPLATES)
        
        # Patrón para identificar etiquetas como {{NOMBRE}}, {{DNI}}, etc.
        token_pattern = re.compile(r"\{\{([A-Z_]+)\}\}")
        
        output_text_parts = []
        entities: List[PIISpan] = []
        current_cursor = 0
        last_match_end = 0

        for match in token_pattern.finditer(raw_template):
            entity_type = match.group(1)
            static_prefix = raw_template[last_match_end:match.start()]
            
            # 1. Agregar fragmento de texto estático previo
            if static_prefix:
                output_text_parts.append(static_prefix)
                current_cursor += len(static_prefix)

            # 2. Generar el valor sensible correspondiente
            generator_fn = self.entity_generators.get(entity_type, lambda: f"[{entity_type}]")
            entity_value = generator_fn()

            # 3. Registrar los límites exactos del span
            start_char = current_cursor
            end_char = start_char + len(entity_value)
            
            span = PIISpan(
                label=entity_type,
                value=entity_value,
                start_char=start_char,
                end_char=end_char
            )
            entities.append(span)

            # 4. Acumular valor generado en el texto
            output_text_parts.append(entity_value)
            current_cursor = end_char
            last_match_end = match.end()

        # 5. Agregar el remanente final del texto tras el último token
        static_suffix = raw_template[last_match_end:]
        if static_suffix:
            output_text_parts.append(static_suffix)

        full_text = "".join(output_text_parts)

        # 6. Validación de integridad: aseguramos que text[start:end] == value
        for ent in entities:
            if not ent.validate_against_text(full_text):
                raise ValueError(
                    f"Error de alineación en span '{ent.label}': "
                    f"esperado '{ent.value}', obtenido '{full_text[ent.start_char:ent.end_char]}'"
                )

        return SyntheticTextSample(
            text=full_text,
            entities=entities,
            locale=self.locale,
            domain=domain
        )

    def generate_batch(self, num_samples: int, output_jsonl_path: Optional[str] = None) -> List[SyntheticTextSample]:
        """Genera un lote de ejemplos y opcionalmente los exporta en formato JSONL."""
        batch = [self.generate_sample() for _ in range(num_samples)]
        
        if output_jsonl_path:
            out_file = Path(output_jsonl_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                for sample in batch:
                    f.write(sample.model_dump_json() + "\n")
            print(f"[✓] Lote de {num_samples} muestras exportado exitosamente a: {output_jsonl_path}")

        return batch


if __name__ == "__main__":
    generator = PIIDataGenerator(locale="es_ES", seed=42)
    sample = generator.generate_sample()
    print("--- Texto Generado ---")
    print(sample.text)
    print("\n--- Spans PII Detectados ---")
    for ent in sample.entities:
        print(f"[{ent.label}] [{ent.start_char}:{ent.end_char}] -> '{ent.value}'")
