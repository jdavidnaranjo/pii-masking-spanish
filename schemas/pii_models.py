from pydantic import BaseModel, Field
from typing import List, Literal
from uuid import uuid4

class PIISpan(BaseModel):
    """Representa una entidad sensible con sus límites exactos en caracteres."""
    label: str = Field(..., description="Categoría de PII (e.g., NOMBRE, DNI, TELEFONO, EMAIL)")
    value: str = Field(..., description="Valor textual de la entidad")
    start_char: int = Field(..., ge=0, description="Índice inicial inclusivo del span en el texto")
    end_char: int = Field(..., ge=0, description="Índice final exclusivo del span en el texto")

    def validate_against_text(self, full_text: str) -> bool:
        """Verifica que el span coincida exactamente con el slice del texto original."""
        return full_text[self.start_char:self.end_char] == self.value

class SyntheticTextSample(BaseModel):
    """Muestra de texto sintético anotada con entidades PII para procesamiento de datos."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., description="Texto completo generado")
    entities: List[PIISpan] = Field(default_factory=list, description="Lista de spans PII anotados")
    locale: str = Field(default="es_ES", description="Configuración regional usada por Faker")
    domain: str = Field(default="atencion_cliente", description="Dominio o contexto conversacional")

class TimeInterval(BaseModel):
    """Intervalo de tiempo en segundos para alineación de audio."""
    start_time: float = Field(..., ge=0.0, description="Segundo de inicio")
    end_time: float = Field(..., ge=0.0, description="Segundo de finalización")
    label: str
    text_segment: str

class AudioPIIAnnotation(BaseModel):
    """Muestra completa de audio con sus intervalos temporales de PII."""
    sample_id: str
    audio_path: str
    duration_seconds: float
    time_intervals: List[TimeInterval]

# ── OCR / Documento ────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Caja delimitadora en píxeles de un campo PII dentro de una imagen de documento."""
    label: str = Field(..., description="Categoría de PII (e.g., NOMBRE, DNI, TELEFONO, EMAIL)")
    value: str = Field(..., description="Texto del campo tal como aparece en la imagen")
    x: int = Field(..., ge=0, description="Coordenada X de la esquina superior-izquierda (píxeles)")
    y: int = Field(..., ge=0, description="Coordenada Y de la esquina superior-izquierda (píxeles)")
    width: int = Field(..., gt=0, description="Ancho de la caja en píxeles")
    height: int = Field(..., gt=0, description="Alto de la caja en píxeles")

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Devuelve (x1, y1, x2, y2) — formato usado por la mayoría de librerías de visión."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

class DocumentPIIAnnotation(BaseModel):
    """Muestra completa de documento sintético con sus cajas PII anotadas."""
    sample_id: str = Field(default_factory=lambda: str(uuid4()))
    image_path: str = Field(..., description="Ruta relativa a la imagen PNG generada")
    doc_type: str = Field(..., description="Tipo de documento: formulario | factura | id_card")
    width_px: int = Field(..., description="Ancho total de la imagen en píxeles")
    height_px: int = Field(..., description="Alto total de la imagen en píxeles")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list,
                                              description="Lista de cajas delimitadoras de campos PII")
