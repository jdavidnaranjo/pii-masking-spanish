"""
document_generator.py
─────────────────────
Genera imágenes PNG de documentos sintéticos con datos PII falsos (Faker)
y devuelve las cajas delimitadoras exactas de cada campo sensible.

Tipos de documento soportados:
  • formulario  – Formulario de atención al cliente
  • factura     – Factura simplificada
  • id_card     – Documento de identidad estilo tarjeta

Dependencias: Pillow, faker
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import Callable

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from schemas.pii_models import BoundingBox, DocumentPIIAnnotation

# ── Configuración global ───────────────────────────────────────────────────────

FAKER = Faker("es_ES")
IMG_W, IMG_H = 794, 1123          # A4 a 96 dpi
MARGIN = 50
LINE_H = 28                        # altura de línea base
FONT_SIZE_TITLE = 22
FONT_SIZE_BODY = 15
FONT_SIZE_LABEL = 13

# Paleta de colores por tipo de documento
PALETTES: dict[str, dict] = {
    "formulario": {
        "bg": (255, 255, 255),
        "header_bg": (30, 80, 160),
        "header_fg": (255, 255, 255),
        "label_fg": (90, 90, 90),
        "value_fg": (10, 10, 10),
        "border": (180, 200, 230),
        "box_fill": (240, 245, 255),
    },
    "factura": {
        "bg": (255, 255, 255),
        "header_bg": (20, 120, 80),
        "header_fg": (255, 255, 255),
        "label_fg": (70, 100, 70),
        "value_fg": (10, 10, 10),
        "border": (180, 220, 200),
        "box_fill": (240, 255, 248),
    },
    "id_card": {
        "bg": (230, 235, 245),
        "header_bg": (180, 0, 0),
        "header_fg": (255, 255, 255),
        "label_fg": (80, 80, 80),
        "value_fg": (0, 0, 120),
        "border": (180, 180, 200),
        "box_fill": (245, 245, 255),
    },
}


# ── Carga de fuentes ───────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Intenta cargar DejaVu; si no existe usa la fuente bitmap por defecto."""
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(FONT_SIZE_TITLE, bold=True)
FONT_BODY  = _load_font(FONT_SIZE_BODY)
FONT_LABEL = _load_font(FONT_SIZE_LABEL)


# ── Helpers de dibujo ──────────────────────────────────────────────────────────

def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Devuelve (ancho, alto) del texto renderizado."""
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def _draw_field(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int,
    label: str, value: str,
    palette: dict,
    pii_label: str | None = None,
) -> BoundingBox | None:
    """
    Dibuja un campo con su etiqueta gris y su valor.
    Si pii_label no es None, devuelve el BoundingBox del valor.
    """
    pad = 6
    field_h = LINE_H + pad * 2

    # Fondo del campo
    draw.rectangle([x, y, x + w, y + field_h], fill=palette["box_fill"],
                   outline=palette["border"], width=1)
    # Etiqueta
    draw.text((x + pad, y + 2), label, font=FONT_LABEL, fill=palette["label_fg"])
    # Valor
    val_y = y + pad + 8
    draw.text((x + pad, val_y), value, font=FONT_BODY, fill=palette["value_fg"])

    if pii_label:
        vw, vh = _text_bbox(draw, value, FONT_BODY)
        return BoundingBox(
            label=pii_label,
            value=value,
            x=x + pad,
            y=val_y,
            width=max(vw, 4),
            height=max(vh, 12),
        )
    return None


def _draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str,
                 palette: dict, w: int) -> int:
    """Dibuja el banner superior y devuelve la coordenada Y donde termina."""
    header_h = 70
    draw.rectangle([0, 0, w, header_h], fill=palette["header_bg"])
    tw, _ = _text_bbox(draw, title, FONT_TITLE)
    draw.text(((w - tw) // 2, 14), title, font=FONT_TITLE, fill=palette["header_fg"])
    sw, _ = _text_bbox(draw, subtitle, FONT_LABEL)
    draw.text(((w - sw) // 2, 44), subtitle, font=FONT_LABEL, fill=palette["header_fg"])
    return header_h + 20


# ── Generadores por tipo ───────────────────────────────────────────────────────

def _generate_formulario(img_w: int, img_h: int) -> tuple[Image.Image, list[BoundingBox]]:
    palette = PALETTES["formulario"]
    img = Image.new("RGB", (img_w, img_h), palette["bg"])
    draw = ImageDraw.Draw(img)
    boxes: list[BoundingBox] = []

    y = _draw_header(draw, "FORMULARIO DE ATENCIÓN AL CLIENTE",
                     "Por favor complete todos los campos", palette, img_w)

    col_w = (img_w - MARGIN * 2 - 16) // 2
    fields_left = [
        ("Nombre completo",   FAKER.name(),               "NOMBRE"),
        ("DNI / NIF",         FAKER.nif(),                "DNI"),
        ("Fecha de nacimiento", FAKER.date_of_birth(minimum_age=18, maximum_age=90)
                                    .strftime("%d/%m/%Y"),  None),
        ("Profesión",         FAKER.job(),                None),
        ("Número de expediente", f"EXP-{FAKER.numerify('######')}", None),
    ]
    fields_right = [
        ("Teléfono",          FAKER.phone_number(),       "TELEFONO"),
        ("Correo electrónico", FAKER.email(),             "EMAIL"),
        ("Código postal",     FAKER.postcode(),           None),
        ("Ciudad",            FAKER.city(),               None),
        ("Fecha de solicitud", FAKER.date_this_year().strftime("%d/%m/%Y"), None),
    ]

    fh = LINE_H + 6 * 2 + 10   # alto de cada campo más separación

    for i, (lbl, val, pii) in enumerate(fields_left):
        fy = y + i * fh
        bb = _draw_field(draw, MARGIN, fy, col_w, lbl, val, palette, pii)
        if bb:
            boxes.append(bb)

    for i, (lbl, val, pii) in enumerate(fields_right):
        fy = y + i * fh
        bb = _draw_field(draw, MARGIN + col_w + 16, fy, col_w, lbl, val, palette, pii)
        if bb:
            boxes.append(bb)

    # Bloque de dirección completo (ancho total)
    addr_y = y + len(fields_left) * fh + 10
    address = FAKER.address().replace("\n", ", ")
    bb = _draw_field(draw, MARGIN, addr_y, img_w - MARGIN * 2,
                     "Dirección completa", address, palette, "DIRECCION")
    if bb:
        boxes.append(bb)

    # Área de observaciones (texto libre, sin PII)
    obs_y = addr_y + fh + 10
    draw.rectangle([MARGIN, obs_y, img_w - MARGIN, obs_y + 80],
                   fill=palette["box_fill"], outline=palette["border"])
    draw.text((MARGIN + 6, obs_y + 4), "Observaciones:", font=FONT_LABEL,
              fill=palette["label_fg"])

    # Línea de firma
    sig_y = obs_y + 110
    draw.line([MARGIN, sig_y, MARGIN + 200, sig_y], fill=(0, 0, 0), width=1)
    draw.text((MARGIN, sig_y + 4), "Firma del solicitante", font=FONT_LABEL,
              fill=palette["label_fg"])
    draw.line([img_w - MARGIN - 200, sig_y, img_w - MARGIN, sig_y],
              fill=(0, 0, 0), width=1)
    draw.text((img_w - MARGIN - 160, sig_y + 4), "Firma del agente", font=FONT_LABEL,
              fill=palette["label_fg"])

    return img, boxes


def _generate_factura(img_w: int, img_h: int) -> tuple[Image.Image, list[BoundingBox]]:
    palette = PALETTES["factura"]
    img = Image.new("RGB", (img_w, img_h), palette["bg"])
    draw = ImageDraw.Draw(img)
    boxes: list[BoundingBox] = []

    folio = f"FAC-{FAKER.numerify('####-####')}"
    y = _draw_header(draw, f"FACTURA  {folio}", "Documento fiscal simplificado",
                     palette, img_w)

    # ── Datos del emisor ──
    draw.text((MARGIN, y), "DATOS DEL EMISOR", font=FONT_LABEL, fill=palette["label_fg"])
    y += 18
    emisor_lines = [
        FAKER.company(),
        f"RFC: {FAKER.numerify('???######???')}",
        FAKER.address().replace("\n", " | "),
        FAKER.phone_number(),
    ]
    for line in emisor_lines:
        draw.text((MARGIN + 10, y), line, font=FONT_BODY, fill=palette["value_fg"])
        y += LINE_H

    y += 10
    draw.line([MARGIN, y, img_w - MARGIN, y], fill=palette["border"], width=1)
    y += 14

    # ── Datos del cliente ──
    draw.text((MARGIN, y), "DATOS DEL CLIENTE", font=FONT_LABEL, fill=palette["label_fg"])
    y += 18

    col_w = (img_w - MARGIN * 2 - 16) // 2
    fh = LINE_H + 6 * 2 + 10

    client_fields = [
        ("Nombre / Razón social", FAKER.name(),         "NOMBRE"),
        ("RFC / NIF del cliente",  FAKER.nif(),         "DNI"),
        ("Teléfono",               FAKER.phone_number(), "TELEFONO"),
        ("Correo electrónico",     FAKER.email(),        "EMAIL"),
    ]
    for i, (lbl, val, pii) in enumerate(client_fields[:2]):
        bb = _draw_field(draw, MARGIN, y + i * fh, col_w, lbl, val, palette, pii)
        if bb:
            boxes.append(bb)
    for i, (lbl, val, pii) in enumerate(client_fields[2:]):
        bb = _draw_field(draw, MARGIN + col_w + 16, y + i * fh, col_w, lbl, val, palette, pii)
        if bb:
            boxes.append(bb)

    y += 2 * fh + 10

    # Dirección del cliente
    address = FAKER.address().replace("\n", ", ")
    bb = _draw_field(draw, MARGIN, y, img_w - MARGIN * 2,
                     "Dirección de facturación", address, palette, "DIRECCION")
    if bb:
        boxes.append(bb)
    y += fh + 14

    # ── Tabla de conceptos ──
    draw.line([MARGIN, y, img_w - MARGIN, y], fill=palette["border"], width=1)
    y += 10
    draw.text((MARGIN, y), "CONCEPTOS", font=FONT_LABEL, fill=palette["label_fg"])
    y += 20

    col_widths = [320, 80, 90, 100]
    headers = ["Descripción", "Cantidad", "Precio unit.", "Total"]
    x_cursor = MARGIN
    for header, cw in zip(headers, col_widths):
        draw.text((x_cursor + 4, y), header, font=FONT_LABEL, fill=palette["label_fg"])
        x_cursor += cw
    y += LINE_H

    subtotal = 0.0
    for _ in range(random.randint(2, 4)):
        desc = FAKER.bs().capitalize()[:40]
        qty = random.randint(1, 10)
        price = round(random.uniform(10, 500), 2)
        total = qty * price
        subtotal += total

        row = [desc, str(qty), f"${price:.2f}", f"${total:.2f}"]
        x_cursor = MARGIN
        for val, cw in zip(row, col_widths):
            draw.text((x_cursor + 4, y), val, font=FONT_BODY, fill=palette["value_fg"])
            x_cursor += cw
        y += LINE_H

    y += 6
    draw.line([MARGIN, y, img_w - MARGIN, y], fill=palette["border"], width=1)
    y += 10
    iva = round(subtotal * 0.16, 2)
    total_final = round(subtotal + iva, 2)
    for label, amount in [("Subtotal", subtotal), ("IVA (16%)", iva), ("TOTAL", total_final)]:
        tw, _ = _text_bbox(draw, f"{label}: ${amount:.2f}", FONT_BODY)
        draw.text((img_w - MARGIN - tw - 10, y), f"{label}: ${amount:.2f}",
                  font=FONT_BODY, fill=palette["value_fg"])
        y += LINE_H

    return img, boxes


def _generate_id_card(img_w: int, img_h: int) -> tuple[Image.Image, list[BoundingBox]]:
    """Genera una tarjeta de identidad en formato apaisado (ancho > alto)."""
    card_w, card_h = 680, 430
    palette = PALETTES["id_card"]
    img = Image.new("RGB", (card_w, card_h), palette["bg"])
    draw = ImageDraw.Draw(img)
    boxes: list[BoundingBox] = []

    # Header
    draw.rectangle([0, 0, card_w, 55], fill=palette["header_bg"])
    title = "DOCUMENTO NACIONAL DE IDENTIDAD"
    tw, _ = _text_bbox(draw, title, FONT_TITLE)
    draw.text(((card_w - tw) // 2, 16), title, font=FONT_TITLE,
              fill=palette["header_fg"])

    # Foto simulada (rectángulo gris)
    photo_x, photo_y, photo_w, photo_h = 30, 75, 130, 170
    draw.rectangle([photo_x, photo_y, photo_x + photo_w, photo_y + photo_h],
                   fill=(200, 200, 200), outline=(150, 150, 150), width=2)
    draw.text((photo_x + 30, photo_y + 70), "FOTO", font=FONT_LABEL,
              fill=(120, 120, 120))

    # Campos de identidad
    data_x = photo_x + photo_w + 25
    fields = [
        ("Apellidos",          FAKER.last_name() + " " + FAKER.last_name(), "NOMBRE"),
        ("Nombre",             FAKER.first_name(),                           "NOMBRE"),
        ("Fecha de nacimiento", FAKER.date_of_birth(minimum_age=18, maximum_age=80)
                                     .strftime("%d %b %Y"),                  None),
        ("Lugar de nacimiento", FAKER.city(),                                None),
        ("DNI / NIF",          FAKER.nif(),                                  "DNI"),
        ("Fecha de expiración", FAKER.date_between(start_date="+1y", end_date="+10y")
                                     .strftime("%d/%m/%Y"),                  None),
    ]

    fy = 75
    fw = card_w - data_x - 30
    fh_small = LINE_H + 6 * 2 + 8

    for lbl, val, pii in fields:
        bb = _draw_field(draw, data_x, fy, fw, lbl, val, palette, pii)
        if bb:
            boxes.append(bb)
        fy += fh_small

    # Número de serie en la parte inferior (MRZ simulado)
    mrz_y = card_h - 50
    draw.rectangle([0, mrz_y, card_w, card_h], fill=(220, 220, 235))
    mrz_text = FAKER.numerify("P<ESP" + "?" * 20 + "<" * 5)
    draw.text((20, mrz_y + 8), mrz_text[:card_w // 9], font=FONT_LABEL,
              fill=(100, 100, 130))

    # Devolver en lienzo A4 centrado
    canvas = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    paste_x = (img_w - card_w) // 2
    paste_y = 80
    canvas.paste(img, (paste_x, paste_y))

    # Ajustar coordenadas de los boxes al canvas
    adjusted_boxes = [
        BoundingBox(
            label=bb.label, value=bb.value,
            x=bb.x + paste_x, y=bb.y + paste_y,
            width=bb.width, height=bb.height,
        )
        for bb in boxes
    ]
    return canvas, adjusted_boxes


# ── API pública ────────────────────────────────────────────────────────────────

_GENERATORS: dict[str, Callable] = {
    "formulario": _generate_formulario,
    "factura":    _generate_factura,
    "id_card":    _generate_id_card,
}

DOC_TYPES = list(_GENERATORS.keys())


def generate_document(
    doc_type: str = "formulario",
    output_path: str | Path | None = None,
) -> DocumentPIIAnnotation:
    """
    Genera un documento sintético y lo guarda como PNG.

    Args:
        doc_type:    Tipo de documento ("formulario", "factura", "id_card").
        output_path: Ruta donde guardar la imagen. Si es None, no se guarda en disco.

    Returns:
        DocumentPIIAnnotation con la ruta, dimensiones y cajas PII.
    """
    if doc_type not in _GENERATORS:
        raise ValueError(f"doc_type debe ser uno de {DOC_TYPES}; recibido: {doc_type!r}")

    img, boxes = _GENERATORS[doc_type](IMG_W, IMG_H)

    path_str = ""
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="PNG")
        path_str = str(out)

    return DocumentPIIAnnotation(
        image_path=path_str,
        doc_type=doc_type,
        width_px=img.width,
        height_px=img.height,
        bounding_boxes=boxes,
    )
