"""
Generador de datos sintéticos y guiones conversacionales con LLMs:
- Ollama (Local, 100% gratuito, offline con Llama 3.2)
- Google Gemini y OpenAI (Opcionales con API keys)
"""

import os
import sys
import json
import re
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Cargar variables de entorno desde .env si existen
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schemas.pii_models import PIISpan, SyntheticTextSample

try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

try:
    from google import genai
    from google.genai import types as genai_types
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class _RawEntity(BaseModel):
    label: str = Field(description="Tipo de PII: NOMBRE, DIRECCION, TELEFONO, EMAIL, DNI, etc.")
    value: str = Field(description="El valor textual exacto del secreto tal como aparece en el texto")


class _LLMDialogOutput(BaseModel):
    text: str = Field(description="Texto fluido y natural de la conversación en español")
    entities: List[_RawEntity] = Field(description="Lista de secretos personales mencionados en el texto")
    domain: str = Field(default="atencion_cliente", description="Dominio temático")


class LLMPIIDataGenerator:
    """
    Genera guiones conversacionales sintéticos usando Ollama (local) o APIs en la nube.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key

        if self.provider == "ollama":
            self.model = model or "llama3.2:1b"
        elif self.provider == "gemini":
            self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            self.model = model or "gemini-2.5-flash"
        elif self.provider == "openai":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4o-mini"
        else:
            raise ValueError(f"Proveedor desconocido: '{provider}'. Usa 'ollama', 'gemini' u 'openai'.")

    def _generate_with_ollama(self, domain: str, custom_prompt: Optional[str] = None) -> _LLMDialogOutput:
        if not _HAS_OLLAMA:
            raise ImportError("Instala el cliente de ollama con 'pip install ollama'.")

        prompt = custom_prompt or (
            f"Genera una frase o mensaje de llamada telefónica realista en español para el sector de '{domain}'. "
            "El cliente debe hablar de forma natural e incluir su nombre (ej: Carlos Mendoza) y su ciudad o dirección. "
            "Devuelve OBLIGATORIAMENTE un JSON válido con esta estructura exacta:\n"
            "{\n"
            '  "text": "mensaje completo que dice el cliente",\n'
            '  "entities": [\n'
            '    {"label": "NOMBRE", "value": "nombre exacto que aparece en text"},\n'
            '    {"label": "DIRECCION", "value": "ciudad o direccion exacta que aparece en text"}\n'
            "  ],\n"
            f'  "domain": "{domain}"\n'
            "}"
        )

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )

        raw_json = response["message"]["content"]
        data = json.loads(raw_json)
        return _LLMDialogOutput(**data)

    def _generate_with_gemini(self, system_instruction: str, user_prompt: str) -> _LLMDialogOutput:
        client = genai.Client(api_key=self.api_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=_LLMDialogOutput,
            temperature=0.7
        )
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config
        )
        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed
        return _LLMDialogOutput(**json.loads(response.text))

    def _generate_with_openai(self, system_instruction: str, user_prompt: str) -> _LLMDialogOutput:
        client = OpenAI(api_key=self.api_key)
        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format=_LLMDialogOutput
        )
        return completion.choices[0].message.parsed

    def generate_dialogue(
        self,
        domain: str = "banca",
        custom_prompt: Optional[str] = None
    ) -> SyntheticTextSample:
        """
        Genera el guion con el LLM y calcula los límites exactos de caracteres.
        """
        if self.provider == "ollama":
            raw_output = self._generate_with_ollama(domain, custom_prompt)
        elif self.provider == "gemini":
            system_instruction = "Genera mensajes conversacionales realistas en español con PII."
            raw_output = self._generate_with_gemini(system_instruction, custom_prompt or f"Consulta de {domain}")
        else:
            system_instruction = "Genera mensajes conversacionales realistas en español con PII."
            raw_output = self._generate_with_openai(system_instruction, custom_prompt or f"Consulta de {domain}")

        full_text = raw_output.text

        # Calcular los límites exactos de caracteres (start_char, end_char)
        validated_spans: List[PIISpan] = []
        for raw_ent in raw_output.entities:
            val = raw_ent.value.strip()
            if not val:
                continue

            idx = full_text.find(val)
            if idx == -1:
                idx = full_text.lower().find(val.lower())
                if idx != -1:
                    val = full_text[idx:idx + len(val)]

            if idx != -1:
                start_c = idx
                end_c = idx + len(val)
                span = PIISpan(
                    label=raw_ent.label.upper(),
                    value=val,
                    start_char=start_c,
                    end_char=end_c
                )
                if span.validate_against_text(full_text):
                    validated_spans.append(span)

        return SyntheticTextSample(
            text=full_text,
            entities=validated_spans,
            locale="es_ES",
            domain=raw_output.domain or domain
        )
