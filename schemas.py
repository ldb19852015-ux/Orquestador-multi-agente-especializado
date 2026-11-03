# schemas.py
from enum import Enum

from pydantic import BaseModel, Field


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMConfig(BaseModel):
    provider: Provider = Provider.OPENAI
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str
    error: str | None = None


class FormulacionLote(BaseModel):
    producto: str = Field(description="Nombre o estilo del producto artesanal (ej. Salame picado fino).")
    peso_inicial_kg: float = Field(description="Peso total de la masa cárnica en kilogramos.")
    ingredientes_gramos: dict[str, float] = Field(description="Diccionario con los gramos exactos de cada ingrediente (sal, curalana, especias).")
    peso_final_esperado_g: dict[str, float] = Field(description="Rango estimado de peso final tras la merma del secadero.")


class CostosProduccion(BaseModel):
    kilos_lote: float = Field(description="Kilos totales del lote evaluado.")
    costo_materia_prima_total: float = Field(description="Costo total sumando carne, tocino y aditivos.")
    costo_produccion_por_kilo: float = Field(description="Costo unitario de producción por kilo terminado.")
    fuente_precios: str = Field(description="Origen de los datos, ej. 'Planilla Excel de Insumos'.")