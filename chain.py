# chain.py
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas import FormulacionLote

# Configuración de logs para observar procesamiento
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()


def obtener_cadena_lcel():
    """Construye y retorna la cadena LCEL estructurada para formulación de charcutería."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en las variables de entorno.")

    # 1. Configuración del LLM
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=api_key,
    )

    # 2. Configuración de Salida Estructurada basada en Pydantic
    structured_llm = model.with_structured_output(FormulacionLote)

    # 3. Lógica de Resiliencia: Reintentos automáticos (.with_retry)
    structured_llm_with_retry = structured_llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    # 4. ChatPromptTemplate modular
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Sos un maestro charcutero experto en formulación de recetas y secadero. "
                    "Tu tarea es analizar el texto proporcionado y extraer la información "
                    "técnica del lote de producción siguiendo estrictamente el esquema."
                ),
            ),
            ("human", "Texto de producción a analizar:\n\"\"\"{texto_entrada}\"\"\""),
        ]
    )

    # 5. Cadena LCEL: prompt | model.with_structured_output().with_retry()
    chain = prompt | structured_llm_with_retry
    return chain


async def process_text(text: str) -> FormulacionLote:
    """Función asíncrona que procesa notas de elaboración mediante la cadena LCEL."""
    logger.info("Iniciando procesamiento de nota de producción...")
    chain = obtener_cadena_lcel()

    try:
        resultado: FormulacionLote = await chain.ainvoke({"texto_entrada": text})
        logger.info("Formulación analizada y validada con éxito.")
        return resultado
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        raise


async def main():
    # Prueba del pipeline con una nota de producción de salames
    nota_prueba = (
        "Lote preparado para 15 kg de salame picado grueso. Se agregaron 330g de sal fina, "
        "37.5g de sales de cura y 45g de pimienta negra. La merma esperada ronda entre el 32% y 38%."
    )

    print("\n--- INICIANDO PRUEBA DEL PIPELINE DE CHARCUTERÍA ---")
    resultado = await process_text(nota_prueba)

    print("\n--- RESULTADO VALIDADO (Objeto Pydantic / JSON) ---")
    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())