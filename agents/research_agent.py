# agents/research_agent.py
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def calcular_mezcla_charcuteria(peso_carne_kg: float, tipo_producto: str) -> str:
    """Calcula los gramos de aditivos y la merma esperada en el secadero para un lote."""
    # Proporciones estándar por kilo de carne
    sal_g = peso_carne_kg * 22
    cura_g = peso_carne_kg * 2.5
    pimienta_g = peso_carne_kg * 3
    
    # Merma estimada en el secadero (32% - 38%)
    peso_min = peso_carne_kg * 1000 * (1 - 0.38)
    peso_max = peso_carne_kg * 1000 * (1 - 0.32)

    resultado = {
        "producto": tipo_producto,
        "peso_inicial_kg": peso_carne_kg,
        "receta_gramos": {
            "sal_fina": round(sal_g, 1),
            "sales_de_cura": round(cura_g, 1),
            "pimienta_negra": round(pimienta_g, 1),
        },
        "merma_secadero_esperada": {
            "rango_porcentaje": "32% - 38%",
            "peso_final_min_g": round(peso_min, 1),
            "peso_final_max_g": round(peso_max, 1),
        }
    }
    return json.dumps(resultado, ensure_ascii=False)


def crear_research_agent(llm: ChatOpenAI):
    """Crea el agente especialista en recetas y mermas con instrucciones estrictas."""
    tools = [calcular_mezcla_charcuteria]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "Sos el Agente de Investigación y Recetas de Charcutería La Merced. "
        "Tu única tarea es calcular la formulación de aditivos (sal, sales de cura, especias) "
        "y la merma estimada en el secadero para UN LOTE GLOBAL de producción cárnica. "
        "NUNCA dividas un mismo lote en subproductos separados (por ejemplo, no calcules el salame por un lado "
        "y el tocino por otro si forman parte de la misma masa del lote). "
        "Utiliza la herramienta 'calcular_mezcla_charcuteria' una sola vez por lote."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])

    return prompt | llm_with_tools