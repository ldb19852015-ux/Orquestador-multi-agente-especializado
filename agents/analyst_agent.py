# agents/analyst_agent.py
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def calcular_costos_produccion(peso_carne_kg: float, costo_kg_cerdo: float = 11000.0, costo_kg_tocino: float = 6000.0) -> str:
    """Calcula el costo total de materia prima, desglose y costo unitario por kilo de embutido."""
    kilos_cerdo = peso_carne_kg * 0.8
    kilos_tocino = peso_carne_kg * 0.2
    
    costo_total = (kilos_cerdo * costo_kg_cerdo) + (kilos_tocino * costo_kg_tocino)
    costo_por_kilo = costo_total / peso_carne_kg if peso_carne_kg > 0 else 0

    resultado = {
        "kilos_lote": peso_carne_kg,
        "costo_materia_prima_total": round(costo_total, 2),
        "costo_produccion_por_kilo": round(costo_por_kilo, 2),
        "fuente": "Planilla de costos y análisis financiero de Charcutería La Merced"
    }
    return json.dumps(resultado, ensure_ascii=False)


def crear_analyst_agent(llm: ChatOpenAI):
    tools = [calcular_costos_produccion]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "Sos el Agente de Análisis y Cómputo Financiero de Charcutería La Merced. "
        "Tu objetivo es procesar los datos cuantitativos, calcular costos de producción y verificar "
        "la viabilidad económica utilizando la herramienta 'calcular_costos_produccion'. "
        "Sintetiza de forma clara los resultados analíticos para el cierre de la tarea."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])

    return prompt | llm_with_tools