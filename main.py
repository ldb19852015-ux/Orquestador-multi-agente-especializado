# main.py
import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.analyst_agent import calcular_costos_lote, crear_analyst_agent
from agents.research_agent import calcular_mezcla_charcuteria, crear_research_agent
from state import CharcuteriaState

load_dotenv()

# Inicializar modelo supervisor y LLM base para agentes
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Instanciar los agentes especializados usando sus funciones
research_agent_runnable = crear_research_agent(llm)
analyst_agent_runnable = crear_analyst_agent(llm)

# Lista de herramientas disponibles por los agentes
research_tools = [calcular_mezcla_charcuteria]
analyst_tools = [calcular_costos_lote]

# Nodos de herramientas para LangGraph
research_tool_node = ToolNode(research_tools)
analyst_tool_node = ToolNode(analyst_tools)

# ==========================================
# 1. NODO SUPERVISOR (Router Inteligente)
# ==========================================
async def supervisor_node(state: CharcuteriaState):
    """Analiza la conversación y decide qué especialista debe actuar o si se finaliza."""
    messages = state["messages"]
    step_count = state.get("step_count", 0) + 1

    # Criterio de parada estricto para evitar el "Supervisor Infinito"
    if step_count > 6:
        return {"next_agent": "FINISH", "step_count": step_count}

    system_prompt = (
        "Sos el Supervisor General de Charcutería La Merced. Tu trabajo es coordinar a dos especialistas:\n"
        "1. 'research_agent': Experto en recetas, formulación de aditivos y mermas de secadero.\n"
        "2. 'analyst_agent': Experto en costos de producción y precios de insumos.\n\n"
        "Debes responder estrictamente con uno de estos tres valores según la consulta: "
        "'research_agent', 'analyst_agent' o 'FINISH' si la tarea ya está completamente resuelta."
    )

    response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
    content = response.content.strip().lower()

    if "research" in content:
        next_agent = "research_agent"
    elif "analyst" in content or "costo" in content:
        next_agent = "analyst_agent"
    else:
        next_agent = "FINISH"

    return {"next_agent": next_agent, "step_count": step_count}


# ==========================================
# 2. NODOS DE LOS ESPECIALISTAS
# ==========================================
async def research_node(state: CharcuteriaState):
    response = await research_agent_runnable.ainvoke({"messages": state["messages"]})
    return {"messages": [response], "active_domain": "recetas_y_mermas"}

async def analyst_node(state: CharcuteriaState):
    response = await analyst_agent_runnable.ainvoke({"messages": state["messages"]})
    return {"messages": [response], "active_domain": "costos_y_precios"}


# ==========================================
# 3. CONSTRUCCIÓN DEL GRAFO
# ==========================================
workflow = StateGraph(CharcuteriaState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("research_tools", research_tool_node)
workflow.add_node("analyst_agent", analyst_node)
workflow.add_node("analyst_tools", analyst_tool_node)

# Flujo inicial desde START hacia el Supervisor
workflow.add_edge(START, "supervisor")

# Aristas condicionales desde el Supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "research_agent": "research_agent",
        "analyst_agent": "analyst_agent",
        "FINISH": END
    }
)

# Conexiones de herramientas para cada especialista
workflow.add_edge("research_agent", "research_tools")
workflow.add_edge("research_tools", "supervisor")

workflow.add_edge("analyst_agent", "analyst_tools")
workflow.add_edge("analyst_tools", "supervisor")

# Compilación del grafo
app = workflow.compile()


async def main():
    print("--- 🥩 INICIANDO ORQUESTADOR DE CHARCUTERÍA LA MERCED ---")
    query = "Necesito calcular la receta para un lote de 12 kg de salames y además saber el costo total si el cerdo sale 11000 el kilo y el tocino 6000."
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "research_agent",
        "active_domain": "inicio",
        "step_count": 0
    }

    async for chunk in app.astream(initial_state):
        print(chunk)
        print("-" * 40)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())