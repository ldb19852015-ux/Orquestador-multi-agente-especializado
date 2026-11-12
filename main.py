# main.py
import asyncio
import sys
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.analyst_agent import calcular_costos_produccion, crear_analyst_agent
from agents.research_agent import buscar_recetas_y_mermas, crear_research_agent
from state import CharcuteriaState

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

research_agent_runnable = crear_research_agent(llm)
analyst_agent_runnable = crear_analyst_agent(llm)

research_tool_node = ToolNode([buscar_recetas_y_mermas])
analyst_tool_node = ToolNode([calcular_costos_produccion])


async def supervisor_node(state: CharcuteriaState):
    messages = state["messages"]
    step_count = state.get("step_count", 0) + 1

    if step_count > 6:
        return {"next_agent": "FINISH", "step_count": step_count}

    system_prompt = (
        "Sos el Supervisor General y Validador de Charcutería La Merced. Tu trabajo es coordinar a dos especialistas:\n"
        "1. 'research_agent': Experto en buscar recetas, formulación de aditivos y mermas en la base RAG.\n"
        "2. 'analyst_agent': Experto en análisis de costos, cómputos y viabilidad financiera.\n\n"
        "Rúbrica de validación: La tarea solo finaliza cuando el investigador aportó la receta/proceso "
        "y el analista calculó los costos correspondientes. "
        "Debes responder estrictamente con uno de estos tres valores de mapeo: "
        "'research_agent', 'analyst_agent' o 'FINISH'."
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


async def research_node(state: CharcuteriaState):
    response = await research_agent_runnable.ainvoke({"messages": state["messages"]})
    return {"messages": [response], "active_domain": "investigacion_rag"}


async def analyst_node(state: CharcuteriaState):
    response = await analyst_agent_runnable.ainvoke({"messages": state["messages"]})
    return {"messages": [response], "active_domain": "analisis_financiero"}


workflow = StateGraph(CharcuteriaState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("research_tools", research_tool_node)
workflow.add_node("analyst_agent", analyst_node)
workflow.add_node("analyst_tools", analyst_tool_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "research_agent": "research_agent",
        "analyst_agent": "analyst_agent",
        "FINISH": END
    }
)

workflow.add_edge("research_agent", "research_tools")
workflow.add_edge("research_tools", "supervisor")

workflow.add_edge("analyst_agent", "analyst_tools")
workflow.add_edge("analyst_tools", "supervisor")

app = workflow.compile()


async def main():
    print("--- 🥩 INICIANDO ORQUESTADOR MULTI-AGENTE: CHARCUTERÍA LA MERCED ---")
    query = "Necesito investigar la receta y proporciones para un lote de salames de 15 kg y además calcular el costo total de producción con el analista."
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "research_agent",
        "active_domain": "inicio",
        "step_count": 0
    }

    async for chunk in app.astream(initial_state):
        print(chunk)
        print("-" * 50)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())