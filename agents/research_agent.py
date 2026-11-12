# agents/research_agent.py
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from retriever import CharcuteriaRAGSystem

rag_system = CharcuteriaRAGSystem()
rag_retriever = rag_system.get_retriever(k=3)


def buscar_recetas_y_mermas(query: str) -> str:
    """Busca información técnica, recetas, proporciones y mermas en la base de conocimiento RAG de Charcutería La Merced."""
    try:
        docs = rag_retriever.invoke(query)
        contexto = "\n".join([doc.page_content for doc in docs])
        return json.dumps({
            "fuentes_encontradas": len(docs),
            "informacion_tecnica": contexto
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def crear_research_agent(llm: ChatOpenAI):
    tools = [buscar_recetas_y_mermas]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "Sos el Agente de Búsqueda e Investigación de Charcutería La Merced. "
        "Tu rol es consultar la base de conocimiento técnico (RAG) utilizando la herramienta 'buscar_recetas_y_mermas' "
        "para extraer recetas exactas, proporciones de ingredientes y parámetros de merma de secadero. "
        "Sé preciso y entrégale los datos duros al supervisor."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])

    return prompt | llm_with_tools