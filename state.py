# state.py
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CharcuteriaState(TypedDict):
    # Historial de mensajes acumulado de forma segura
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Control de flujo dinámico del nodo Supervisor
    next_agent: Literal["research_agent", "analyst_agent", "FINISH"]
    
    # Dominio activo para trazabilidad y contexto
    active_domain: str
    
    # Contador de pasos estricto para prevenir el "Supervisor Infinito"
    step_count: int