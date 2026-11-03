# 🥩 Orquestador Multi-Agente Especializado - Charcutería La Merced

Prototipo funcional de un sistema multi-agente basado en **LangGraph** desarrollado como proyecto para la Pre-entrega de Coderhouse. El sistema automatiza el cálculo de formulaciones, mermas de secadero y costos de producción para un negocio de charcutería artesanal.

---

## 🏗️ Topología y Arquitectura del Sistema

Se implementó una **Topología Jerárquica** controlada por un nodo central (**Supervisor**) que actúa como router inteligente y gestor del ciclo de vida de la tarea, delegando subtareas a agentes especialistas de manera eficiente y aplicando un criterio de parada estricto para evitar bucles infinitos.

---

### 📺 Demostración del Flujo de Delegación

Puedes ver el video de demostración del funcionamiento del orquestador multi-agente en el siguiente enlace:

👉 **[Ver Video de Demostración (demo.mp4)](./demo.mp4)**

<video src="demo.mp4" controls width="100%"></video>

---

#### Diagrama de Flujo (Mermaid)

```mermaid
graph TD
    START --> supervisor
    supervisor -->|research_agent| research_agent
    supervisor -->|analyst_agent| analyst_agent
    supervisor -->|FINISH| END
    
    research_agent --> research_tools
    research_tools --> supervisor
    
    analyst_agent --> analyst_tools
    analyst_tools --> supervisor
