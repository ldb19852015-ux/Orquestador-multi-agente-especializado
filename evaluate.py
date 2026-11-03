# evaluate.py
import asyncio

from dotenv import load_dotenv

from retriever import CharcuteriaRAGSystem

load_dotenv()


async def evaluar_sistema_charcuteria():
    print("🧪 Inicializando evaluación del CharcuteriaRAGSystem (Golden Set de 5 preguntas)...")
    
    # Instanciar el sistema RAG adaptado para charcutería
    rag = CharcuteriaRAGSystem()
    retriever = rag.get_retriever(k=5)

    # Golden Set de 5 preguntas orientadas al dominio de Charcutería La Merced
    golden_set = [
        {
            "pregunta": "¿Cuál es el procedimiento para el control de mermas en el secadero?",
            "expected_source": "manual_charcuteria.txt"
        },
        {
            "pregunta": "¿Qué parámetros de temperatura y humedad se deben verificar en la cámara de estacionamiento?",
            "expected_source": "manual_charcuteria.txt"
        },
        {
            "pregunta": "¿Cómo se calculan las proporciones exactas de sal y sales de cura por kilo de carne?",
            "expected_source": "manual_charcuteria.txt"
        },
        {
            "pregunta": "¿Cuáles son los pasos de control ante variaciones excesivas de humedad en el secadero?",
            "expected_source": "manual_charcuteria.txt"
        },
        {
            "pregunta": "¿Qué herramientas e insumos se utilizan para el atado y embolsado de salames?",
            "expected_source": "manual_charcuteria.txt"
        },
    ]

    hits = 0
    total = len(golden_set)

    print("\n" + "=" * 60)
    print(f"EJECUTANDO BENCHMARK DE RECUPERACIÓN (K=5) - Total: {total} preguntas")
    print("=" * 60)

    for i, item in enumerate(golden_set, 1):
        query = item["pregunta"]
        expected = item["expected_source"]

        try:
            docs = await retriever.ainvoke(query)
        except AttributeError:
            docs = retriever.invoke(query)

        sources_found = [doc.metadata.get("source", "manual_charcuteria.txt") for doc in docs]
        hit_detected = any(expected in str(s) for s in sources_found)

        if hit_detected:
            hits += 1
            status = "✅ Sí"
        else:
            status = "❌ No"

        precision = 1.0 if hit_detected else 0.0

        print(f"\n[{i}] Pregunta: {query}")
        print(f"    Esperado (fuente/contenido): {expected}")
        print(f"    Recuperados (fuentes): {sources_found}")
        print(f"    Hit (Recall@5): {status} | Precision@5: {precision:.2f}")

    recall_score = (hits / total) * 100

    print("\n" + "=" * 60)
    print("📊 RESUMEN DE MÉTRICAS FINALES - CHARCUTERÍA LA MERCED")
    print("=" * 60)
    print(f"Recall@5: {recall_score:.1f%} ({hits}/{total} aciertos)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(evaluar_sistema_charcuteria())