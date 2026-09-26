import ollama
from langgraph.types import interrupt
from app.graph.estado import EstadoChat
from app.rag.retriever import buscar_contexto
from app.db import guardar_perfil
from app.verification.preguntas import verificar_respuesta, siguiente_nivel

USUARIO_ID = "default"

INSTRUCCIONES_NIVEL = {
    "principiante": "Explicá en lenguaje simple, sin jerga técnica. Usá analogías cotidianas cuando ayuden. No asumas conocimientos previos.",
    "intermedio": "Explicá con cierto nivel técnico, pero aclarando los términos clave la primera vez que aparecen.",
    "avanzado": "Respondé de forma directa y técnica, sin explicar conceptos básicos que ya se asumen conocidos.",
}


def _armar_prompt_sistema(nivel: str, contexto_texto: str) -> str:
    return (
        "Sos un asistente de educación financiera para inversores minoristas en Argentina. "
        f"El usuario tiene un nivel de conocimiento '{nivel}' en el tema de esta pregunta. "
        f"{INSTRUCCIONES_NIVEL[nivel]} "
        "Apoyate en el siguiente contexto extraído de material oficial de educación financiera. "
        "Si el contexto no tiene información suficiente para responder con precisión, decilo "
        "explícitamente en vez de inventar una respuesta.\n\n"
        f"Contexto:\n{contexto_texto}"
    )


def detectar_tema_y_contexto(state: EstadoChat) -> dict:
    resultado = buscar_contexto(state["mensaje"])
    return {
        "tema_detectado": resultado["tema_detectado"],
        "contexto_texto": "\n\n".join(resultado["chunks"]),
    }


def explicar(state: EstadoChat) -> dict:
    tema = state["tema_detectado"]
    nivel = state["knowledge_profile"].get(tema, "intermedio")
    prompt_sistema = _armar_prompt_sistema(nivel, state["contexto_texto"])
    result = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": state["mensaje"]},
        ],
    )
    return {"nivel_aplicado": nivel, "respuesta": result["message"]["content"]}


def verificar(state: EstadoChat) -> dict:
    # El grafo se pausa acá. Lo que le pasamos a interrupt() es lo que recibe
    # quien llame a /chat la primera vez (para armar la pregunta en el frontend).
    opcion = interrupt({"tema": state["tema_detectado"]})
    return {"opcion_elegida": opcion}


def evaluar_verificacion(state: EstadoChat) -> dict:
    tema = state["tema_detectado"]
    acierto = verificar_respuesta(tema, state["opcion_elegida"])
    nivel_anterior = state["knowledge_profile"].get(tema, "intermedio")
    nivel_nuevo = siguiente_nivel(nivel_anterior, acierto)

    niveles_actualizados = dict(state["knowledge_profile"])
    niveles_actualizados[tema] = nivel_nuevo
    guardar_perfil(USUARIO_ID, niveles_actualizados)

    actualizacion = {
        "correcto": acierto,
        "nivel_nuevo": nivel_nuevo,
        "knowledge_profile": niveles_actualizados,
    }

    # Acá está la rama condicional real: solo si falló, se genera una segunda
    # explicación con otro enfoque. Antes de este sprint, esto no existía.
    if not acierto:
        prompt_reexplicacion = (
            "Sos un asistente de educación financiera. El usuario respondió mal una "
            "pregunta de verificación sobre este tema, lo que indica que la explicación "
            "anterior no fue suficientemente clara para él. Explicá el mismo concepto de "
            "nuevo, usando un enfoque o una analogía DISTINTA a la que usaste antes, en "
            "lenguaje simple.\n\n"
            f"Explicación anterior:\n{state['respuesta']}\n\n"
            f"Pregunta original del usuario: {state['mensaje']}"
        )
        result = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": prompt_reexplicacion},
                {"role": "user", "content": state["mensaje"]},
            ],
        )
        actualizacion["reexplicacion"] = result["message"]["content"]

    return actualizacion