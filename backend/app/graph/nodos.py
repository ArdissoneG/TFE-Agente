import ollama
from langgraph.types import interrupt
from app.graph.estado import EstadoChat
from app.rag.retriever import buscar_contexto
from app.db import guardar_perfil
from app.verification.preguntas import elegir_pregunta, calificar_respuesta, siguiente_nivel

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
    tema = state["tema_detectado"]
    pregunta_id = elegir_pregunta(tema, state["mensaje"])
    respuesta_usuario = interrupt({"tema": tema, "pregunta_id": pregunta_id})
    return {"respuesta_usuario": respuesta_usuario, "pregunta_id": pregunta_id}


def evaluar_verificacion(state: EstadoChat) -> dict:
    tema = state["tema_detectado"]
    calificacion = calificar_respuesta(state["pregunta_id"], state["respuesta_usuario"])
    acierto = calificacion["correcta"] if calificacion else None

    nivel_anterior = state["knowledge_profile"].get(tema, "intermedio")
    nivel_nuevo = siguiente_nivel(nivel_anterior, acierto)

    niveles_actualizados = dict(state["knowledge_profile"])
    niveles_actualizados[tema] = nivel_nuevo

    actualizacion = {
        "correcto": acierto,
        "nivel_nuevo": nivel_nuevo,
        "knowledge_profile": niveles_actualizados,
    }

    
    if acierto is not True:
        prompt_reexplicacion = (
            "Sos un asistente de educación financiera. El usuario respondió una pregunta de "
            "verificación sobre este tema de forma incompleta o incorrecta, lo que indica que "
            "la explicación anterior no fue suficientemente clara. Explicá el mismo concepto de "
            "nuevo, usando un enfoque o una analogía DISTINTA a la que usaste antes, en lenguaje simple.\n\n"
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