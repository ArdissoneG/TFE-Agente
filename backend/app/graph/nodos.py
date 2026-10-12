from langgraph.types import interrupt
from app.graph.estado import EstadoChat
from app.rag.retriever import buscar_contexto
from app.db import guardar_perfil
from app.llm import generar
from app.registro import registrar
from app.verification.preguntas import CATEGORIAS_ERROR, elegir_pregunta, calificar_respuesta, siguiente_nivel

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


def _meta(state: EstadoChat) -> dict:
    return state.get("meta") or {}


def _campos_comunes(state: EstadoChat, tipo: str) -> dict:
    meta = _meta(state)
    return {
        "tipo": tipo,
        "condicion": meta.get("condicion", "agente_completo"),
        "run_id": meta.get("run_id"),
        "escenario_id": meta.get("escenario_id"),
        "repeticion": meta.get("repeticion"),
        "seed": meta.get("seed"),
        "mensaje": state["mensaje"],
    }


def detectar_tema_y_contexto(state: EstadoChat) -> dict:
    resultado = buscar_contexto(state["mensaje"])
    return {
        "tema_detectado": resultado["tema_detectado"],
        "contexto_texto": "\n\n".join(resultado["chunks"]),
        "chunk_ids": resultado["ids"],
        "distancias": resultado["distancias"],
    }


def explicar(state: EstadoChat) -> dict:
    meta = _meta(state)
    tema = state["tema_detectado"]
    nivel = state["knowledge_profile"].get(tema, "intermedio")
    prompt_sistema = _armar_prompt_sistema(nivel, state["contexto_texto"])
    generacion = generar(
        [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": state["mensaje"]},
        ],
        seed=meta.get("seed"),
    )
    registrar(
        {
            **_campos_comunes(state, "explicacion"),
            "modelo": generacion["modelo"],
            "digest": generacion["digest"],
            "parametros": generacion["parametros"],
            "tema_detectado": tema,
            "nivel_aplicado": nivel,
            "chunk_ids": state.get("chunk_ids"),
            "distancias": state.get("distancias"),
            "respuesta": generacion["texto"],
        },
        ruta=meta.get("ruta_log"),
    )
    return {"nivel_aplicado": nivel, "respuesta": generacion["texto"]}


def verificar(state: EstadoChat) -> dict:
    tema = state["tema_detectado"]
    pregunta_id = elegir_pregunta(tema, state["mensaje"])
    respuesta_usuario = interrupt({"tema": tema, "pregunta_id": pregunta_id})
    return {"respuesta_usuario": respuesta_usuario, "pregunta_id": pregunta_id}


def evaluar_verificacion(state: EstadoChat) -> dict:
    meta = _meta(state)
    tema = state["tema_detectado"]
    # state["respuesta_usuario"] es el id de la opción elegida ("a", "b", "c" o "d")
    calificacion = calificar_respuesta(state["pregunta_id"], state["respuesta_usuario"])
    acierto = calificacion["correcta"] if calificacion else None

    nivel_anterior = state["knowledge_profile"].get(tema, "intermedio")
    nivel_nuevo = siguiente_nivel(nivel_anterior, acierto)

    niveles_actualizados = dict(state["knowledge_profile"])
    niveles_actualizados[tema] = nivel_nuevo

    # En evaluación (meta["persistir"] = False) el perfil queda solo en el estado del grafo,
    # así cada persona ficticia tiene el suyo y no se pisa el perfil real guardado en SQLite.
    if meta.get("persistir", True):
        guardar_perfil(meta.get("usuario_id", USUARIO_ID), niveles_actualizados)

    registrar(
        {
            **_campos_comunes(state, "verificacion"),
            "tema_detectado": tema,
            "pregunta_id": state["pregunta_id"],
            "opcion_elegida": state["respuesta_usuario"],
            "opcion_correcta": calificacion["opcion_correcta"] if calificacion else None,
            "categoria_error": calificacion["categoria_error"] if calificacion else None,
            "correcto": acierto,
            "nivel_anterior": nivel_anterior,
            "nivel_nuevo": nivel_nuevo,
        },
        ruta=meta.get("ruta_log"),
    )

    actualizacion = {
        "calificacion": calificacion,
        "correcto": acierto,
        "nivel_nuevo": nivel_nuevo,
        "knowledge_profile": niveles_actualizados,
    }

    if acierto is False:
        # Nueva explicación: ya no recibe la explicación anterior (el modelo tendía a copiarla y a
        # repetir sus errores). Recibe el contexto recuperado, la opción incorrecta que eligió la
        # persona con su categoría de error, y el enunciado correcto del banco, que está escrito
        # de antemano a partir del corpus: el modelo solo tiene que reformularlo para esa persona.
        prompt_reexplicacion = (
            "Sos un asistente de educación financiera para inversores minoristas en Argentina. "
            "El usuario leyó una explicación y después respondió una pregunta de opción múltiple "
            "eligiendo una opción incorrecta. "
            f"La opción que eligió fue: \"{calificacion['texto_elegido']}\". "
            f"Ese error es de este tipo: {CATEGORIAS_ERROR[calificacion['categoria_error']]} "
            f"Lo correcto, según el material, es: \"{calificacion['texto_correcto']}\". "
            "Explicá el concepto de nuevo en lenguaje simple, con una analogía distinta, "
            "corrigiendo ese error puntual. Hablale directamente al usuario, sin mencionar "
            "su error ni referirte a él en tercera persona. "
            "Apoyate en el siguiente contexto extraído de material oficial de educación financiera. "
            "Si el contexto no tiene información suficiente para responder con precisión, decilo "
            "explícitamente en vez de inventar una respuesta.\n\n"
            f"Contexto:\n{state['contexto_texto']}\n\n"
            f"Pregunta original del usuario: {state['mensaje']}"
        )
        generacion = generar(
            [
                {"role": "system", "content": prompt_reexplicacion},
                {"role": "user", "content": state["mensaje"]},
            ],
            seed=meta.get("seed"),
        )
        registrar(
            {
                **_campos_comunes(state, "reexplicacion"),
                "modelo": generacion["modelo"],
                "digest": generacion["digest"],
                "parametros": generacion["parametros"],
                "tema_detectado": tema,
                "chunk_ids": state.get("chunk_ids"),
                "categoria_error": calificacion["categoria_error"],
                "respuesta": generacion["texto"],
            },
            ruta=meta.get("ruta_log"),
        )
        actualizacion["reexplicacion"] = generacion["texto"]

    return actualizacion