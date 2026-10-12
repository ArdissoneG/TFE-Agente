import json

import numpy as np
import ollama

PREGUNTAS = [
    # --- conceptos_basicos ---
    {
        "id": "conceptos_basicos_ahorro_vs_inversion",
        "tema": "conceptos_basicos",
        "pregunta": "Explicá con tus palabras la diferencia entre ahorrar e invertir.",
        "rubrica": [
            "ahorrar es guardar dinero sin buscar que crezca por encima de la inflación",
            "invertir implica asumir algún nivel de riesgo buscando un rendimiento",
        ],
        "contexto_semantico": "diferencia entre ahorrar e invertir, ejemplos de cada práctica",
    },
    {
        "id": "conceptos_basicos_riesgo_rentabilidad",
        "tema": "conceptos_basicos",
        "pregunta": (
            "Si una inversión promete un rendimiento mucho más alto que el resto del "
            "mercado sin explicar bien por qué, ¿qué deberías pensar sobre su riesgo? Explicá por qué."
        ),
        "rubrica": [
            "un rendimiento inusualmente alto suele venir acompañado de un riesgo mayor",
            "conviene desconfiar quiando no se explica el origen de ese rendimiento",
        ],
        "contexto_semantico": "relación entre riesgo y rentabilidad, desconfiar de rendimientos prometidos muy por encima del mercado",
    },

    # --- instrumentos_inversion ---
    {
        "id": "instrumentos_inversion_renta_fija",
        "tema": "instrumentos_inversion",
        "pregunta": "Explicá con tus palabras qué es un bono y en qué se diferencia de una acción.",
        "rubrica": [
            "un bono es un instrumento de deuda: quien lo compra le presta dinero al emisor",
            "una acción representa una porción de propiedad de la empresa, no un préstamo",
        ],
        "contexto_semantico": "instrumentos de renta fija, diferencia entre bonos y acciones",
    },
    {
        "id": "instrumentos_inversion_cedear",
        "tema": "instrumentos_inversion",
        "pregunta": "Explicá con tus palabras qué es un CEDEAR y para qué sirve.",
        "rubrica": [
            "es un certificado que representa una acción extranjera",
            "permite invertir en esa acción operando en pesos, en el mercado local",
        ],
        "contexto_semantico": "qué es un CEDEAR, certificado de depósito argentino, acceso a acciones extranjeras",
    },

    # --- sesgos_y_riesgos ---
    {
        "id": "sesgos_y_riesgos_autoridad",
        "tema": "sesgos_y_riesgos",
        "pregunta": (
            "Si invertís en algo solo porque te lo recomendó un familiar, amigo o influencer, "
            "sin analizarlo por tu cuenta, ¿qué sesgo cognitivo estarías cometiendo? Explicá brevemente en qué consiste."
        ),
        "rubrica": [
            "se llama sesgo de autoridad (o de prestigio social)",
            "consiste en confiar en una recomendación por la reputación de quien la da, en lugar de verificar la información",
        ],
        "contexto_semantico": "sesgo de autoridad, invertir por recomendación de alguien sin analizarlo uno mismo",
    },
    {
        "id": "sesgos_y_riesgos_regulado_no_regulado",
        "tema": "sesgos_y_riesgos",
        "pregunta": "Explicá con tus palabras qué diferencia a un instrumento de inversión regulado de uno no regulado.",
        "rubrica": [
            "uno regulado está autorizado y supervisado por la Comisión Nacional de Valores (CNV)",
            "uno no regulado se ofrece por fuera de los canales oficiales, muchas veces prometiendo rendimientos fijos inusualmente altos",
        ],
        "contexto_semantico": "distinción entre instrumentos regulados y no regulados, señales de falta de autorización de la CNV",
    },
    {
        "id": "sesgos_y_riesgos_aversion_perdida",
        "tema": "sesgos_y_riesgos",
        "pregunta": (
            "Si alguien mantiene una inversión que viene perdiendo valor solo para no "
            "\"confirmar\" la pérdida vendiéndola, ¿qué sesgo cognitivo es ese? Explicalo brevemente."
        ),
        "rubrica": [
            "se llama aversión a las pérdidas (o aversión a la pérdida)",
            "consiste en evitar vender en baja para no asumir la pérdida como definitiva, aunque eso empeore la decisión",
        ],
        "contexto_semantico": "sesgo de aversión a la pérdida, no vender una inversión en baja para evitar asumir la pérdida",
    },

    # --- fraudes_y_proteccion ---
    {
        "id": "fraudes_y_proteccion_senal_estafa",
        "tema": "fraudes_y_proteccion",
        "pregunta": "Nombrá y explicá con tus palabras al menos una señal típica de una estafa de inversión.",
        "rubrica": [
            "promesa de un rendimiento alto o garantizado sin explicar de dónde sale",
            "falta de transparencia o de documentación oficial disponible para revisar",
        ],
        "contexto_semantico": "señales de alerta de una estafa de inversión, promesas de rendimiento sin riesgo",
    },
    {
        "id": "fraudes_y_proteccion_canal_denuncia",
        "tema": "fraudes_y_proteccion",
        "pregunta": (
            "Si sospechás que te contactaron con una oferta de inversión fraudulenta en "
            "Argentina, ¿ante qué organismo podés hacer la denuncia o consulta? Explicá brevemente su rol."
        ),
        "rubrica": [
            "la Comisión Nacional de Valores (CNV)",
            "es el organismo de control y protección al inversor en el mercado de capitales argentino",
        ],
        "contexto_semantico": "canales de denuncia ante fraude financiero, rol de la CNV como organismo de control",
    },
]

NIVELES_ORDEN = ["principiante", "intermedio", "avanzado"]


def _por_tema(tema: str) -> list[dict]:
    return [p for p in PREGUNTAS if p["tema"] == tema]


def _por_id(pregunta_id: str) -> dict | None:
    return next((p for p in PREGUNTAS if p["id"] == pregunta_id), None)


def _embed(texto: str) -> list[float]:
    return ollama.embed(model="nomic-embed-text", input=texto)["embeddings"][0]


def _similitud_coseno(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def elegir_pregunta(tema: str, mensaje: str) -> str | None:
    """
    De las 2-3 preguntas fijas del banco para ese tema, devuelve el id de la
    más relacionada semánticamente con el mensaje del usuario (mismo criterio
    que antes: similitud coseno contra 'contexto_semantico', embeddings de
    nomic-embed-text, igual que el RAG).
    """
    candidatas = _por_tema(tema)
    if not candidatas:
        return None
    if len(candidatas) == 1:
        return candidatas[0]["id"]

    embedding_mensaje = _embed(mensaje)
    mejor_id, mejor_similitud = candidatas[0]["id"], -1.0
    for candidata in candidatas:
        similitud = _similitud_coseno(embedding_mensaje, _embed(candidata["contexto_semantico"]))
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_id = candidata["id"]
    return mejor_id


def armar_pregunta(pregunta_id: str) -> dict | None:
    """Devuelve tema + enunciado abierto de la pregunta (ya no hay opciones que mezclar)."""
    pregunta = _por_id(pregunta_id)
    if pregunta is None:
        return None
    return {"tema": pregunta["tema"], "pregunta": pregunta["pregunta"]}


def calificar_respuesta(pregunta_id: str, respuesta_usuario: str) -> dict | None:
    """
    Califica una respuesta libre contra la rúbrica FIJA de la pregunta (los
    puntos clave están escritos de antemano en PREGUNTAS, no los genera el
    modelo). El LLM solo juzga si esos puntos están presentes en la respuesta
    del usuario, parafraseados o no -- una tarea acotada y auditable, a
    diferencia de dejar que decida "qué es correcto" sin ninguna referencia.

    Devuelve {"correcta": True/False/None, "puntos_detectados": [...]}.
    correcta=None significa que el calificador no devolvió un JSON parseable
    (falla del LLM, no del usuario) -- en ese caso no se penaliza ni se
    premia, el nivel se mantiene igual (ver siguiente_nivel).
    """
    pregunta = _por_id(pregunta_id)
    if pregunta is None:
        return None

    puntos_clave = pregunta["rubrica"]
    prompt = (
        "Sos un corrector de respuestas de educación financiera para inversores minoristas. "
        "Se te da una pregunta, una lista de puntos clave que debería mencionar una respuesta "
        "correcta (no hace falta que use las mismas palabras, alcanza con la idea) y la "
        "respuesta que dio un usuario.\n\n"
        f"Pregunta: {pregunta['pregunta']}\n\n"
        "Puntos clave esperados:\n"
        + "\n".join(f"- {p}" for p in puntos_clave)
        + f"\n\nRespuesta del usuario: \"{respuesta_usuario}\"\n\n"
        "Evaluá cuántos puntos clave están presentes en la respuesta, aunque estén "
        "parafraseados. Considerala correcta si menciona al menos la mitad de los puntos "
        "clave (redondeando hacia arriba) sin contradecirlos.\n\n"
        "Respondé ÚNICAMENTE con un JSON válido, sin texto adicional, con este formato exacto: "
        '{"correcta": true o false, "puntos_detectados": ["..."]}'
    )
    result = ollama.chat(model="phi3:mini", messages=[{"role": "user", "content": prompt}])
    contenido = result["message"]["content"]

    try:
        inicio, fin = contenido.index("{"), contenido.rindex("}") + 1
        calificacion = json.loads(contenido[inicio:fin])
        return {
            "correcta": bool(calificacion["correcta"]),
            "puntos_detectados": calificacion.get("puntos_detectados", []),
        }
    except (ValueError, KeyError, json.JSONDecodeError):
        return {"correcta": None, "puntos_detectados": []}


def siguiente_nivel(nivel_actual: str, acierto: bool | None) -> str:
    if acierto is None:
        return nivel_actual  # el calificador falló: no subimos ni bajamos de nivel
    idx = NIVELES_ORDEN.index(nivel_actual)
    idx = min(idx + 1, len(NIVELES_ORDEN) - 1) if acierto else max(idx - 1, 0)
    return NIVELES_ORDEN[idx]