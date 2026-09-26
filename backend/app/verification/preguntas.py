import random

PREGUNTAS = {
    "conceptos_basicos": {
        "pregunta": "¿Cuál de estas opciones es un ejemplo de invertir y no de ahorrar?",
        "opciones": {
            "a": "Guardar dinero en una caja fuerte",
            "b": "Comprar acciones de una empresa",
            "c": "Guardar billetes bajo el colchón",
            "d": "Dejar el dinero en la billetera",
        },
        "correcta": "b",
    },
    "instrumentos_inversion": {
        "pregunta": "¿Cuál de los siguientes es un instrumento de renta fija?",
        "opciones": {
            "a": "Acción",
            "b": "Bono",
            "c": "CEDEAR",
            "d": "Fondo común de inversión de acciones",
        },
        "correcta": "b",
    },
    "sesgos_y_riesgos": {
        "pregunta": "Invertir solo porque un familiar, amigo o influencer lo recomendó, sin analizarlo por tu cuenta, es un sesgo llamado...",
        "opciones": {
            "a": "Anclaje",
            "b": "Autoridad",
            "c": "Aversión a las pérdidas",
            "d": "Descuento hiperbólico",
        },
        "correcta": "b",
    },
    "fraudes_y_proteccion": {
        "pregunta": "¿Cuál de estas es una señal típica de una estafa de inversión?",
        "opciones": {
            "a": "Rendimiento moderado explicado con claridad",
            "b": "Promesa de alto rendimiento sin riesgo",
            "c": "Documentación disponible para revisar antes de firmar",
            "d": "Estar registrado en la CNV",
        },
        "correcta": "b",
    },
}

NIVELES_ORDEN = ["principiante", "intermedio", "avanzado"]


def armar_pregunta(tema: str) -> dict | None:
    """Devuelve la pregunta con las opciones en orden aleatorio (sin revelar la correcta)."""
    pregunta = PREGUNTAS.get(tema)
    if pregunta is None:
        return None
    ids_mezclados = list(pregunta["opciones"].keys())
    random.shuffle(ids_mezclados)
    return {
        "tema": tema,
        "pregunta": pregunta["pregunta"],
        "opciones": [{"id": id_, "texto": pregunta["opciones"][id_]} for id_ in ids_mezclados],
    }

def verificar_respuesta(tema: str, opcion_elegida: str) -> bool | None:
    pregunta = PREGUNTAS.get(tema)
    if pregunta is None:
        return None
    return opcion_elegida == pregunta["correcta"]


def siguiente_nivel(nivel_actual: str, acierto: bool) -> str:
    idx = NIVELES_ORDEN.index(nivel_actual)
    idx = min(idx + 1, len(NIVELES_ORDEN) - 1) if acierto else max(idx - 1, 0)
    return NIVELES_ORDEN[idx]