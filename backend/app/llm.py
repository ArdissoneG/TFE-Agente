from functools import lru_cache

import ollama

from app.config import MODELO_LLM, NUM_CTX, NUM_PREDICT, TEMPERATURA


@lru_cache(maxsize=1)
def digest_modelo() -> str:
    """Digest del modelo instalado en Ollama, para registrarlo en cada entrada del log."""
    try:
        for modelo in ollama.list().models:
            if modelo.model == MODELO_LLM:
                return modelo.digest
    except Exception:
        pass
    return "desconocido"


def parametros_generacion(seed: int | None = None) -> dict:
    opciones = {"temperature": TEMPERATURA, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
    if seed is not None:
        opciones["seed"] = seed
    return opciones


def generar(messages: list[dict], seed: int | None = None) -> dict:
    """
    Única puerta de entrada al LLM: garantiza que el baseline y el agente completo
    usen el mismo modelo y los mismos parámetros. Devuelve el texto generado y lo
    que hay que registrar en el log (modelo, digest y parámetros).
    """
    opciones = parametros_generacion(seed)
    resultado = ollama.chat(model=MODELO_LLM, messages=messages, options=opciones)
    try:
        finalizacion = resultado["done_reason"]  # "stop" = terminó sola; "length" = la cortó el tope
    except (KeyError, TypeError):
        finalizacion = None
    return {
        "texto": resultado["message"]["content"],
        "finalizacion": finalizacion,
        "modelo": MODELO_LLM,
        "digest": digest_modelo(),
        "parametros": opciones,
    }