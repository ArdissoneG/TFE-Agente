from app.llm import generar
from app.registro import registrar

PROMPT_BASELINE = "Sos un asistente de educación financiera para inversores minoristas en Argentina."


def responder_baseline(mensaje: str, meta: dict | None = None) -> dict:
    """
    Caso base: el mismo modelo y los mismos parámetros que el agente completo, pero
    sin RAG, sin adaptación al nivel y sin verificación. Solo recibe la línea de rol.
    """
    meta = meta or {}
    generacion = generar(
        [
            {"role": "system", "content": PROMPT_BASELINE},
            {"role": "user", "content": mensaje},
        ],
        seed=meta.get("seed"),
    )
    registrar(
        {
            "tipo": "explicacion",
            "condicion": "baseline",
            "run_id": meta.get("run_id"),
            "escenario_id": meta.get("escenario_id"),
            "repeticion": meta.get("repeticion"),
            "seed": meta.get("seed"),
            "mensaje": mensaje,
            "modelo": generacion["modelo"],
            "digest": generacion["digest"],
            "parametros": generacion["parametros"],
            "tema_detectado": None,
            "nivel_aplicado": None,
            "chunk_ids": None,
            "distancias": None,
            "respuesta": generacion["texto"],
        },
        ruta=meta.get("ruta_log"),
    )
    return {"respuesta": generacion["texto"]}