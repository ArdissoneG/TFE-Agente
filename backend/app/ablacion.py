from app.llm import generar
from app.rag.retriever import buscar_contexto
from app.registro import registrar

ROL = "Sos un asistente de educación financiera para inversores minoristas en Argentina. "
HONESTIDAD = (
    "Si no tenés información suficiente para responder con precisión, decilo "
    "explícitamente en vez de inventar una respuesta."
)

CON_CONTEXTO = (
    "Apoyate en el siguiente contexto extraído de material oficial de educación financiera. "
    "Si el contexto no tiene información suficiente para responder con precisión, decilo "
    "explícitamente en vez de inventar una respuesta.\n\n"
    "Contexto:\n"
)


def prompt_honestidad() -> str:
    return ROL + HONESTIDAD


def prompt_rag_sin_nivel(contexto_texto: str) -> str:
    return ROL + CON_CONTEXTO + contexto_texto


def _responder(condicion: str, mensaje: str, prompt: str, meta: dict, extra: dict) -> dict:
    generacion = generar(
        [{"role": "system", "content": prompt}, {"role": "user", "content": mensaje}],
        seed=meta.get("seed"),
    )
    registrar(
        {
            "tipo": "explicacion",
            "condicion": condicion,
            "run_id": meta.get("run_id"),
            "escenario_id": meta.get("escenario_id"),
            "repeticion": meta.get("repeticion"),
            "seed": meta.get("seed"),
            "mensaje": mensaje,
            "modelo": generacion["modelo"],
            "digest": generacion["digest"],
            "parametros": generacion["parametros"],
            "finalizacion": generacion["finalizacion"],
            "nivel_aplicado": None,
            **extra,
            "respuesta": generacion["texto"],
        },
        ruta=meta.get("ruta_log"),
    )
    return {"respuesta": generacion["texto"]}


def responder_prompt_honestidad(mensaje: str, meta: dict | None = None) -> dict:
    """Condición 2: rol + instrucción de honestidad, sin RAG ni nivel."""
    meta = meta or {}
    extra = {"tema_detectado": None, "chunk_ids": None, "distancias": None}
    return _responder("prompt_honestidad", mensaje, prompt_honestidad(), meta, extra)


def responder_rag_sin_nivel(mensaje: str, meta: dict | None = None) -> dict:
    """Condición 3: lo mismo que el agente completo, sin adaptación al nivel ni verificación."""
    meta = meta or {}
    r = buscar_contexto(mensaje)
    extra = {"tema_detectado": r["tema_detectado"], "chunk_ids": r["ids"], "distancias": r["distancias"]}
    prompt = prompt_rag_sin_nivel("\n\n".join(r["chunks"]))
    return _responder("rag_sin_nivel", mensaje, prompt, meta, extra)