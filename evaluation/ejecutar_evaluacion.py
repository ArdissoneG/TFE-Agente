"""
Runner de la evaluación experimental (Sprint 12).

Ejecuta las 30 preguntas (20 de recuperación + 10 fuera de corpus) en dos condiciones, con el mismo
modelo y los mismos parámetros:
  - baseline:         solo la línea de rol, sin RAG, sin nivel y sin verificación.
  - agente_completo:  RAG + nivel de la persona + verificación + nueva explicación si se equivoca.
Cada combinación se repite con las semillas 1, 2 y 3 (temperatura fija).

Cada escenario usa una persona ficticia con su propio perfil (el que sale de su diagnóstico) que
nunca se guarda en SQLite. En la verificación la persona responde según personas.opcion_elegida.
No necesita uvicorn: llama al grafo directamente. Sí necesita Ollama levantado.

Uso (desde la raíz del repo, con el venv del BACKEND activo):
    python evaluation/ejecutar_evaluacion.py --prueba            # 2 escenarios, semilla 1, carpeta aparte
    python evaluation/ejecutar_evaluacion.py                     # evaluación completa (phi3:mini)
    python evaluation/ejecutar_evaluacion.py --modelo llama3.1:8b  # mismo experimento con otro modelo

Es reanudable: si se corta (Ctrl+C, se apaga la máquina), se vuelve a lanzar el mismo comando y
retoma donde quedó. Cada corrida se guarda en su propio archivo; una corrida incompleta se rehace.

Resultados: evaluation/resultados/<modelo>/
    configuracion.json      modelo, digest, parámetros, commit y corpus de la evaluación
    corridas/<run_id>.jsonl un archivo por corrida (todos los eventos registrados)
    interacciones.jsonl     todas las corridas juntas (se regenera al terminar)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

RAIZ_EVALUACION = Path(__file__).resolve().parent
RAIZ_REPO = RAIZ_EVALUACION.parent
sys.path.insert(0, str(RAIZ_EVALUACION))

CONDICIONES = ["baseline", "agente_completo"]


def carpeta_modelo(modelo: str, prueba: bool = False) -> Path:
    slug = re.sub(r"[^A-Za-z0-9.\-]+", "_", modelo).strip("_")
    return RAIZ_EVALUACION / "resultados" / (slug + ("__prueba" if prueba else ""))


def _cargar_backend() -> SimpleNamespace:
    """Importa el backend recién ahora, para que TFE_MODELO_LLM ya esté definida."""
    from langgraph.types import Command

    import personas
    from app import config
    from app.baseline import responder_baseline
    from app.graph.grafo import grafo
    from app.llm import digest_modelo
    from app.rag.retriever import CHROMA_PATH, COLLECTION_NAME
    from app.registro import registrar

    return SimpleNamespace(
        Command=Command, config=config, responder_baseline=responder_baseline, grafo=grafo,
        digest_modelo=digest_modelo, CHROMA_PATH=CHROMA_PATH, COLLECTION_NAME=COLLECTION_NAME,
        registrar=registrar, cargar_personas=personas.cargar_personas,
        perfil_diagnosticado=personas.perfil_diagnosticado, opcion_elegida=personas.opcion_elegida,
    )


# ----------------------------------------------------------------------------- entorno
def _commit() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ_REPO, capture_output=True,
                                text=True, check=True).stdout.strip()
        sucio = bool(subprocess.run(["git", "status", "--porcelain", "backend", "evaluation"],
                                    cwd=RAIZ_REPO, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "cambios_sin_commitear": sucio}
    except Exception:
        return {"commit": None, "cambios_sin_commitear": None}


def verificar_entorno(be, forzar: bool) -> dict:
    """Comprueba Ollama, los modelos y el corpus congelado. Devuelve la configuración a registrar."""
    import chromadb
    import ollama

    from utils import cargar_corpus

    try:
        instalados = {m.model for m in ollama.list().models}
    except Exception as e:
        sys.exit(f"No puedo conectar con Ollama ({e}). ¿Está levantado?")
    for nombre in (be.config.MODELO_LLM, be.config.MODELO_EMBEDDINGS):
        if nombre not in instalados and f"{nombre}:latest" not in instalados:
            sys.exit(f"El modelo '{nombre}' no está instalado en Ollama (ollama pull {nombre}).")

    esperados = len(cargar_corpus())
    try:
        coleccion = chromadb.PersistentClient(path=be.CHROMA_PATH).get_collection(be.COLLECTION_NAME)
        actuales = coleccion.count()
    except Exception as e:
        sys.exit(f"No puedo abrir la base vectorial en {be.CHROMA_PATH} ({e}). ¿Corriste ingest.py?")
    if actuales != esperados and not forzar:
        sys.exit(f"La base vectorial tiene {actuales} fragmentos y el corpus congelado "
                 f"(evaluation/corpus_chunks.jsonl) tiene {esperados}. Re-ingestá o usá --forzar.")

    return {
        "modelo": be.config.MODELO_LLM,
        "digest": be.digest_modelo(),
        "modelo_embeddings": be.config.MODELO_EMBEDDINGS,
        "temperatura": be.config.TEMPERATURA,
        "num_ctx": be.config.NUM_CTX,
        "top_k": 3,
        "fragmentos_corpus": actuales,
        **_commit(),
    }


def guardar_configuracion(carpeta: Path, nueva: dict, semillas: list[int], forzar: bool) -> None:
    ruta = carpeta / "configuracion.json"
    if ruta.exists():
        previa = json.loads(ruta.read_text(encoding="utf-8"))
        cambios = [k for k in ("modelo", "digest", "temperatura", "num_ctx", "fragmentos_corpus")
                   if previa.get(k) != nueva.get(k)]
        if cambios and not forzar:
            sys.exit("La configuración cambió desde la primera corrida de esta carpeta "
                     f"({', '.join(cambios)}). Mezclar corridas con configuraciones distintas "
                     "invalida la comparación. Usá otra carpeta (otro --modelo) o --forzar.")
        if previa.get("commit") != nueva.get("commit"):
            print("AVISO: el commit del código es distinto al de las primeras corridas "
                  f"({previa.get('commit')} -> {nueva.get('commit')}).")
        return
    registro = {**nueva, "semillas": semillas, "creado": datetime.now().isoformat(timespec="seconds")}
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- corridas
def corrida_completa(ruta: Path) -> bool:
    if not ruta.exists():
        return False
    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    if not lineas:
        return False
    try:
        ultimo = json.loads(lineas[-1])
    except json.JSONDecodeError:
        return False
    return ultimo.get("tipo") == "corrida_completa" and not ultimo.get("error")


def correr_baseline(be, esc: dict, meta: dict) -> dict:
    be.responder_baseline(esc["pregunta"], meta)
    return {}


def correr_agente(be, esc: dict, persona: dict, meta: dict) -> dict:
    perfil = be.perfil_diagnosticado(persona)
    meta = {**meta, "persistir": False, "usuario_id": persona["id"]}
    config = {"configurable": {"thread_id": f"{meta['run_id']}-{time.time_ns()}"}}
    resultado = be.grafo.invoke(
        {"mensaje": esc["pregunta"], "knowledge_profile": perfil, "meta": meta}, config=config)
    interrupcion = resultado["__interrupt__"][0].value
    pregunta_id = interrupcion["pregunta_id"]
    tema = interrupcion["tema"]

    opcion = be.opcion_elegida(persona, pregunta_id) if pregunta_id else "a"
    final = be.grafo.invoke(be.Command(resume=opcion), config=config)
    return {
        "tema_detectado": tema,
        "pregunta_id": pregunta_id,
        "opcion_elegida": opcion,
        "correcto": final.get("correcto"),
        "nivel_inicial": perfil.get(tema, "intermedio"),
        "nivel_final": final.get("nivel_nuevo"),
        "hubo_reexplicacion": bool(final.get("reexplicacion")),
    }


def ejecutar(be, escenarios, personas, semillas, condiciones, carpeta: Path) -> int:
    corridas = carpeta / "corridas"
    corridas.mkdir(parents=True, exist_ok=True)
    tareas = [(s, e, c) for s in semillas for e in escenarios for c in condiciones]
    pendientes = [t for t in tareas
                  if not corrida_completa(corridas / f"{t[1]['id']}_{t[2]}_s{t[0]}.jsonl")]
    print(f"{len(tareas)} corridas en total, {len(tareas) - len(pendientes)} ya completas, "
          f"{len(pendientes)} por hacer.")

    fallidas, duraciones = 0, []
    for i, (seed, esc, cond) in enumerate(pendientes, 1):
        run_id = f"{esc['id']}_{cond}_s{seed}"
        ruta = corridas / f"{run_id}.jsonl"
        if ruta.exists():
            ruta.unlink()  # quedó incompleta: se rehace entera
        persona = personas[esc["persona_id"]]
        meta = {"seed": seed, "run_id": run_id, "escenario_id": esc["id"], "repeticion": seed,
                "condicion": cond, "ruta_log": str(ruta)}
        t0 = time.time()
        error, extra = None, {}
        try:
            extra = correr_baseline(be, esc, meta) if cond == "baseline" \
                else correr_agente(be, esc, persona, meta)
        except Exception as e:  # una corrida fallida no frena a las demás
            error = f"{type(e).__name__}: {e}"
            fallidas += 1
        segundos = round(time.time() - t0, 1)
        be.registrar({"tipo": "corrida_completa", "run_id": run_id, "escenario_id": esc["id"],
                      "tipo_pregunta": esc["tipo"], "persona_id": persona["id"], "condicion": cond,
                      "seed": seed, "segundos": segundos, "error": error, **extra}, ruta=ruta)
        duraciones.append(segundos)
        faltan = len(pendientes) - i
        eta = round(faltan * sum(duraciones) / len(duraciones) / 60)
        estado = f"ERROR {error}" if error else "ok"
        print(f"[{i}/{len(pendientes)}] {run_id} {segundos}s {estado}  (faltan ~{eta} min)")
    return fallidas


def consolidar(carpeta: Path) -> int:
    n = 0
    with open(carpeta / "interacciones.jsonl", "w", encoding="utf-8") as salida:
        for archivo in sorted((carpeta / "corridas").glob("*.jsonl")):
            for linea in archivo.read_text(encoding="utf-8").splitlines():
                salida.write(linea + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", help="modelo de Ollama (por defecto el de app/config.py)")
    ap.add_argument("--semillas", default="1,2,3")
    ap.add_argument("--condiciones", default=",".join(CONDICIONES))
    ap.add_argument("--escenarios", help="ids separados por coma (por ejemplo R01,O03); por defecto todos")
    ap.add_argument("--prueba", action="store_true",
                    help="humo: R01 y O01, semilla 1, en una carpeta aparte (<modelo>__prueba)")
    ap.add_argument("--forzar", action="store_true",
                    help="ignora el control de corpus/configuración (no recomendado)")
    args = ap.parse_args()

    if args.modelo:
        os.environ["TFE_MODELO_LLM"] = args.modelo
    be = _cargar_backend()

    from consultas_util import cargar_consultas

    escenarios = cargar_consultas()
    ids = args.escenarios.split(",") if args.escenarios else (["R01", "O01"] if args.prueba else None)
    if ids:
        escenarios = [e for e in escenarios if e["id"] in ids]
    semillas = [1] if args.prueba else [int(s) for s in args.semillas.split(",")]
    condiciones = args.condiciones.split(",")
    for c in condiciones:
        if c not in CONDICIONES:
            sys.exit(f"Condición desconocida: {c}")
    personas = {p["id"]: p for p in be.cargar_personas()}

    config = verificar_entorno(be, args.forzar)
    carpeta = carpeta_modelo(config["modelo"], args.prueba)
    carpeta.mkdir(parents=True, exist_ok=True)
    guardar_configuracion(carpeta, config, semillas, args.forzar)
    print(f"Modelo {config['modelo']} (digest {config['digest'][:12]}) · "
          f"temperatura {config['temperatura']} · num_ctx {config['num_ctx']}")
    print(f"Resultados en {carpeta}")

    fallidas = ejecutar(be, escenarios, personas, semillas, condiciones, carpeta)
    eventos = consolidar(carpeta)
    print(f"\nListo. {eventos} eventos en interacciones.jsonl.")
    if fallidas:
        print(f"{fallidas} corridas fallaron; volvé a lanzar el mismo comando para reintentarlas.")


if __name__ == "__main__":
    main()