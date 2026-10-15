"""
Arma las planillas para la revisión manual con rúbrica, a partir de las corridas del runner.

Genera en evaluation/resultados/<modelo>/revision/:
  revision_ciega.csv          30 respuestas por condición (60 filas mezcladas, sin indicar la condición)
                              con el material de referencia y las columnas de la rúbrica para completar.
  clave_revision.csv          qué condición/escenario/semilla es cada fila. NO abrir hasta terminar.
  revision_reexplicaciones.csv  nuevas explicaciones del agente tras un error (tratamiento del error
                              conceptual), para evaluar si corrigen el error puntual.

Diseño de la muestra (fijo y reproducible con --semilla-muestreo):
  - Pares emparejados: para cada (escenario, semilla) elegido se revisan las dos condiciones, así la
    comparación baseline vs agente es sobre las mismas preguntas.
  - Dos estratos: preguntas con respuesta en el corpus y preguntas fuera de corpus (6 de 30 por defecto).
  - En cada estrato se cubren primero todos los escenarios una vez (con una semilla al azar) y el resto
    se completa con otros pares (escenario, semilla) al azar.

Uso (desde la raíz del repo, venv del backend):
    python evaluation/muestrear_revision.py
    python evaluation/muestrear_revision.py --modelo llama3.1:8b
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from ejecutar_evaluacion import carpeta_modelo  # noqa: E402
from consultas_util import cargar_consultas  # noqa: E402
from utils import cargar_corpus  # noqa: E402

TEXTO_FUERA_DE_CORPUS = ("(Sin material de referencia: el corpus no cubre este tema. "
                         "Evaluá si la respuesta lo reconoce y verificá los datos con tu conocimiento.)")


def leer_corridas(carpeta: Path) -> dict[str, list[dict]]:
    """run_id -> eventos, solo de corridas completas y sin error."""
    corridas = {}
    for archivo in sorted((carpeta / "corridas").glob("*.jsonl")):
        eventos = [json.loads(l) for l in archivo.read_text(encoding="utf-8").splitlines() if l.strip()]
        if eventos and eventos[-1].get("tipo") == "corrida_completa" and not eventos[-1].get("error"):
            corridas[archivo.stem] = eventos
    return corridas


def _evento(eventos: list[dict], tipo: str) -> dict | None:
    return next((e for e in eventos if e.get("tipo") == tipo), None)


def elegir_pares(pares: list[tuple], n: int, rng: random.Random) -> list[tuple]:
    """Primero un par por escenario (semilla al azar); si faltan, completa con otros pares al azar."""
    if n > len(pares):
        sys.exit(f"Piden {n} pares pero solo hay {len(pares)} disponibles. ¿Terminó el runner?")
    por_escenario = defaultdict(list)
    for par in pares:
        por_escenario[par[0]].append(par)
    escenarios = sorted(por_escenario)
    if n <= len(escenarios):
        return [rng.choice(por_escenario[e]) for e in rng.sample(escenarios, n)]
    elegidos = [rng.choice(por_escenario[e]) for e in escenarios]
    restantes = [p for p in pares if p not in elegidos]
    return elegidos + rng.sample(restantes, n - len(elegidos))


def material_referencia(consulta: dict, corpus: dict) -> str:
    if consulta["tipo"] == "fuera_de_corpus":
        return TEXTO_FUERA_DE_CORPUS
    return "\n\n".join(f"[{i}] {corpus[i]['texto'].strip()}" for i in consulta["relevantes"])


def escribir_csv(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    with open(ruta, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel lo abre bien
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def planilla_reexplicaciones(corridas: dict, consultas: dict, n: int, rng: random.Random, destino: Path) -> int:
    sys.path.insert(0, str(RAIZ_EVALUACION.parent / "backend"))
    from app.verification import preguntas as banco

    candidatas = []
    for run_id, eventos in corridas.items():
        re_ = _evento(eventos, "reexplicacion")
        ver = _evento(eventos, "verificacion")
        if re_ and ver:
            candidatas.append((run_id, eventos[-1], ver, re_))
    candidatas.sort(key=lambda c: c[0])
    elegidas = candidatas if len(candidatas) <= n else rng.sample(candidatas, n)
    rng.shuffle(elegidas)

    filas = []
    for k, (run_id, marca, ver, re_) in enumerate(elegidas, 1):
        p = banco._por_id(ver["pregunta_id"])
        elegida = next(o for o in p["opciones"] if o["id"] == ver["opcion_elegida"])
        correcta = next(o for o in p["opciones"] if o["categoria_error"] is None)
        filas.append({
            "id_item": f"E{k:02d}", "run_id": run_id,
            "pregunta_original": consultas[marca["escenario_id"]]["pregunta"],
            "pregunta_verificacion": p["pregunta"],
            "opcion_elegida": elegida["texto"], "categoria_error": ver["categoria_error"],
            "opcion_correcta": correcta["texto"], "reexplicacion": re_["respuesta"],
            "corrige_error (si/parcial/no)": "", "reproduce_error (si/no)": "",
            "alucinacion (0/1)": "", "artefactos (0/1)": "", "observaciones": "",
        })
    if filas:
        escribir_csv(destino, list(filas[0].keys()), filas)
    return len(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="phi3:mini")
    ap.add_argument("--prueba", action="store_true", help="usa la carpeta de la corrida de humo")
    ap.add_argument("--n", type=int, default=30, help="respuestas por condición")
    ap.add_argument("--n-fuera-corpus", type=int, default=6, help="de esas, cuántas fuera de corpus")
    ap.add_argument("--n-reexplicaciones", type=int, default=30)
    ap.add_argument("--semilla-muestreo", type=int, default=2026)
    ap.add_argument("--sobrescribir", action="store_true",
                    help="pisa planillas ya existentes (perdés lo que hayas completado)")
    args = ap.parse_args()

    carpeta = carpeta_modelo(args.modelo, args.prueba)
    destino = carpeta / "revision"
    destino.mkdir(parents=True, exist_ok=True)
    ciega = destino / "revision_ciega.csv"
    if ciega.exists() and not args.sobrescribir:
        sys.exit(f"{ciega} ya existe y puede tener tu revisión. Usá --sobrescribir si querés rehacerla.")

    corridas = leer_corridas(carpeta)
    if not corridas:
        sys.exit(f"No hay corridas completas en {carpeta / 'corridas'}.")
    consultas = {c["id"]: c for c in cargar_consultas()}
    corpus = cargar_corpus()
    rng = random.Random(args.semilla_muestreo)

    # pares (escenario, semilla) con las dos condiciones completas
    disponibles = {"recuperacion": [], "fuera_de_corpus": []}
    semillas = {(r.rsplit("_s", 1)[0].split("_", 1)[0], int(r.rsplit("_s", 1)[1])) for r in corridas}
    for esc, seed in sorted(semillas):
        if f"{esc}_baseline_s{seed}" in corridas and f"{esc}_agente_completo_s{seed}" in corridas:
            disponibles[consultas[esc]["tipo"]].append((esc, seed))

    n_o = args.n_fuera_corpus
    pares = elegir_pares(disponibles["recuperacion"], args.n - n_o, rng) \
        + elegir_pares(disponibles["fuera_de_corpus"], n_o, rng)

    items = []
    for esc, seed in pares:
        c = consultas[esc]
        for cond in ("baseline", "agente_completo"):
            eventos = corridas[f"{esc}_{cond}_s{seed}"]
            expl = _evento(eventos, "explicacion")
            items.append({"esc": esc, "seed": seed, "cond": cond, "tipo": c["tipo"],
                          "pregunta": c["pregunta"], "referencia": material_referencia(c, corpus),
                          "respuesta": expl["respuesta"], "persona": eventos[-1]["persona_id"],
                          "tema": expl.get("tema_detectado"), "chunks": expl.get("chunk_ids"),
                          "run_id": f"{esc}_{cond}_s{seed}"})
    rng.shuffle(items)

    filas, clave = [], []
    for k, it in enumerate(items, 1):
        item_id = f"X{k:02d}"
        fuera = it["tipo"] == "fuera_de_corpus"
        filas.append({
            "id_item": item_id, "pregunta": it["pregunta"], "material_de_referencia": it["referencia"],
            "respuesta": it["respuesta"],
            "fidelidad (0-2)": "NA" if fuera else "", "alucinacion (0/1)": "", "pertinencia (0-2)": "",
            "artefactos (0/1)": "", "reconoce_limite (si/parcial/no)": "" if fuera else "NA",
            "observaciones": "",
        })
        clave.append({"id_item": item_id, "run_id": it["run_id"], "escenario_id": it["esc"],
                      "tipo_pregunta": it["tipo"], "condicion": it["cond"], "repeticion": it["seed"],
                      "persona_id": it["persona"], "tema_detectado": it["tema"] or "",
                      "chunk_ids": ";".join(it["chunks"]) if it["chunks"] else ""})
    escribir_csv(ciega, list(filas[0].keys()), filas)
    escribir_csv(destino / "clave_revision.csv", list(clave[0].keys()), clave)

    n_re = planilla_reexplicaciones(corridas, consultas, args.n_reexplicaciones, rng,
                                    destino / "revision_reexplicaciones.csv")
    print(f"{len(pares)} pares ({args.n - n_o} con respuesta en el corpus + {n_o} fuera de corpus) "
          f"= {len(filas)} respuestas a revisar, mezcladas.")
    print(f"{n_re} nuevas explicaciones para revisar (tratamiento del error conceptual).")
    print(f"Planillas en {destino}")
    print("Completá revision_ciega.csv SIN abrir clave_revision.csv.")


if __name__ == "__main__":
    main()