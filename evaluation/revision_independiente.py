"""
Segundo revisor independiente: planilla y comparación con la revisión del autor.

  preparar  elige 30 respuestas ya puntuadas por el autor (10 de v1 phi3:mini, 10 de v1 llama3.1:8b y
            10 de la ablación; 8 con respuesta en el corpus y 2 fuera de corpus por grupo), mezcladas
            y sin modelo ni condición. Genera revision_independiente.html para el revisor.
  comparar  acuerdo exacto, kappa ponderado y diferencia media (revisor - autor) por criterio.

Uso (desde la raíz del repo):
    python evaluation/revision_independiente.py preparar
    (el revisor devuelve revision_independiente.csv; guardarlo en evaluation/resultados/revision_independiente/)
    python evaluation/revision_independiente.py comparar
"""
import argparse
import json
import random
import sys
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from ablacion_analisis import kappa_ponderado  # noqa: E402
from ablacion_revision import CAMPOS, COL, ORDEN, PAGINA, SIN_MATERIAL, escribir, leer, valor  # noqa: E402
from ejecutar_evaluacion import carpeta_modelo  # noqa: E402

DESTINO = RAIZ_EVALUACION / "resultados" / "revision_independiente"
ABLACION = RAIZ_EVALUACION / "resultados" / "ablacion"
MODELOS = ["phi3:mini", "llama3.1:8b"]


def respuestas_del_autor() -> list[dict]:
    """Todas las respuestas puntuadas por el autor en una primera vuelta."""
    filas = []
    for m in MODELOS:
        rev = carpeta_modelo(m) / "revision"
        clave = {c["id_item"]: c for c in leer(rev / "clave_revision.csv")}
        for f in leer(rev / "revision_ciega.csv"):
            filas.append({"fuente": f"v1 {m}", "id_original": f["id_item"], "modelo": m,
                          "condicion": clave[f["id_item"]]["condicion"], "fila": f})
    clave = {c["id_item"]: c for c in leer(ABLACION / "clave_ablacion.csv")}
    for f in leer(ABLACION / "revision_ablacion.csv"):
        k = clave[f["id_item"]]
        if k["origen"] == "nueva":
            filas.append({"fuente": "ablacion llama3.1:8b", "id_original": f["id_item"], "modelo": k["modelo"],
                          "condicion": k["condicion"], "fila": f})
    return filas


def preparar(args) -> None:
    if (DESTINO / "clave_independiente.csv").exists() and not args.sobrescribir:
        sys.exit(f"Ya existe {DESTINO}. Usá --sobrescribir solo si el revisor no empezó.")
    todas = respuestas_del_autor()
    rng = random.Random(args.semilla)
    elegidas = []
    for fuente in sorted({p["fuente"] for p in todas}):
        grupo = [p for p in todas if p["fuente"] == fuente]
        fuera = [p for p in grupo if p["fila"]["material_de_referencia"].startswith(SIN_MATERIAL)]
        corpus = [p for p in grupo if p not in fuera]
        elegidas += rng.sample(corpus, args.n_corpus) + rng.sample(fuera, args.n_fuera)
    rng.shuffle(elegidas)

    cols = CAMPOS + list(COL.values()) + ["observaciones"]
    blanco, clave, datos = [], [], []
    for k, p in enumerate(elegidas, 1):
        f = p["fila"]
        item_id = f"B{k:02d}"
        base = {"id_item": item_id, "pregunta": f["pregunta"], "material_de_referencia": f["material_de_referencia"],
                "respuesta": f["respuesta"]}
        blanco.append({**base, **{c: "" for c in cols if c not in base}})
        datos.append({**base, "fuera": f["material_de_referencia"].startswith(SIN_MATERIAL)})
        clave.append({"id_item": item_id, "fuente": p["fuente"], "id_original": p["id_original"],
                      "modelo": p["modelo"], "condicion": p["condicion"]})
    DESTINO.mkdir(parents=True, exist_ok=True)
    escribir(DESTINO / "revision_independiente_en_blanco.csv", cols, blanco)
    escribir(DESTINO / "clave_independiente.csv", list(clave[0].keys()), clave)
    pagina = (PAGINA.replace("Revisión ciega (ablación)", "Revisión independiente")
              .replace("revision_ablacion.csv", "revision_independiente.csv")
              .replace("revision-ablacion-v1-", "revision-independiente-v1-"))
    html = pagina.replace("__DATOS__", json.dumps({"cols": cols, "filas": datos}, ensure_ascii=False).replace("</", "<\\/"))
    (DESTINO / "revision_independiente.html").write_text(html, encoding="utf-8")
    print(f"{len(elegidas)} respuestas elegidas (semilla {args.semilla}).")
    print(f"Para el revisor: {DESTINO / 'revision_independiente.html'} y evaluation/rubrica.md")


def comparar(_) -> None:
    clave = {c["id_item"]: c for c in leer(DESTINO / "clave_independiente.csv")}
    autor = {(p["fuente"], p["id_original"]): p["fila"] for p in respuestas_del_autor()}
    revisor = leer(DESTINO / "revision_independiente.csv")
    vacias = [f["id_item"] for f in revisor if valor(f["pertinencia (0-2)"]) is None]
    if vacias:
        sys.exit(f"Hay {len(vacias)} filas sin completar (por ejemplo {vacias[:5]}).")
    salida, detalle = {}, []
    for crit, col in COL.items():
        a, b, por_fuente = [], [], {}
        for f in revisor:
            k = clave[f["id_item"]]
            v1, v2 = valor(autor[(k["fuente"], k["id_original"])][col]), valor(f[col])
            if v1 is None or v2 is None:
                continue
            a.append(v1)
            b.append(v2)
            por_fuente.setdefault(k["fuente"], []).append(v1 == v2)
        if not a:
            continue
        pos = lambda v: ORDEN[crit].index(v)
        salida[crit] = {"n": len(a), "acuerdo_exacto": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
                        "kappa_ponderado": kappa_ponderado(a, b, ORDEN[crit]),
                        "diferencia_media_revisor_menos_autor": round(sum(pos(y) - pos(x) for x, y in zip(a, b)) / len(a), 3),
                        "acuerdo_por_fuente": {s: round(sum(v) / len(v), 3) for s, v in por_fuente.items()}}
    for f in revisor:
        k = clave[f["id_item"]]
        p = autor[(k["fuente"], k["id_original"])]
        detalle.append({"id_item": f["id_item"], "fuente": k["fuente"], "id_original": k["id_original"],
                        "condicion": k["condicion"],
                        **{f"{c}_autor": valor(p[col]) for c, col in COL.items()},
                        **{f"{c}_revisor": valor(f[col]) for c, col in COL.items()}})
    print(f"{'criterio':16} {'n':>3} {'acuerdo':>8} {'kappa pond.':>12} {'dif. media':>11}   acuerdo por fuente")
    for crit, d in salida.items():
        kp = "n/d" if d["kappa_ponderado"] is None else f"{d['kappa_ponderado']:.2f}"
        print(f"{crit:16} {d['n']:>3} {d['acuerdo_exacto']:>8.0%} {kp:>12} {d['diferencia_media_revisor_menos_autor']:>+11.2f}   {d['acuerdo_por_fuente']}")
    print("\ndif. media: positivo = el revisor puntúa más alto que el autor. kappa n/d: sin variación suficiente.")
    (DESTINO / "comparacion_independiente.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir(DESTINO / "comparacion_independiente_detalle.csv", list(detalle[0].keys()), detalle)
    print(f"Guardado en {DESTINO}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="orden", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--n-corpus", type=int, default=8, help="por grupo, con respuesta en el corpus")
    p.add_argument("--n-fuera", type=int, default=2, help="por grupo, fuera de corpus")
    p.add_argument("--semilla", type=int, default=20261112)
    p.add_argument("--sobrescribir", action="store_true")
    p.set_defaults(fn=preparar)
    sub.add_parser("comparar").set_defaults(fn=comparar)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()