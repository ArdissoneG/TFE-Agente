"""
Análisis con corrección de Holm: v1 recalculada y ablación (llama3.1:8b).

  v1        baseline vs agente_completo en los dos modelos, con p ajustado por Holm.
  ablacion  cuatro condiciones sobre los mismos pares; contrastes planificados con Holm.
  segunda   consistencia del evaluador en las filas repetidas (acuerdo y kappa ponderado).

Familias (se fijan antes de mirar los resultados, ver hipotesis_ablacion.md):
  v1:       4 pruebas por modelo (fidelidad, alucinación, pertinencia y artefactos, preguntas con
            respuesta en el corpus). También se informa el ajuste sobre las 8 pruebas de ambos modelos.
  ablación: 9 pruebas (3 contrastes x fidelidad, alucinación y pertinencia, preguntas con respuesta
            en el corpus). El resto se informa sin ajustar, como exploratorio.

Uso: python evaluation/ablacion_analisis.py v1 | ablacion | segunda
Guarda JSON y CSV en evaluation/resultados/ablacion/.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from ablacion_revision import COL, DESTINO, ORDEN, leer, valor  # noqa: E402
from comparacion_pareada import CRITERIOS, bootstrap_ic, numero, prueba_signo  # noqa: E402
from ejecutar_evaluacion import carpeta_modelo  # noqa: E402

MODELOS = ["phi3:mini", "llama3.1:8b"]
LLAMA = "llama3.1:8b"
ESCALERA = ["baseline", "prompt_honestidad", "rag_sin_nivel", "agente_completo"]
CONTRASTES = [("prompt_honestidad", "baseline"), ("rag_sin_nivel", "prompt_honestidad"),
              ("agente_completo", "rag_sin_nivel")]
PRIMARIOS = ["fidelidad", "alucinacion", "pertinencia"]
CORPUS, FUERA = "recuperacion", "fuera_de_corpus"


def holm(ps: list[float]) -> list[float]:
    orden = sorted(range(len(ps)), key=lambda i: ps[i])
    ajustados, previo = [0.0] * len(ps), 0.0
    for rango, i in enumerate(orden):
        previo = max(previo, min(1.0, (len(ps) - rango) * ps[i]))
        ajustados[i] = previo
    return ajustados


def pares_v1(modelo: str) -> dict:
    """(tipo, escenario, semilla) -> {condicion: fila}, en el mismo orden que comparacion_pareada.py."""
    rev = carpeta_modelo(modelo) / "revision"
    clave = {c["id_item"]: c for c in leer(rev / "clave_revision.csv")}
    pares = {}
    for fila in leer(rev / "revision_ciega.csv"):
        k = clave[fila["id_item"]]
        pares.setdefault((k["tipo_pregunta"], k["escenario_id"], k["repeticion"]), {})[k["condicion"]] = fila
    return pares


def contraste(pares: dict, estrato: str, a: str, b: str, criterio: str, rng: random.Random):
    """Mejora media de a sobre b (positivo = a es mejor), IC bootstrap y prueba del signo."""
    col, sentido = CRITERIOS[criterio]
    difs = []
    for (tipo, _, _), p in pares.items():
        if tipo != estrato or a not in p or b not in p:
            continue
        x, y = numero(p[a][col]), numero(p[b][col])
        if x is not None and y is not None:
            difs.append((x - y) * sentido)
    if not difs:
        return None
    gana, pierde = sum(d > 0 for d in difs), sum(d < 0 for d in difs)
    lo, hi = bootstrap_ic(difs, rng)
    return {"pares": len(difs), "mejora_media": round(sum(difs) / len(difs), 3), "ic95": [round(lo, 3), round(hi, 3)],
            "a_mejor": gana, "empates": len(difs) - gana - pierde, "b_mejor": pierde,
            "p": round(prueba_signo(gana, pierde), 4)}


def guardar(nombre: str, datos) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / nombre).write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_filas(nombre: str, filas: list[dict]) -> None:
    with open(DESTINO / nombre, "w", encoding="utf-8-sig", newline="") as f:
        columnas = list(dict.fromkeys(k for fila in filas for k in fila))
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def linea(c: dict | None, p_aj: float | None = None) -> str:
    if c is None:
        return "sin datos"
    ic = f"[{c['ic95'][0]:+.2f}, {c['ic95'][1]:+.2f}]"
    aj = "" if p_aj is None else f" {p_aj:>8.3f}"
    return f"{c['pares']:>4} {c['mejora_media']:>+7.2f} {ic:>16} {c['a_mejor']:>3}/{c['empates']}/{c['b_mejor']:<3} {c['p']:>7.3f}{aj}"


# ----------------------------------------------------------------------------- v1
def cmd_v1(_) -> None:
    filas = []
    for modelo in MODELOS:
        pares = pares_v1(modelo)
        rng = random.Random(2026)  # misma semilla y orden que comparacion_pareada.py
        for estrato in (CORPUS, FUERA):
            for criterio in CRITERIOS:
                c = contraste(pares, estrato, "agente_completo", "baseline", criterio, rng)
                if c:
                    filas.append({"modelo": modelo, "estrato": estrato, "criterio": criterio, **c})
    familia = [f for f in filas if f["estrato"] == CORPUS]
    for modelo in MODELOS:
        fam = [f for f in familia if f["modelo"] == modelo]
        for f, aj in zip(fam, holm([f["p"] for f in fam])):
            f["p_holm_por_modelo"] = round(aj, 4)
    for f, aj in zip(familia, holm([f["p"] for f in familia])):
        f["p_holm_ambos_modelos"] = round(aj, 4)

    print("Agente completo vs baseline (positivo = mejor el agente). Familia: preguntas con respuesta en el corpus.")
    print(f"Pruebas por modelo: 4 (fidelidad, alucinación, pertinencia, artefactos). Ambos modelos: {len(familia)}.\n")
    for modelo in MODELOS:
        print(f"{modelo}\n{'estrato | criterio':32} pares  mejora           IC 95 %  a/e/b   p(signo)  Holm(4)  Holm(8)")
        for f in (f for f in filas if f["modelo"] == modelo):
            nombre = f"{f['estrato']} | {f['criterio']}"
            extra = (f"  {f['p_holm_por_modelo']:>7.3f}  {f['p_holm_ambos_modelos']:>7.3f}"
                     if "p_holm_por_modelo" in f else "  (sin ajustar: exploratorio, 6 pares)")
            print(f"{nombre:32}{linea(f)}{extra}")
        print()
    guardar("v1_holm.json", {"familia_por_modelo": 4, "familia_ambos_modelos": len(familia), "pruebas": filas})
    csv_filas("v1_holm.csv", [{**f, "ic95": f"{f['ic95'][0]}; {f['ic95'][1]}"} for f in filas])
    print(f"Guardado en {DESTINO}")


# ----------------------------------------------------------------------------- ablación
def pares_ablacion() -> dict:
    pares = pares_v1(LLAMA)
    try:
        clave = {c["id_item"]: c for c in leer(DESTINO / "clave_ablacion.csv")}
        hechas = leer(DESTINO / "revision_ablacion.csv")
    except FileNotFoundError as e:
        sys.exit(f"Falta {e.filename}. Copiá el CSV descargado de la página a {DESTINO} como revision_ablacion.csv.")
    vacias = [f["id_item"] for f in hechas if valor(f["pertinencia (0-2)"]) is None]
    if vacias:
        sys.exit(f"Hay {len(vacias)} filas sin completar en revision_ablacion.csv (por ejemplo {vacias[:5]}).")
    for fila in hechas:
        k = clave[fila["id_item"]]
        if k["origen"] == "nueva":
            pares.setdefault((k["tipo_pregunta"], k["escenario_id"], k["repeticion"]), {})[k["condicion"]] = fila
    return {k: p for k, p in pares.items() if all(c in p for c in ESCALERA)}


def medias(pares: dict, estrato: str, cond: str) -> dict:
    grupo = [p[cond] for (t, _, _), p in pares.items() if t == estrato]
    out = {"pares": len(grupo)}
    for crit, (col, _) in CRITERIOS.items():
        v = [numero(f[col]) for f in grupo if numero(f[col]) is not None]
        out[crit] = round(sum(v) / len(v), 3) if v else None
    if estrato == FUERA:
        lim = [valor(f[COL["reconoce_limite"]]) for f in grupo]
        out["reconoce_limite"] = {k: lim.count(k) for k in ORDEN["reconoce_limite"]}
    return out


def chequeo_diseno(pares: dict) -> dict:
    """Mismos fragmentos en rag_sin_nivel y agente_completo; respuestas cortadas por el tope."""
    corridas = carpeta_modelo(LLAMA) / "corridas"
    iguales = total = 0
    cortadas = {c: 0 for c in ESCALERA}
    for (_, esc, seed), _p in pares.items():
        ev = {}
        for cond in ESCALERA:
            ruta = corridas / f"{esc}_{cond}_s{seed}.jsonl"
            lineas = [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines() if x.strip()]
            ev[cond] = next(e for e in lineas if e.get("tipo") == "explicacion")
            cortadas[cond] += ev[cond].get("finalizacion") == "length"
        total += 1
        iguales += ev["rag_sin_nivel"]["chunk_ids"] == ev["agente_completo"]["chunk_ids"]
    return {"pares": total, "mismos_fragmentos_rag_vs_agente": iguales, "respuestas_cortadas_por_tope": cortadas}


def cmd_ablacion(_) -> None:
    pares = pares_ablacion()
    n_c = sum(t == CORPUS for t, _, _ in pares)
    print(f"Pares con las cuatro condiciones: {len(pares)} ({n_c} con respuesta en el corpus, {len(pares) - n_c} fuera de corpus)")
    chequeo = chequeo_diseno(pares)
    print(f"Chequeo: mismos fragmentos en rag_sin_nivel y agente_completo en {chequeo['mismos_fragmentos_rag_vs_agente']}/{chequeo['pares']} pares; "
          f"respuestas cortadas por el tope: {chequeo['respuestas_cortadas_por_tope']}\n")

    descripcion = []
    for estrato in (CORPUS, FUERA):
        print(f"Medias por condición ({estrato}); alucinación y artefactos = proporción")
        print(f"{'condición':20} pares fidelidad alucinación pertinencia artefactos" + ("  reconoce_límite no/parcial/sí" if estrato == FUERA else ""))
        for cond in ESCALERA:
            m = medias(pares, estrato, cond)
            fid = "NA" if m["fidelidad"] is None else f"{m['fidelidad']:.2f}"
            lim = "" if "reconoce_limite" not in m else f"  {m['reconoce_limite']['no']}/{m['reconoce_limite']['parcial']}/{m['reconoce_limite']['si']}"
            print(f"{cond:20} {m['pares']:>5} {fid:>8} {m['alucinacion']:>10.2f} {m['pertinencia']:>11.2f} {m['artefactos']:>9.2f}{lim}")
            descripcion.append({"estrato": estrato, "condicion": cond, **{k: v for k, v in m.items() if k != "reconoce_limite"},
                                **({"reconoce_limite": json.dumps(m["reconoce_limite"])} if "reconoce_limite" in m else {})})
        print()

    rng = random.Random(2026)
    filas = []
    for a, b in CONTRASTES + [("agente_completo", "baseline")]:
        for estrato in (CORPUS, FUERA):
            for criterio in CRITERIOS:
                c = contraste(pares, estrato, a, b, criterio, rng)
                if c:
                    primaria = estrato == CORPUS and criterio in PRIMARIOS and (a, b) in CONTRASTES
                    filas.append({"contraste": f"{a} vs {b}", "estrato": estrato, "criterio": criterio,
                                  "familia": "primaria" if primaria else "exploratoria", **c})
    prim = [f for f in filas if f["familia"] == "primaria"]
    for f, aj in zip(prim, holm([f["p"] for f in prim])):
        f["p_holm"] = round(aj, 4)

    print(f"Contrastes (positivo = mejor la primera condición). Familia primaria: {len(prim)} pruebas, Holm.")
    print(f"{'contraste | estrato | criterio':64} pares  mejora           IC 95 %  a/e/b   p(signo)  p Holm")
    for f in filas:
        nombre = f"{f['contraste']} | {f['estrato'][:5]} | {f['criterio']}"
        print(f"{nombre:64}{linea(f, f.get('p_holm'))}{'' if 'p_holm' in f else '  exploratorio'}")
    guardar("ablacion.json", {"chequeo": chequeo, "condiciones": descripcion, "contrastes": filas,
                              "familia_primaria": len(prim)})
    csv_filas("ablacion_condiciones.csv", descripcion)
    csv_filas("ablacion_contrastes.csv", [{**f, "ic95": f"{f['ic95'][0]}; {f['ic95'][1]}"} for f in filas])
    print(f"\nGuardado en {DESTINO}")


# ----------------------------------------------------------------------------- segunda revisión
def kappa_ponderado(a: list[str], b: list[str], categorias: list[str]):
    """Kappa de Cohen con pesos cuadráticos; None si no hay variación."""
    k, n = len(categorias), len(a)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(categorias)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1 / n
    fa = [sum(obs[i]) for i in range(k)]
    fb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    peso = lambda i, j: ((i - j) / (k - 1)) ** 2 if k > 1 else 0.0
    d_obs = sum(peso(i, j) * obs[i][j] for i in range(k) for j in range(k))
    d_esp = sum(peso(i, j) * fa[i] * fb[j] for i in range(k) for j in range(k))
    return None if d_esp == 0 else round(1 - d_obs / d_esp, 3)


def cmd_segunda(_) -> None:
    clave = {c["id_item"]: c for c in leer(DESTINO / "clave_ablacion.csv")}
    segunda = [f for f in leer(DESTINO / "revision_ablacion.csv") if clave[f["id_item"]]["origen"] == "repetida"]
    primera = {}
    for m in MODELOS:
        for f in leer(carpeta_modelo(m) / "revision" / "revision_ciega.csv"):
            primera[(m, f["id_item"])] = f
    salida, detalle = {}, []
    for crit, col in COL.items():
        a, b, por_modelo = [], [], {}
        for f in segunda:
            k = clave[f["id_item"]]
            v1, v2 = valor(primera[(k["modelo"], k["id_original"])][col]), valor(f[col])
            if v1 is None or v2 is None:
                continue
            a.append(v1)
            b.append(v2)
            por_modelo.setdefault(k["modelo"], []).append(v1 == v2)
        if a:
            salida[crit] = {"n": len(a), "acuerdo_exacto": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
                            "kappa_ponderado": kappa_ponderado(a, b, ORDEN[crit]),
                            "acuerdo_por_modelo": {m: round(sum(v) / len(v), 3) for m, v in por_modelo.items()}}
    for f in segunda:
        k = clave[f["id_item"]]
        p1 = primera[(k["modelo"], k["id_original"])]
        detalle.append({"id_item": f["id_item"], "modelo": k["modelo"], "id_original": k["id_original"],
                        "condicion": k["condicion"], "escenario_id": k["escenario_id"], "repeticion": k["repeticion"],
                        **{f"{c}_1": valor(p1[col]) for c, col in COL.items()},
                        **{f"{c}_2": valor(f[col]) for c, col in COL.items()}})
    cuenta = {m: sum(clave[f["id_item"]]["modelo"] == m for f in segunda) for m in MODELOS}
    print(f"Filas repetidas: {len(segunda)} ({cuenta})")
    print(f"{'criterio':16} {'n':>3} {'acuerdo':>8} {'kappa pond.':>12}   acuerdo por modelo")
    for crit, d in salida.items():
        kp = "n/d" if d["kappa_ponderado"] is None else f"{d['kappa_ponderado']:.2f}"
        print(f"{crit:16} {d['n']:>3} {d['acuerdo_exacto']:>8.0%} {kp:>12}   {d['acuerdo_por_modelo']}")
    print("\nkappa n/d: no hay variación suficiente para calcularlo (se informa solo el acuerdo).")
    guardar("segunda_revision.json", salida)
    csv_filas("segunda_revision_detalle.csv", detalle)
    print(f"Guardado en {DESTINO} (el detalle registra qué filas fueron)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="orden", required=True)
    for nombre, fn in (("v1", cmd_v1), ("ablacion", cmd_ablacion), ("segunda", cmd_segunda)):
        sub.add_parser(nombre).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()