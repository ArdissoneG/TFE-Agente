"""
Comparación pareada baseline vs agente completo sobre la revisión manual ya completada.

Cada fila de revision_ciega.csv es una respuesta; las dos condiciones se revisaron sobre los mismos
pares (escenario, semilla). Para cada criterio y estrato se calcula, par por par, la diferencia
agente - baseline y se informa:
  - media de las diferencias, con intervalo de confianza del 95 % por remuestreo (bootstrap);
  - pares en que gana el agente / empatan / gana el baseline, y p-valor de la prueba del signo exacta
    (bilateral, sin los empates). En alucinación y artefactos, "mejor" significa 0 (no hay).
Es una comparación exploratoria: son pocos pares, un único evaluador, y algunos escenarios se repiten
con otra semilla, así que los pares no son del todo independientes.

Uso: python evaluation/comparacion_pareada.py [--modelo phi3:mini]
Guarda evaluation/resultados/<modelo>/revision/comparacion_pareada.json
"""
import argparse
import csv
import json
import random
import sys
from math import comb
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from ejecutar_evaluacion import carpeta_modelo  # noqa: E402

# criterio -> (columna, sentido): +1 si un valor más alto es mejor, -1 si uno más bajo es mejor
CRITERIOS = {
    "fidelidad": ("fidelidad (0-2)", +1),
    "alucinacion": ("alucinacion (0/1)", -1),
    "pertinencia": ("pertinencia (0-2)", +1),
    "artefactos": ("artefactos (0/1)", -1),
}


def leer(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numero(valor: str):
    try:
        return float(valor.replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


def prueba_signo(a_favor: int, en_contra: int) -> float:
    """p-valor bilateral exacto de la prueba del signo (binomial con p = 0,5), sin empates."""
    n = a_favor + en_contra
    if n == 0:
        return 1.0
    k = min(a_favor, en_contra)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ic(difs: list[float], rng: random.Random, remuestreos: int = 10000):
    n = len(difs)
    medias = sorted(sum(rng.choice(difs) for _ in range(n)) / n for _ in range(remuestreos))
    return medias[int(0.025 * remuestreos)], medias[int(0.975 * remuestreos) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="phi3:mini")
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--semilla", type=int, default=2026, help="semilla del remuestreo (reproducible)")
    args = ap.parse_args()
    carpeta = carpeta_modelo(args.modelo, args.prueba) / "revision"

    clave = {c["id_item"]: c for c in leer(carpeta / "clave_revision.csv")}
    pares = {}  # (estrato, escenario, repeticion) -> {condicion: fila}
    for fila in leer(carpeta / "revision_ciega.csv"):
        k = clave[fila["id_item"]]
        pares.setdefault((k["tipo_pregunta"], k["escenario_id"], k["repeticion"]), {})[k["condicion"]] = fila

    rng = random.Random(args.semilla)
    salida = {}
    for estrato in ("recuperacion", "fuera_de_corpus"):
        grupo = [p for (t, _, _), p in pares.items() if t == estrato and len(p) == 2]
        for nombre, (col, sentido) in CRITERIOS.items():
            difs = []
            for p in grupo:
                a, b = numero(p["agente_completo"][col]), numero(p["baseline"][col])
                if a is not None and b is not None:
                    difs.append((a - b) * sentido)  # positivo = el agente es mejor
            if not difs:
                continue
            gana, pierde = sum(d > 0 for d in difs), sum(d < 0 for d in difs)
            lo, hi = bootstrap_ic(difs, rng)
            salida[f"{estrato} | {nombre}"] = {
                "pares": len(difs),
                "mejora_media_del_agente": round(sum(difs) / len(difs), 3),
                "ic95": [round(lo, 3), round(hi, 3)],
                "agente_mejor": gana, "empates": len(difs) - gana - pierde, "baseline_mejor": pierde,
                "p_signo": round(prueba_signo(gana, pierde), 4),
            }

    (carpeta / "comparacion_pareada.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Mejora media del agente (positivo = mejor), pares con respuesta en el corpus y fuera de corpus\n")
    print(f"{'estrato | criterio':38} {'pares':>5} {'mejora':>7} {'IC 95 %':>16} {'agente/empate/base':>19} {'p (signo)':>10}")
    for nombre, d in salida.items():
        ic = f"[{d['ic95'][0]:+.2f}, {d['ic95'][1]:+.2f}]"
        print(f"{nombre:38} {d['pares']:>5} {d['mejora_media_del_agente']:>+7.2f} {ic:>16} "
              f"{d['agente_mejor']:>6}/{d['empates']}/{d['baseline_mejor']:<10} {d['p_signo']:>10.3f}")
    print(f"\nGuardado en {carpeta / 'comparacion_pareada.json'}")


if __name__ == "__main__":
    main()