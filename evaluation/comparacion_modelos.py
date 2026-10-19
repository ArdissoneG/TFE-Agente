"""
Comparación pareada entre dos modelos (mismos escenarios y semillas) sobre la revisión manual ya completada.

Para cada condición (baseline / agente completo), estrato y criterio, se empareja cada respuesta del
modelo nuevo con la del modelo base para el mismo (escenario, semilla) y se calcula la mejora
modelo_nuevo - modelo_base (positivo = mejor el modelo nuevo; en alucinación y artefactos, mejor = menos).
Informa media de diferencias con IC 95 % por remuestreo y prueba del signo exacta (sin empates).
Exploratorio: pocos pares, un único evaluador, revisiones hechas por separado para cada modelo.

Uso: python evaluation/comparacion_modelos.py [--base phi3:mini] [--nuevo llama3.1:8b]
Guarda evaluation/resultados/<nuevo>/revision/comparacion_modelos.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from comparacion_pareada import CRITERIOS, bootstrap_ic, leer, numero, prueba_signo  # noqa: E402
from ejecutar_evaluacion import carpeta_modelo  # noqa: E402


def cargar(modelo: str) -> dict:
    carpeta = carpeta_modelo(modelo) / "revision"
    clave = {c["id_item"]: c for c in leer(carpeta / "clave_revision.csv")}
    filas = {}
    for fila in leer(carpeta / "revision_ciega.csv"):
        k = clave[fila["id_item"]]
        filas[(k["tipo_pregunta"], k["condicion"], k["escenario_id"], k["repeticion"])] = fila
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="phi3:mini")
    ap.add_argument("--nuevo", default="llama3.1:8b")
    ap.add_argument("--semilla", type=int, default=2026)
    args = ap.parse_args()

    base, nuevo = cargar(args.base), cargar(args.nuevo)
    comunes = sorted(set(base) & set(nuevo))
    if not comunes:
        sys.exit("No hay pares (escenario, semilla) en común entre los dos modelos.")
    rng = random.Random(args.semilla)
    salida = {}
    for estrato in ("recuperacion", "fuera_de_corpus"):
        for cond in ("baseline", "agente_completo"):
            grupo = [k for k in comunes if k[0] == estrato and k[1] == cond]
            for nombre, (col, sentido) in CRITERIOS.items():
                difs = []
                for k in grupo:
                    a, b = numero(nuevo[k][col]), numero(base[k][col])
                    if a is not None and b is not None:
                        difs.append((a - b) * sentido)
                if not difs:
                    continue
                gana, pierde = sum(d > 0 for d in difs), sum(d < 0 for d in difs)
                lo, hi = bootstrap_ic(difs, rng)
                salida[f"{estrato} | {cond} | {nombre}"] = {
                    "pares": len(difs), "mejora_media_modelo_nuevo": round(sum(difs) / len(difs), 3),
                    "ic95": [round(lo, 3), round(hi, 3)], "nuevo_mejor": gana,
                    "empates": len(difs) - gana - pierde, "base_mejor": pierde,
                    "p_signo": round(prueba_signo(gana, pierde), 4),
                }
    destino = carpeta_modelo(args.nuevo) / "revision" / "comparacion_modelos.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mejora media de {args.nuevo} respecto de {args.base} (positivo = mejor), mismos escenarios y semillas\n")
    print(f"{'estrato | condicion | criterio':52} {'pares':>5} {'mejora':>7} {'IC 95 %':>16} {'nuevo/empate/base':>18} {'p (signo)':>10}")
    for nombre, d in salida.items():
        ic = f"[{d['ic95'][0]:+.2f}, {d['ic95'][1]:+.2f}]"
        print(f"{nombre:52} {d['pares']:>5} {d['mejora_media_modelo_nuevo']:>+7.2f} {ic:>16} "
              f"{d['nuevo_mejor']:>6}/{d['empates']}/{d['base_mejor']:<9} {d['p_signo']:>10.3f}")
    print(f"\nGuardado en {destino}")


if __name__ == "__main__":
    main()