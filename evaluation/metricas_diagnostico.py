"""
Precisión diagnóstica: compara el nivel que asigna el sistema a cada persona (a partir de sus
respuestas al diagnóstico) con su nivel real por tema, que se fijó al diseñar la persona.
Calcula la matriz de confusión (filas = nivel real, columnas = nivel diagnosticado), la
precisión global (aciertos / comparaciones), y precisión y exhaustividad por nivel.
Es determinista: se ejecuta una sola vez. No necesita Ollama.

Uso (con el venv activo, desde la raíz del repo):  python evaluation/metricas_diagnostico.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from personas import NIVELES, cargar_personas, perfil_diagnosticado  # noqa: E402
from utils import RAIZ_EVALUACION  # noqa: E402

TEMAS = ["conceptos_basicos", "instrumentos_inversion", "sesgos_y_riesgos", "fraudes_y_proteccion"]
SALIDA = RAIZ_EVALUACION / "resultados"


def razon(a: int, b: int):
    return round(a / b, 3) if b else None


def main():
    SALIDA.mkdir(exist_ok=True)
    filas = []
    for p in cargar_personas():
        diag = perfil_diagnosticado(p)
        for tema in TEMAS:
            filas.append({
                "persona": p["id"], "autopercepcion": p["autopercepcion"].split(" ")[0], "tema": tema,
                "nivel_real": p["nivel_real"][tema], "nivel_diagnosticado": diag[tema],
                "coincide": p["nivel_real"][tema] == diag[tema],
            })

    matriz = {r: {c: 0 for c in NIVELES} for r in NIVELES}
    for f in filas:
        matriz[f["nivel_real"]][f["nivel_diagnosticado"]] += 1

    total = len(filas)
    aciertos = sum(f["coincide"] for f in filas)
    por_nivel = {}
    for n in NIVELES:
        vp = matriz[n][n]
        predichos = sum(matriz[r][n] for r in NIVELES)
        reales = sum(matriz[n].values())
        por_nivel[n] = {"precision": razon(vp, predichos), "exhaustividad": razon(vp, reales),
                        "reales": reales, "predichos": predichos}
    por_tipo = {}
    for tipo in ("calibrada", "descalibrada"):
        sub = [f for f in filas if f["autopercepcion"] == tipo]
        por_tipo[tipo] = {"comparaciones": len(sub), "aciertos": sum(f["coincide"] for f in sub),
                          "precision": razon(sum(f["coincide"] for f in sub), len(sub))}

    resultado = {"comparaciones": total, "aciertos": aciertos, "precision_global": razon(aciertos, total),
                 "matriz_confusion": matriz, "por_nivel": por_nivel, "por_autopercepcion": por_tipo}

    print(f"Precisión diagnóstica global: {aciertos}/{total} = {razon(aciertos, total)}")
    print("\nMatriz de confusión (filas = real, columnas = diagnosticado):")
    print(" " * 14 + "".join(f"{c:>14}" for c in NIVELES))
    for r in NIVELES:
        print(f"{r:<14}" + "".join(f"{matriz[r][c]:>14}" for c in NIVELES))
    print("\nPor nivel:")
    for n, m in por_nivel.items():
        print(f"  {n:<12} precisión={m['precision']}  exhaustividad={m['exhaustividad']}  (reales={m['reales']}, predichos={m['predichos']})")
    print("\nPor autopercepción de la persona:")
    for t, m in por_tipo.items():
        print(f"  {t:<13} {m['aciertos']}/{m['comparaciones']} = {m['precision']}")

    with open(SALIDA / "diagnostico.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    with open(SALIDA / "diagnostico_detalle.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nResultados en {SALIDA}")


if __name__ == "__main__":
    main()