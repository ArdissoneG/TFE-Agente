"""
Resumen de la verificación y del tratamiento del error conceptual en las corridas del agente completo.

Lee las corridas del runner (sin llamar a ningún modelo) y calcula:
  - aciertos y errores por persona y por tema, y cambios de nivel;
  - qué categorías de error conceptual aparecieron (según el distractor elegido);
  - cuántas veces se generó una nueva explicación.
Guarda evaluation/resultados/<modelo>/verificacion.json y verificacion_detalle.csv.

Uso: python evaluation/resumen_verificacion.py [--modelo phi3:mini]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from ejecutar_evaluacion import carpeta_modelo  # noqa: E402
from muestrear_revision import _evento, leer_corridas  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="phi3:mini")
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    carpeta = carpeta_modelo(args.modelo, args.prueba)

    detalle = []
    for run_id, eventos in sorted(leer_corridas(carpeta).items()):
        marca, ver = eventos[-1], _evento(eventos, "verificacion")
        if marca["condicion"] != "agente_completo" or not ver:
            continue
        detalle.append({
            "run_id": run_id, "escenario_id": marca["escenario_id"], "persona_id": marca["persona_id"],
            "repeticion": marca["seed"], "tema": ver["tema_detectado"], "pregunta_id": ver["pregunta_id"],
            "opcion_elegida": ver["opcion_elegida"], "opcion_correcta": ver["opcion_correcta"],
            "correcto": ver["correcto"], "categoria_error": ver["categoria_error"] or "",
            "nivel_anterior": ver["nivel_anterior"], "nivel_nuevo": ver["nivel_nuevo"],
            "hubo_reexplicacion": _evento(eventos, "reexplicacion") is not None,
        })
    if not detalle:
        sys.exit(f"No hay corridas del agente completo en {carpeta}.")

    por_persona = defaultdict(lambda: Counter())
    for d in detalle:
        por_persona[d["persona_id"]]["total"] += 1
        por_persona[d["persona_id"]]["aciertos"] += d["correcto"] is True
    errores = [d for d in detalle if d["correcto"] is False]
    resumen = {
        "corridas_agente": len(detalle),
        "aciertos": sum(d["correcto"] is True for d in detalle),
        "errores": len(errores),
        "sin_corregir": sum(d["correcto"] is None for d in detalle),
        "por_persona": {p: dict(c) for p, c in sorted(por_persona.items())},
        "categorias_de_error": dict(Counter(d["categoria_error"] for d in errores)),
        "errores_por_tema": dict(Counter(d["tema"] for d in errores)),
        "reexplicaciones_generadas": sum(d["hubo_reexplicacion"] for d in detalle),
        "cambios_de_nivel": dict(Counter(f"{d['nivel_anterior']}->{d['nivel_nuevo']}" for d in detalle)),
    }
    (carpeta / "verificacion.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(carpeta / "verificacion_detalle.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detalle[0].keys()))
        w.writeheader()
        w.writerows(detalle)

    print(f"Corridas del agente: {resumen['corridas_agente']}  "
          f"aciertos={resumen['aciertos']}  errores={resumen['errores']}")
    print("Por persona:", resumen["por_persona"])
    print("Categorías de error:", resumen["categorias_de_error"])
    print("Errores por tema:", resumen["errores_por_tema"])
    print("Nuevas explicaciones generadas:", resumen["reexplicaciones_generadas"])
    print("Cambios de nivel:", resumen["cambios_de_nivel"])
    print(f"Resultados en {carpeta}")


if __name__ == "__main__":
    main()