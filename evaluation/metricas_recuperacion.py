"""
Métricas de recuperación sobre las 20 preguntas anotadas (evaluation/consultas.json), con el
mismo recuperador que usa el agente (buscar_contexto, top-3):
  - Precision@3 = relevantes recuperados / 3
  - Recall@3    = relevantes recuperados / relevantes totales de la pregunta
  - MRR         = 1 / posición del primer relevante (0 si no hay ninguno entre los 3)
  - Coincidencia de tema = el tema detectado (voto por mayoría) coincide con el esperado
Con pocos fragmentos relevantes por pregunta (1 a 4) y k = 3, Precision@3 tiene un máximo de
|relevantes| / 3 (por ejemplo 0,33 si hay un solo relevante). Para las 10 preguntas fuera de
corpus no hay relevantes: se registran los fragmentos recuperados y sus distancias.
La recuperación es determinista: se ejecuta una sola vez. Necesita Ollama (embeddings) y la
base vectorial de backend/chroma_db.

Uso (con el venv activo y Ollama corriendo, desde la raíz del repo):
    python evaluation/metricas_recuperacion.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from consultas_util import cargar_consultas  # noqa: E402
from utils import RAIZ_EVALUACION, id_corto  # noqa: E402
from app.rag.retriever import buscar_contexto  # noqa: E402

SALIDA = RAIZ_EVALUACION / "resultados"
K = 3
TEMAS = ["conceptos_basicos", "instrumentos_inversion", "sesgos_y_riesgos", "fraudes_y_proteccion"]


def media(valores):
    return round(sum(valores) / len(valores), 3) if valores else None


def main():
    SALIDA.mkdir(exist_ok=True)
    detalle = []
    for c in cargar_consultas():
        r = buscar_contexto(c["pregunta"], n_resultados=K)
        recuperados = [id_corto(i) for i in r["ids"]]
        fila = {"id": c["id"], "tipo": c["tipo"], "pregunta": c["pregunta"],
                "recuperados": recuperados, "distancias": [round(d, 4) for d in r["distancias"]],
                "tema_detectado": r["tema_detectado"], "tema_esperado": c["tema_esperado"]}
        if c["tipo"] == "recuperacion":
            relevantes = set(c["relevantes"])
            aciertos = [i for i in recuperados if i in relevantes]
            posiciones = [n + 1 for n, i in enumerate(recuperados) if i in relevantes]
            fila.update({
                "relevantes": c["relevantes"],
                "precision_at_3": round(len(aciertos) / K, 3),
                "recall_at_3": round(len(aciertos) / len(relevantes), 3),
                "mrr": round(1 / posiciones[0], 3) if posiciones else 0.0,
                "tema_coincide": r["tema_detectado"] == c["tema_esperado"],
            })
        detalle.append(fila)

    rec = [f for f in detalle if f["tipo"] == "recuperacion"]
    fuera = [f for f in detalle if f["tipo"] == "fuera_de_corpus"]

    def resumen(filas):
        return {"preguntas": len(filas),
                "precision_at_3": media([f["precision_at_3"] for f in filas]),
                "recall_at_3": media([f["recall_at_3"] for f in filas]),
                "mrr": media([f["mrr"] for f in filas]),
                "coincidencia_tema": media([1.0 if f["tema_coincide"] else 0.0 for f in filas]),
                "sin_ningun_relevante": sum(1 for f in filas if f["mrr"] == 0.0)}

    resultado = {"k": K, "global": resumen(rec),
                 "por_tema": {t: resumen([f for f in rec if f["tema_esperado"] == t]) for t in TEMAS},
                 "distancia_top1_media": {"con_respuesta_en_corpus": media([f["distancias"][0] for f in rec]),
                                          "fuera_de_corpus": media([f["distancias"][0] for f in fuera])}}

    print(f"Recuperación sobre {len(rec)} preguntas (top-{K})")
    g = resultado["global"]
    print(f"  Precision@3 = {g['precision_at_3']}   Recall@3 = {g['recall_at_3']}   MRR = {g['mrr']}   "
          f"Coincidencia de tema = {g['coincidencia_tema']}   Sin ningún relevante: {g['sin_ningun_relevante']}")
    print("\nPor tema:")
    for t, m in resultado["por_tema"].items():
        print(f"  {t:<24} P@3={m['precision_at_3']}  R@3={m['recall_at_3']}  MRR={m['mrr']}  tema={m['coincidencia_tema']}")
    print("\nDistancia del primer resultado (menor = más cercano):")
    print(f"  preguntas con respuesta en el corpus: {resultado['distancia_top1_media']['con_respuesta_en_corpus']}")
    print(f"  preguntas fuera de corpus:            {resultado['distancia_top1_media']['fuera_de_corpus']}")
    print("\nDetalle por pregunta:")
    for f in rec:
        print(f"  {f['id']} P@3={f['precision_at_3']} R@3={f['recall_at_3']} MRR={f['mrr']} "
              f"tema={'ok' if f['tema_coincide'] else 'MAL'} {f['recuperados']}")

    with open(SALIDA / "recuperacion.json", "w", encoding="utf-8") as f:
        json.dump({"resumen": resultado, "detalle": detalle}, f, ensure_ascii=False, indent=2)
    with open(SALIDA / "recuperacion_detalle.csv", "w", encoding="utf-8", newline="") as f:
        campos = ["id", "tipo", "pregunta", "tema_esperado", "tema_detectado", "tema_coincide",
                  "precision_at_3", "recall_at_3", "mrr", "recuperados", "distancias", "relevantes"]
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for fila in detalle:
            w.writerow({**fila, "recuperados": " | ".join(fila["recuperados"]),
                        "distancias": " | ".join(map(str, fila["distancias"])),
                        "relevantes": " | ".join(fila.get("relevantes", []))})
    print(f"\nResultados en {SALIDA}")


if __name__ == "__main__":
    main()