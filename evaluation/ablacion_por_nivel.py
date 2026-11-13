"""
Desglose exploratorio de la fidelidad del agente completo frente al RAG sin nivel (llama3.1:8b).
Posterior a la revisión; no estaba en hipotesis_ablacion.md. Solo descriptivo, sin pruebas.

Uso: python evaluation/ablacion_por_nivel.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from ablacion_analisis import CORPUS, LLAMA, pares_ablacion  # noqa: E402
from ejecutar_evaluacion import carpeta_modelo  # noqa: E402

FID, ALU = "fidelidad (0-2)", "alucinacion (0/1)"


def filas() -> list[dict]:
    corridas = carpeta_modelo(LLAMA) / "corridas"
    salida = []
    for (tipo, esc, seed), p in sorted(pares_ablacion().items()):
        if tipo != CORPUS:
            continue
        ruta = corridas / f"{esc}_agente_completo_s{seed}.jsonl"
        eventos = [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines()]
        expl = next(e for e in eventos if e["tipo"] == "explicacion")
        salida.append({"escenario": esc, "semilla": seed, "nivel": expl["nivel_aplicado"],
                       "tema": expl["tema_detectado"], "persona": eventos[-1]["persona_id"],
                       "fid_rag": float(p["rag_sin_nivel"][FID]), "fid_agente": float(p["agente_completo"][FID]),
                       "alu_rag": float(p["rag_sin_nivel"][ALU]), "alu_agente": float(p["agente_completo"][ALU])})
    return salida


def tabla(datos: list[dict], clave: str) -> None:
    grupos = defaultdict(list)
    for d in datos:
        grupos[d[clave]].append(d)
    print(f"Por {clave}")
    print(f"{'grupo':26} {'pares':>5} {'fid RAG':>8} {'fid agente':>10} {'dif':>6} {'a/e/r':>7} {'alu RAG':>8} {'alu ag.':>8}")
    for g, v in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
        n = len(v)
        m = lambda c: sum(x[c] for x in v) / n
        dif = [x["fid_agente"] - x["fid_rag"] for x in v]
        print(f"{g:26} {n:>5} {m('fid_rag'):>8.2f} {m('fid_agente'):>10.2f} {m('fid_agente') - m('fid_rag'):>+6.2f} "
              f"{sum(x > 0 for x in dif)}/{sum(x == 0 for x in dif)}/{sum(x < 0 for x in dif):<3} "
              f"{m('alu_rag'):>8.2f} {m('alu_agente'):>8.2f}")
    print()


def main() -> None:
    datos = filas()
    print(f"{len(datos)} pares con respuesta en el corpus. a/e/r: agente mejor / empate / RAG sin nivel mejor (fidelidad).\n")
    for clave in ("nivel", "tema", "persona"):
        tabla(datos, clave)


if __name__ == "__main__":
    main()