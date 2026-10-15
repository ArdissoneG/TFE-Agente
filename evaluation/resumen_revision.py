"""
Resume la revisión manual una vez completada revision_ciega.csv (y revision_reexplicaciones.csv).

Cruza las filas con clave_revision.csv para saber la condición, y calcula por condición y estrato
(con respuesta en el corpus / fuera de corpus):
  fidelidad media (0-2), % con alucinación, pertinencia media (0-2), % con artefactos de generación,
  y, para fuera de corpus, cuántas respuestas reconocen el límite del material.
También resume las nuevas explicaciones tras un error (si corrigen o reproducen el error).
Las celdas vacías o "NA" no se cuentan; el script avisa cuántas filas quedaron sin completar.

Uso: python evaluation/resumen_revision.py [--modelo phi3:mini]
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


def leer(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numero(valor: str):
    try:
        return float(valor.replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


def media(valores: list[float]):
    return round(sum(valores) / len(valores), 3) if valores else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="phi3:mini")
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    carpeta = carpeta_modelo(args.modelo, args.prueba) / "revision"

    filas = leer(carpeta / "revision_ciega.csv")
    clave = {c["id_item"]: c for c in leer(carpeta / "clave_revision.csv")}

    grupos = defaultdict(list)
    sin_completar = 0
    for f in filas:
        k = clave[f["id_item"]]
        if numero(f["pertinencia (0-2)"]) is None or numero(f["alucinacion (0/1)"]) is None:
            sin_completar += 1
            continue
        grupos[(k["condicion"], k["tipo_pregunta"])].append(f)
    if sin_completar:
        print(f"AVISO: {sin_completar} filas sin completar (pertinencia o alucinación vacías); se excluyen.\n")

    resumen = {}
    for (cond, tipo), items in sorted(grupos.items()):
        fid = [numero(i["fidelidad (0-2)"]) for i in items]
        halu = [numero(i["alucinacion (0/1)"]) for i in items]
        per = [numero(i["pertinencia (0-2)"]) for i in items]
        art = [numero(i["artefactos (0/1)"]) for i in items]
        entrada = {
            "n": len(items),
            "fidelidad_media": media([x for x in fid if x is not None]),
            "alucinacion_tasa": media([x for x in halu if x is not None]),
            "pertinencia_media": media([x for x in per if x is not None]),
            "artefactos_tasa": media([x for x in art if x is not None]),
        }
        if tipo == "fuera_de_corpus":
            entrada["reconoce_limite"] = dict(Counter(i["reconoce_limite (si/parcial/no)"].strip().lower()
                                                      for i in items))
        resumen[f"{cond} | {tipo}"] = entrada

    reexp = {}
    ruta_re = carpeta / "revision_reexplicaciones.csv"
    if ruta_re.exists():
        filas_re = [f for f in leer(ruta_re) if f["corrige_error (si/parcial/no)"].strip()]
        reexp = {
            "n": len(filas_re),
            "corrige_error": dict(Counter(f["corrige_error (si/parcial/no)"].strip().lower() for f in filas_re)),
            "reproduce_error": dict(Counter(f["reproduce_error (si/no)"].strip().lower() for f in filas_re)),
            "alucinacion_tasa": media([numero(f["alucinacion (0/1)"]) for f in filas_re
                                       if numero(f["alucinacion (0/1)"]) is not None]),
            "artefactos_tasa": media([numero(f["artefactos (0/1)"]) for f in filas_re
                                      if numero(f["artefactos (0/1)"]) is not None]),
            "por_categoria_de_error": dict(Counter(f["categoria_error"] for f in filas_re)),
        }

    salida = {"respuestas": resumen, "reexplicaciones": reexp}
    (carpeta / "resumen_revision.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    for nombre, datos in resumen.items():
        print(nombre, datos)
    if reexp:
        print("\nNuevas explicaciones tras un error:", reexp)
    print(f"\nGuardado en {carpeta / 'resumen_revision.json'}")


if __name__ == "__main__":
    main()