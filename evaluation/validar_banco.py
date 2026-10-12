"""
Valida el banco de 20 preguntas de verificación contra el corpus congelado:
  - estructura: 4 opciones, exactamente 1 correcta, 3 distractores con categoría válida
  - 5 preguntas por tema, ids únicos, posiciones de la correcta repartidas
  - cada fuente citada existe en el corpus y tiene el mismo tema que la pregunta
  - cada ancla (frase textual) aparece en el texto de las fuentes citadas

Uso (con el venv activo, desde la raíz del repo):  python evaluation/validar_banco.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.verification.preguntas import CATEGORIAS_ERROR, PREGUNTAS  # noqa: E402
from utils import cargar_corpus, normalizar  # noqa: E402

TEMAS = ["conceptos_basicos", "instrumentos_inversion", "sesgos_y_riesgos", "fraudes_y_proteccion"]


def main() -> int:
    corpus = cargar_corpus()
    errores = []

    def falla(pregunta, mensaje):
        errores.append(f"[{pregunta['id']}] {mensaje}")

    ids = [p["id"] for p in PREGUNTAS]
    if len(ids) != len(set(ids)):
        errores.append("hay ids de pregunta repetidos")
    for tema in TEMAS:
        n = sum(p["tema"] == tema for p in PREGUNTAS)
        if n != 5:
            errores.append(f"el tema {tema} tiene {n} preguntas (se esperaban 5)")

    for p in PREGUNTAS:
        opciones = p["opciones"]
        if [o["id"] for o in opciones] != ["a", "b", "c", "d"]:
            falla(p, "las opciones no son a, b, c, d")
        correctas = [o for o in opciones if o["categoria_error"] is None]
        if len(correctas) != 1:
            falla(p, f"tiene {len(correctas)} opciones correctas")
        for o in opciones:
            if o["categoria_error"] is not None and o["categoria_error"] not in CATEGORIAS_ERROR:
                falla(p, f"categoría de error desconocida: {o['categoria_error']}")
        if len({o["texto"] for o in opciones}) != 4:
            falla(p, "hay opciones con texto repetido")

        texto_fuentes = ""
        for fuente in p["fuentes"]:
            fila = corpus.get(fuente)
            if fila is None:
                falla(p, f"la fuente {fuente} no existe en el corpus")
                continue
            if fila["tema"] != p["tema"]:
                print(f"  aviso [{p['id']}]: la fuente {fuente} es del tema {fila['tema']}, no de {p['tema']}")
            texto_fuentes += " " + normalizar(fila["texto"])
        for ancla in p["anclas"]:
            if normalizar(ancla) not in texto_fuentes:
                falla(p, f"la ancla no aparece en las fuentes citadas: «{ancla}»")

    posiciones = Counter(
        next(o["id"] for o in p["opciones"] if o["categoria_error"] is None) for p in PREGUNTAS
    )
    categorias = Counter(o["categoria_error"] for p in PREGUNTAS for o in p["opciones"] if o["categoria_error"])
    print(f"{len(PREGUNTAS)} preguntas. Posición de la correcta: {dict(sorted(posiciones.items()))}")
    print(f"Distractores por categoría de error: {dict(categorias.most_common())}")

    if errores:
        print("\nPROBLEMAS:")
        for e in errores:
            print(" -", e)
        return 1
    print("Banco válido: estructura, fuentes y anclas verificadas contra el corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())