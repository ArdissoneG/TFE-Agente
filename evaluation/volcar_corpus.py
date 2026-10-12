"""
Vuelca todos los chunks de la base vectorial a evaluation/corpus_chunks.jsonl, con su id,
tema, página y fuente. Sirve para anotar las preguntas de recuperación (qué chunks son
relevantes para cada pregunta) sobre un corpus congelado.
"""
import json
from pathlib import Path

import chromadb

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CHROMA = RAIZ / "backend" / "chroma_db"
SALIDA = Path(__file__).resolve().parent / "corpus_chunks.jsonl"


def main():
    coleccion = chromadb.PersistentClient(path=str(RUTA_CHROMA)).get_collection("educacion_financiera")
    datos = coleccion.get(include=["documents", "metadatas"])
    filas = [
        {
            "id": id_,
            "tema": meta["tema"],
            "pagina": meta["pagina"],
            "fuente": meta["fuente"].split("/")[-1],
            "transcripcion": bool(meta.get("transcripcion")),
            "texto": texto,
        }
        for id_, texto, meta in zip(datos["ids"], datos["documents"], datos["metadatas"])
    ]
    filas.sort(key=lambda f: (f["fuente"], f["pagina"], f["id"]))
    with open(SALIDA, "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"{len(filas)} chunks escritos en {SALIDA}")


if __name__ == "__main__":
    main()