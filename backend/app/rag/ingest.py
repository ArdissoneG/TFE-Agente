import pypdf
import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Cada documento define su propio mapeo de páginas a tema,
# porque cada PDF tiene su propia estructura/índice.
DOCUMENTOS = [
    {
        "path": "../data/raw/texto_guia_de_inversoras_v2_0_2.pdf",
        "rangos_tema": [
            (5, 10, "conceptos_basicos"),
            (12, 14, "instrumentos_inversion"),
            (21, 23, "sesgos_y_riesgos"),
            (24, 32, "fraudes_y_proteccion"),
        ],
    },
    {
        "path": "../data/raw/Educ. Financiera_agosto_2023.pdf",
        "rangos_tema": [
            (9, 34, "conceptos_basicos"),
        ],
    },
    # Agregar más documentos aquí
]


def tema_para_pagina(num_pagina: int, rangos_tema: list) -> str:
    for inicio, fin, tema in rangos_tema:
        if inicio <= num_pagina <= fin:
            return tema
    return "general"


def dividir_en_chunks(texto: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunks.append(texto[inicio:fin])
        inicio += chunk_size - overlap
    return chunks


def generar_embedding(texto: str) -> list[float]:
    result = ollama.embed(model="nomic-embed-text", input=texto)
    return result["embeddings"][0]


def ingestar_documento(collection, path: str, rangos_tema: list) -> int:
    print(f"\nProcesando: {path}")
    reader = pypdf.PdfReader(path)
    total_chunks = 0

    for i, page in enumerate(reader.pages):
        num_pagina = i + 1
        texto_pagina = page.extract_text()
        if not texto_pagina.strip():
            continue

        tema = tema_para_pagina(num_pagina, rangos_tema)
        chunks = dividir_en_chunks(texto_pagina, CHUNK_SIZE, CHUNK_OVERLAP)

        for j, chunk in enumerate(chunks):
            embedding = generar_embedding(chunk)
            # id único combinando nombre de archivo + página + chunk,
            # para que no choquen ids entre documentos distintos
            chunk_id = f"{path}_pagina{num_pagina}_chunk{j}"
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"tema": tema, "pagina": num_pagina, "fuente": path}],
            )
            total_chunks += 1
            print(f"  Página {num_pagina} ({tema}) - chunk {j+1}/{len(chunks)} procesado")

    return total_chunks


def main():
    print("Conectando a ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Colección anterior eliminada.")
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_general = 0
    for doc in DOCUMENTOS:
        total_general += ingestar_documento(collection, doc["path"], doc["rangos_tema"])

    print(f"\n¡Ingestión completa! Total: {total_general} chunks, de {len(DOCUMENTOS)} documento(s).")


if __name__ == "__main__":
    main()