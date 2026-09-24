import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"


def buscar_con_filtro(pregunta: str, tema: str, n_resultados: int = 2):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedding_pregunta = ollama.embed(model="nomic-embed-text", input=pregunta)["embeddings"][0]

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
        where={"tema": tema},
    )

    print(f"\nPregunta: {pregunta}  |  Filtrado por tema: {tema}\n")
    for i, doc in enumerate(resultados["documents"][0]):
        metadata = resultados["metadatas"][0][i]
        print(f"--- Resultado {i+1} (página {metadata['pagina']}) ---")
        print(doc[:200])
        print()


if __name__ == "__main__":
    buscar_con_filtro("¿qué son los CEDEARs?", tema="instrumentos_inversion")
    buscar_con_filtro("¿qué son los CEDEARs?", tema="fraudes_y_proteccion")