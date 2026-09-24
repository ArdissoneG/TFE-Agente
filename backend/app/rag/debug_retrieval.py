import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"


def debug_busqueda(pregunta: str, n_resultados: int = 3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedding_pregunta = ollama.embed(model="nomic-embed-text", input=pregunta)["embeddings"][0]

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
    )

    print(f"\nPregunta: {pregunta}\n")
    for i, doc in enumerate(resultados["documents"][0]):
        metadata = resultados["metadatas"][0][i]
        distancia = resultados["distances"][0][i]
        print(f"--- Resultado {i+1} | página {metadata['pagina']} | tema: {metadata['tema']} | distancia: {distancia:.4f} ---")
        print(doc[:150])
        print()


if __name__ == "__main__":
    debug_busqueda("¿qué es la renta fija y la renta variable?")