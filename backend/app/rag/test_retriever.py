import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"


def buscar(pregunta: str, n_resultados: int = 3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedding_pregunta = ollama.embed(model="nomic-embed-text", input=pregunta)["embeddings"][0]

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
    )

    print(f"\nPregunta: {pregunta}\n")
    for i, doc in enumerate(resultados["documents"][0]):
        print(f"--- Resultado {i+1} ---")
        print(doc[:300])  # primeros 300 caracteres para no saturar la terminal
        print()


if __name__ == "__main__":
    buscar("¿Qué es el perfil de riesgo de un inversor?")