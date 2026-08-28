import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"


def buscar_contexto(pregunta: str, n_resultados: int = 3) -> list[str]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedding_pregunta = ollama.embed(model="nomic-embed-text", input=pregunta)["embeddings"][0]

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
    )

    return resultados["documents"][0]