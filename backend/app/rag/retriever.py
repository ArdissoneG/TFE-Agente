import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"


def buscar_contexto(pregunta: str, n_resultados: int = 3) -> dict:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedding_pregunta = ollama.embed(model="nomic-embed-text", input=pregunta)["embeddings"][0]

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
    )

    chunks = resultados["documents"][0]
    metadatas = resultados["metadatas"][0]
    temas = [m["tema"] for m in metadatas]

    # tema más frecuente entre los chunks recuperados (voto por mayoría simple)
    tema_detectado = max(set(temas), key=temas.count) if temas else "general"

    return {"chunks": chunks, "tema_detectado": tema_detectado}