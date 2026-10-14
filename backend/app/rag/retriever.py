import chromadb
import ollama
from pathlib import Path

from app.config import MODELO_EMBEDDINGS

CHROMA_PATH = str(Path(__file__).resolve().parents[2] / "chroma_db")
COLLECTION_NAME = "educacion_financiera"


def buscar_contexto(pregunta: str, n_resultados: int = 3) -> dict:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedding_pregunta = ollama.embed(model=MODELO_EMBEDDINGS, input=pregunta)["embeddings"][0]

    resultados = collection.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
    )

    chunks = resultados["documents"][0]
    metadatas = resultados["metadatas"][0]
    temas = [m["tema"] for m in metadatas]

    # tema más frecuente entre los chunks recuperados (voto por mayoría simple)
    tema_detectado = max(set(temas), key=temas.count) if temas else "general"

    return {
        "chunks": chunks,
        "tema_detectado": tema_detectado,
        # ids y distancias, en orden de ranking: hacen falta para Precision@3, Recall@3 y MRR
        "ids": resultados["ids"][0],
        "distancias": resultados["distances"][0],
    }