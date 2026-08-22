import pypdf
import chromadb
import ollama

PDF_PATH = "../data/raw/texto_guia_de_inversoras_v2_0_2.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "educacion_financiera"
CHUNK_SIZE = 1000  # caracteres aproximados por chunk
CHUNK_OVERLAP = 200  # caracteres que se repiten entre chunks consecutivos


def extraer_texto(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    texto_completo = ""
    for page in reader.pages:
        texto_completo += page.extract_text() + "\n"
    return texto_completo


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


def main():
    print("Extrayendo texto del PDF...")
    texto = extraer_texto(PDF_PATH)
    print(f"Texto extraído: {len(texto)} caracteres")

    print("Dividiendo en chunks...")
    chunks = dividir_en_chunks(texto, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Se generaron {len(chunks)} chunks")

    print("Conectando a ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Generando embeddings e insertando en ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = generar_embedding(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
        )
        print(f"  Chunk {i+1}/{len(chunks)} procesado")

    print("¡Ingestión completa!")


if __name__ == "__main__":
    main()