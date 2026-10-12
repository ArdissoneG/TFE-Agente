import chromadb
from collections import Counter

col = chromadb.PersistentClient(path="./chroma_db").get_collection("educacion_financiera")
metas = col.get(include=["metadatas"])["metadatas"]
print(Counter(m["tema"] for m in metas))
print(Counter((m["fuente"].split("/")[-1], m["tema"]) for m in metas))
print("transcripciones:", sum(1 for m in metas if m.get("transcripcion")))