import os

MODELO_LLM = os.getenv("TFE_MODELO_LLM", "phi3:mini")
MODELO_EMBEDDINGS = "nomic-embed-text"

# Temperatura y tamaño de contexto fijos y explícitos (antes se usaban los valores
# por defecto de Ollama, que no quedaban registrados en ningún lado).
# TEMPERATURA: confirmar que coincide con el valor que declara el capítulo 3 del documento.
TEMPERATURA = 0.3
NUM_CTX = 4096