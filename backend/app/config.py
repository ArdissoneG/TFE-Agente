import os

MODELO_LLM = os.getenv("TFE_MODELO_LLM", "phi3:mini")
MODELO_EMBEDDINGS = "nomic-embed-text"

# valores fijos para que la evaluación sea reproducible
TEMPERATURA = 0.3
NUM_CTX = 4096
NUM_PREDICT = 1000
# tope de tokens por respuesta (evita desbordes de phi3:mini).