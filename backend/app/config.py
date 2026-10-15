import os

MODELO_LLM = os.getenv("TFE_MODELO_LLM", "phi3:mini")
MODELO_EMBEDDINGS = "nomic-embed-text"

# Temperatura y tamaño de contexto fijos y explícitos (antes se usaban los valores
# por defecto de Ollama, que no quedaban registrados en ningún lado).
# TEMPERATURA: confirmar que coincide con el valor que declara el capítulo 3 del documento.
TEMPERATURA = 0.3
NUM_CTX = 4096
NUM_PREDICT = 1000
# Tope de tokens que puede generar el modelo en cada respuesta. Sin tope, phi3:mini a veces se
# desborda (miles de tokens, más que el contexto), y eso dejó al servidor de Ollama generando texto
# sin sentido durante el resto de la corrida. Con este valor, prompt + respuesta entran en NUM_CTX.
# Es el mismo para el baseline y el agente completo.