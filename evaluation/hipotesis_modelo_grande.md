# Hipótesis previas a la corrida con llama3.1:8b

Fecha de redacción: 19/10/2026, antes de ejecutar la corrida.
Misma metodología que phi3:mini: 30 escenarios, 2 condiciones, semillas 1/2/3,
mismos parámetros (temperatura 0,3; num_ctx 4096; num_predict 1000), mismo corpus.

H1. Menor tasa de artefactos de generación que con phi3:mini (en ambas condiciones).
H2. Menor tasa de alucinación que con phi3:mini.
H3. Las reexplicaciones corrigen el error con más frecuencia (más "sí" que los 11/30 de phi3:mini).
H4. La ventaja del agente sobre el baseline en pertinencia se mantiene.
H5. Las limitaciones de diseño persisten: sin mecanismo de abstención fuera del corpus
    y desbalance del corpus (no dependen del tamaño del modelo).