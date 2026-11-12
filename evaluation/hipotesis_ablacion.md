# Ablación del agente: diseño, hipótesis y plan estadístico (previos a la corrida)

Fecha de redacción: 11/11/2026, antes de generar las condiciones nuevas.
Modelo: llama3.1:8b (la ablación no se repite con phi3:mini).
Versión del sistema evaluado: etiqueta `v1-evaluada` (commit cdc43e8). El código nuevo solo agrega
las dos condiciones; no modifica el agente ni el baseline.

## Condiciones

Escalera acumulativa: cada condición agrega un elemento a la anterior. Mismo modelo, mismos parámetros
(temperatura 0,3; num_ctx 4096; num_predict 1000), semillas 1, 2 y 3, mismos 30 escenarios.

| # | Condición | Prompt de sistema | RAG | Nivel |
|---|---|---|---|---|
| 1 | `baseline` (ya ejecutada) | línea de rol | no | no |
| 2 | `prompt_honestidad` | rol + instrucción de no inventar y de decir cuando falta información | no | no |
| 3 | `rag_sin_nivel` | rol + contexto recuperado + la misma instrucción | sí | no |
| 4 | `agente_completo` (ya ejecutada) | rol + nivel + contexto + la misma instrucción | sí | sí |

Texto de la instrucción en la condición 2: "Si no tenés información suficiente para responder con
precisión, decilo explícitamente en vez de inventar una respuesta." En las condiciones 3 y 4 es la oración
que ya tiene el agente ("Si el contexto no tiene información suficiente…"). El prompt de la condición 3 es
el del agente sin las dos oraciones de nivel; recupera los mismos fragmentos. No se agrega "explicá de forma
clara y didáctica" para no modificar el agente ya evaluado: el estilo didáctico no se aísla.

Alcance: se comparan las explicaciones. La verificación y la reexplicación no modifican la explicación
inicial y no se aíslan en este diseño.

## Revisión

Manual, ciega y pareada, con la misma rúbrica (`rubrica.md`), sin cambios de criterio.
- Se revisan las respuestas de las condiciones 2 y 3 en los mismos 30 pares (escenario, semilla) ya revisados
  para las condiciones 1 y 4 (24 con respuesta en el corpus, 6 fuera de corpus). Las puntuaciones de 1 y 4 se
  reutilizan.
- Se mezclan con 20 respuestas ya puntuadas (10 de phi3:mini y 10 de llama3.1:8b, elegidas al azar con
  semilla 20261025) para estimar la consistencia del evaluador. Cuáles son queda registrado en
  `clave_ablacion.csv`.
- El evaluador no ve modelo ni condición. El orden es aleatorio (semilla 2026).
- Texto degenerado: mismo criterio que la v1 (se descarta, se registra y se repite).

## Hipótesis

Preguntas con respuesta en el corpus (24 pares). Criterios primarios: fidelidad, alucinación y pertinencia.

- **H-A1 (RAG).** La condición 3 mejora la fidelidad y reduce la alucinación respecto de la 2.
- **H-A2 (prompt).** La condición 2 no mejora la fidelidad ni la alucinación respecto de la 1, porque la
  instrucción no aporta información.
- **H-A3 (nivel).** La condición 4 mejora la pertinencia respecto de la 3, sin cambios en fidelidad ni
  alucinación.

Criterios de lectura: una hipótesis de mejora se considera respaldada si el contraste va en la dirección
prevista y el p ajustado es menor que 0,05. Para H-A2 y H-A3, que predicen ausencia de efecto, la falta de
significación no demuestra equivalencia (no se prueba equivalencia); se interpreta con el intervalo de
confianza. Con 24 pares la potencia es limitada.

Exploratorio, sin hipótesis previa y sin ajuste: artefactos, preguntas fuera de corpus, reconocimiento del
límite y el contraste 4 vs 1.

## Plan estadístico

- Unidad: par (escenario, semilla). Diferencia entre condiciones por par; positivo = mejor la condición
  más completa (en alucinación y artefactos, mejor es 0).
- Intervalo de confianza del 95 % por bootstrap (10 000 remuestreos, semilla 2026) y prueba del signo exacta
  bilateral, sin empates, como en la v1.
- Contrastes planificados: 2 vs 1, 3 vs 2 y 4 vs 3.
- Familia primaria: 9 pruebas (3 contrastes x fidelidad, alucinación y pertinencia, en preguntas con
  respuesta en el corpus). Corrección de Holm sobre esas 9.
- Los resultados de la v1 (agente vs baseline) se recalculan con Holm. Es un ajuste posterior al análisis
  original y se informa como tal. Familia: 4 pruebas por modelo (fidelidad, alucinación, pertinencia y
  artefactos, en preguntas con respuesta en el corpus) y, como alternativa, las 8 pruebas de ambos modelos.
  Las preguntas fuera de corpus (6 pares) quedan como exploratorias.
- Las comparaciones entre modelos (`comparacion_modelos.py`) son exploratorias y no se ajustan; si se
  presentan como confirmatorias, se ajustan con Holm sobre su familia.
- Segunda revisión: acuerdo exacto y kappa ponderado cuadrático por criterio, en las 20 filas repetidas.
  Estima la estabilidad del propio evaluador; no reemplaza el acuerdo entre evaluadores.

## Comandos

    python evaluation/ejecutar_evaluacion.py --modelo llama3.1:8b --condiciones prompt_honestidad,rag_sin_nivel
    python evaluation/ablacion_revision.py
    python evaluation/ablacion_analisis.py ablacion | segunda | v1