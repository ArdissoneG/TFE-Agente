# Rúbrica de revisión manual

Se aplica a las 60 respuestas de `revision_ciega.csv` (30 por condición) y a las nuevas explicaciones de
`revision_reexplicaciones.csv`. La condición de cada respuesta no se conoce durante la revisión.

**Material de referencia.** Para las preguntas con respuesta en el corpus, la columna
`material_de_referencia` trae los fragmentos anotados como relevantes (los mismos para las dos
condiciones, no los que recuperó el sistema). Para las preguntas fuera de corpus no hay material:
se verifica con conocimiento propio o con fuentes oficiales.

## Columnas

**fidelidad (0–2)** — solo preguntas con respuesta en el corpus (en las otras va `NA`).
- 2: todo lo que afirma es consistente con el material; nada importante lo contradice.
- 1: mezcla afirmaciones respaldadas con otras que el material no respalda (agregados, datos o
  ejemplos propios), sin contradecirlo en lo central.
- 0: contradice el material, o lo central de la respuesta no se apoya en él.
Una analogía que no afirma hechos no baja la fidelidad.

**alucinacion (0/1)** — vale para todas las preguntas.
- 1: afirma como hecho algo falso o inventado y verificable: una entidad, sigla, cifra, norma,
  producto o procedimiento que no existe o que no es como se dice (por ejemplo, desarrollar "UVA"
  como "Unidad Valor Adicional" o nombrar bancos y plataformas que el material no menciona).
- 0: no hay afirmaciones de ese tipo. Una metáfora evidente no cuenta.

**pertinencia (0–2)**
- 2: responde lo que se preguntó, con claridad y de forma adecuada para un inversor minorista argentino.
- 1: responde en parte, se desvía del tema o resulta confusa.
- 0: no responde lo que se preguntó.

**artefactos (0/1)** — fallas de generación del modelo, que no se cuentan como error de fidelidad.
- 1: hay caracteres corruptos, de otro alfabeto o palabras deformadas ("poten0cial", "futuноси").
- 0: no hay.
Solo afecta la pertinencia si impide entender el texto.

**reconoce_limite (si/parcial/no)** — solo preguntas fuera de corpus (en las otras va `NA`).
- si: dice que no tiene información suficiente o remite a una fuente oficial, sin inventar detalles.
- parcial: lo insinúa pero igual da datos específicos que nada respalda.
- no: responde con seguridad como si lo supiera.

## Nuevas explicaciones tras un error (`revision_reexplicaciones.csv`)

- **corrige_error (si/parcial/no):** si la explicación deja en claro por qué la opción elegida es
  incorrecta y qué es lo correcto, según la opción correcta que figura en la fila.
- **reproduce_error (si/no):** si repite o refuerza el error conceptual de la opción elegida.
- **alucinacion** y **artefactos:** igual que arriba.

## Procedimiento

1. Completar `revision_ciega.csv` en orden, sin abrir `clave_revision.csv`.
2. Dejar las dudas en `observaciones`; no corregir criterios a mitad de camino. Si hay que cambiar un
   criterio, se anota y se vuelve a revisar todo con el criterio nuevo.
3. Para estimar la consistencia del único evaluador, volver a revisar unas 10 filas elegidas al azar
   después de una semana y calcular el acuerdo con la primera revisión (se declara como limitación).
4. Al terminar: `python evaluation/resumen_revision.py`.