# Agente inteligente para educación financiera

TFE — Máster Universitario en Transformación Digital a través de Tecnologías Disruptivas (UNIR).

Agente conversacional educativo para inversores minoristas argentinos con conocimientos limitados.
Diagnostica el nivel de conocimiento por tema, explica con lenguaje adaptado a ese nivel apoyándose en
material oficial (RAG), verifica la comprensión con una pregunta de opción múltiple y, si la persona se
equivoca, vuelve a explicar corrigiendo ese error puntual.

> Proyecto educativo/experimental. No constituye asesoramiento financiero regulado.

## Cómo funciona

1. **Diagnóstico:** 8 preguntas cerradas (no / algo / sí), dos por tema, que dan un nivel
   (principiante / intermedio / avanzado) para cada uno de los cuatro temas: conceptos básicos,
   instrumentos de inversión, sesgos y riesgos, fraudes y protección.
2. **Explicación:** el agente recupera los 3 fragmentos más cercanos de un corpus oficial, detecta el
   tema (voto por mayoría) y genera la respuesta con el nivel de la persona en ese tema.
3. **Verificación:** una pregunta de opción múltiple de un banco fijo de 20 (con distractores
   etiquetados por tipo de error). El modelo de lenguaje **no** genera preguntas ni respuestas
   correctas: la corrección es determinista, contra la clave del banco.
4. **Ajuste:** el nivel sube o baja; si hay error, se genera una nueva explicación que usa el tipo de
   error cometido y el enunciado correcto del banco.

## Stack

- **Backend:** FastAPI + Pydantic
- **Orquestación:** LangGraph (con `interrupt()` para pausar en la verificación)
- **LLM local:** Ollama, modelo `phi3:mini` (temperatura 0,3, contexto 4096, máximo 1000 tokens)
- **Embeddings:** `nomic-embed-text` (Ollama)
- **Base vectorial:** ChromaDB (persistente en `backend/chroma_db`)
- **Persistencia del perfil:** SQLite
- **Frontend:** Streamlit

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com) instalado y en ejecución

## Instalación (Windows, PowerShell)

```powershell
ollama pull phi3:mini
ollama pull nomic-embed-text

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
deactivate

cd ..\frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
deactivate
```

## Corpus

Los PDF **no se versionan**. Descargalos y copialos a `data/raw/` con estos nombres:

| Archivo | Fuente |
|---|---|
| `texto_guia_de_inversoras_v2_0_2.pdf` | Guía de protección a las personas inversoras, CNV. [Descarga](https://www.argentina.gob.ar/sites/default/files/texto_guia_de_inversoras_v2.pdf) (verificar que sea la versión usada) |
| `Educ. Financiera_agosto_2023.pdf` | Cuaderno de Educación Financiera, Ministerio de Educación de Santa Fe – BCRA (2023). <!-- COMPLETAR: enlace de descarga --> |

Las páginas de la guía de la CNV que no se extraen bien (la tabla de sesgos de la 21 y las páginas
28 y 29) están transcriptas a mano en `data/transcripciones/` y sí se versionan. El corpus de la
evaluación quedó congelado en 116 fragmentos (conceptos básicos 72, fraudes y protección 26,
instrumentos de inversión 9, sesgos y riesgos 9).

Para construir la base vectorial (**desde la carpeta `backend`**, con Ollama en ejecución):

```powershell
cd backend
venv\Scripts\activate
python app/rag/ingest.py
```

## Uso

En dos terminales, con Ollama en ejecución:

```powershell
# Backend (puerto 8000)
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

```powershell
# Frontend (puerto 8501)
cd frontend
venv\Scripts\activate
streamlit run app.py
```

## Evaluación

Todos los scripts se corren desde la raíz del repositorio con el entorno virtual **del backend**.
Los que llaman al modelo necesitan Ollama en ejecución.

```powershell
backend\venv\Scripts\activate

python evaluation/validar_banco.py          # el banco de preguntas contra el corpus
python evaluation/validar_instrumentos.py   # personas y preguntas de evaluación
python evaluation/metricas_diagnostico.py   # precisión del diagnóstico y matriz de confusión
python evaluation/metricas_recuperacion.py  # Precision@3, Recall@3, MRR, coincidencia de tema

python evaluation/ejecutar_evaluacion.py    # baseline vs agente, 30 preguntas x 3 semillas
python evaluation/resumen_verificacion.py   # resumen de la verificación
python evaluation/muestrear_revision.py     # planillas para la revisión manual
# (revisión manual con la rúbrica de evaluation/rubrica.md)
python evaluation/resumen_revision.py       # cifras de la revisión
python evaluation/comparacion_pareada.py    # comparación pareada baseline vs agente
```

Los resultados quedan en `evaluation/resultados/<modelo>/`. Con `--modelo <nombre>` el runner repite el
mismo experimento con otro modelo de Ollama, en su propia carpeta.

## Estructura

```
backend/      API, agente (LangGraph), RAG, banco de verificación
frontend/     Interfaz de chat (Streamlit)
data/         Transcripciones versionadas; los PDF del corpus no se versionan
evaluation/   Personas, preguntas, scripts de métricas y resultados
```