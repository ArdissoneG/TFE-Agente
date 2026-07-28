# Agente inteligente para educación financiera

TFE — Máster Universitario en Transformación Digital a través de Tecnologías Disruptivas (UNIR).

Agente conversacional para asistir a inversores minoristas con conocimientos limitados en:
comprensión de perfiles de riesgo, conceptos financieros y alternativas de inversión,
mediante interacción en lenguaje natural.

> Este proyecto es educativo/experimental. No constituye asesoramiento financiero regulado.

## Estado del proyecto

En desarrollo — ver [`docs/retro_semanal.md`](docs/retro_semanal.md) para el avance semana a semana.

## Stack (tentativo)

- **Backend:** FastAPI
- **LLM local:** Ollama
- **Orquestación del agente:** LangGraph
- **Vector store / RAG:** ChromaDB
- **Frontend:** por definir (React o Streamlit)

## Como usarlo

_

## Estructura

```
backend/    API, agente, RAG, modelos
frontend/   Interfaz de chat
data/       Corpus y personas de usuario
evaluation/ Escenarios y métricas de evaluación
docs/       Arquitectura, decisiones técnicas, bitácora
```
