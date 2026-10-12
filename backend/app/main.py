from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse, DiagnosticoRequest, KnowledgeProfile, VerificacionRequest, VerificacionResponse, ResponderVerificacionRequest
from app.models.knowledge_profile import clasificar_conocimiento
from app.verification.preguntas import armar_pregunta
from app.baseline import responder_baseline
from app.rag.retriever import buscar_contexto
from app.db import inicializar_db, guardar_perfil, obtener_perfil, borrar_perfil
import ollama
import uuid
from langgraph.types import Command
from app.graph.grafo import grafo

USUARIO_ID = "default"

app = FastAPI(title="Agente educación financiera - API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    inicializar_db()

@app.get("/perfil")
def obtener_perfil_guardado():
    niveles = obtener_perfil(USUARIO_ID)
    if niveles is None:
        raise HTTPException(status_code=404, detail="No hay un perfil guardado todavía")
    return {"niveles": niveles}

@app.delete("/perfil")
def eliminar_perfil():
    borrar_perfil(USUARIO_ID)
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/diagnostico", response_model=KnowledgeProfile)
def diagnostico(request: DiagnosticoRequest):
    perfil = clasificar_conocimiento(request)
    guardar_perfil(USUARIO_ID, perfil.niveles)
    return clasificar_conocimiento(request)

@app.post("/chat")
def chat(request: ChatRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    estado_inicial = {
        "mensaje": request.message,
        "knowledge_profile": request.knowledge_profile.niveles if request.knowledge_profile else {},
    }
    resultado = grafo.invoke(estado_inicial, config=config)

    interrupcion = resultado["__interrupt__"][0].value
    pregunta = armar_pregunta(interrupcion["pregunta_id"])

    return {
        "thread_id": thread_id,
        "response": resultado["respuesta"],
        "tema_detectado": resultado["tema_detectado"],
        "nivel_aplicado": resultado["nivel_aplicado"],
        "pregunta_verificacion": pregunta,
        }


@app.post("/chat/verificar")
def responder_verificacion(request: ResponderVerificacionRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    resultado = grafo.invoke(Command(resume=request.opcion_id), config=config)

    calificacion = resultado.get("calificacion") or {}
    return {
        "correcto": resultado["correcto"],
        "nivel_nuevo": resultado["nivel_nuevo"],
        "knowledge_profile": {"niveles": resultado["knowledge_profile"]},
        "reexplicacion": resultado.get("reexplicacion"),
        "opcion_elegida": calificacion.get("opcion_elegida"),
        "opcion_correcta": calificacion.get("opcion_correcta"),
        "explicacion": calificacion.get("explicacion"),
    }

    #return {
    #    "correcto": resultado["correcto"],
    #    "nivel_nuevo": resultado["nivel_nuevo"],
    #    "knowledge_profile": {"niveles": resultado["knowledge_profile"]},
    #   "reexplicacion": resultado.get("reexplicacion"),
    #}


@app.post("/chat/baseline")
def chat_baseline(request: ChatRequest):
    # Caso base de la evaluación: mismo modelo y parámetros, sin RAG, nivel ni verificación.
    return responder_baseline(request.message)