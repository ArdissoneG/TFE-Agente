from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse, DiagnosticoRequest, KnowledgeProfile
from app.models.knowledge_profile import clasificar_conocimiento
from app.rag.retriever import buscar_contexto
import ollama

app = FastAPI(title="Agente educación financiera - API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/diagnostico", response_model=KnowledgeProfile)
def diagnostico(request: DiagnosticoRequest):
    return clasificar_conocimiento(request)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    resultado_busqueda = buscar_contexto(request.message)
    contexto_chunks = resultado_busqueda["chunks"]
    tema_detectado = resultado_busqueda["tema_detectado"]
    contexto_texto = "\n\n".join(contexto_chunks)

    nivel_usuario = "intermedio"  # valor por defecto si no hay diagnóstico o el tema no está en niveles
    if request.knowledge_profile:
        nivel_usuario = request.knowledge_profile.niveles.get(tema_detectado, "intermedio")

    instrucciones_nivel = {
        "principiante": "Explicá en lenguaje simple, sin jerga técnica. Usá analogías cotidianas cuando ayuden. No asumas conocimientos previos.",
        "intermedio": "Explicá con cierto nivel técnico, pero aclarando los términos clave la primera vez que aparecen.",
        "avanzado": "Respondé de forma directa y técnica, sin explicar conceptos básicos que ya se asumen conocidos.",
    }

    prompt_sistema = (
        "Sos un asistente de educación financiera para inversores minoristas en Argentina. "
        f"El usuario tiene un nivel de conocimiento '{nivel_usuario}' en el tema de esta pregunta. "
        f"{instrucciones_nivel[nivel_usuario]} "
        "Apoyate en el siguiente contexto extraído de material oficial de educación financiera. "
        "Si el contexto no tiene información suficiente para responder con precisión, decilo "
        "explícitamente en vez de inventar una respuesta.\n\n"
        f"Contexto:\n{contexto_texto}"
    )

    result = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": request.message},
        ]
    )
    reply = result["message"]["content"]
    return ChatResponse(response=reply, tema_detectado=tema_detectado, nivel_aplicado=nivel_usuario)