from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse, UserProfile
from app.models.risk_profile import clasificar_perfil
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

@app.post("/profile")
def profile(user_profile: UserProfile):
    clasificacion = clasificar_perfil(user_profile)
    return {"clasificacion": clasificacion}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    contexto_chunks = buscar_contexto(request.message)
    contexto_texto = "\n\n".join(contexto_chunks)

    prompt_sistema = (
        "Sos un asistente de educación financiera para inversores minoristas en Argentina. "
        "Respondé de forma clara y en lenguaje simple, apoyándote en el siguiente contexto extraído "
        "de material oficial de educación financiera. Si el contexto no tiene información suficiente "
        "para responder con precisión, decilo explícitamente en vez de inventar una respuesta.\n\n"
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
    return ChatResponse(response=reply)