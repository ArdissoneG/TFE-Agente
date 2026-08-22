from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse, UserProfile
from app.models.risk_profile import clasificar_perfil
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
    result = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": request.message}
        ]
    )
    reply = result["message"]["content"]
    return ChatResponse(response=reply)