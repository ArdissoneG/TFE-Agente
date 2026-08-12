from fastapi import FastAPI
from app.models.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Agente educación financiera - API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Mock por ahora - en el Sprint 3 esto va a llamar a Ollama
    mock_reply = f"(respuesta simulada del backend) Recibí: '{request.message}'"
    return ChatResponse(response=mock_reply)