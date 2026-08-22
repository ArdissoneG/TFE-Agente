from pydantic import BaseModel
from typing import Literal, Optional

class UserProfile(BaseModel):
    experiencia: Literal["ninguna", "basica", "intermedia", "avanzada"]
    tolerancia_riesgo: Literal["baja", "media", "alta"]
    horizonte_inversion: Literal["corto", "mediano", "largo"]
    monto_aproximado: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    profile: Optional[UserProfile] = None

class ChatResponse(BaseModel):
    response: str

