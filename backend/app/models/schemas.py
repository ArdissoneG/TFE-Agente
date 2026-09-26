from pydantic import BaseModel
from typing import Literal, Optional, Dict

Respuesta = Literal["no", "algo", "si"]

class DiagnosticoRequest(BaseModel):
    conceptos_basicos_ahorro_inversion: Respuesta
    conceptos_basicos_riesgo_rentabilidad: Respuesta
    instrumentos_conoce_acciones_bonos: Respuesta
    instrumentos_conoce_fci_cedears: Respuesta
    sesgos_conoce_sesgos_cognitivos: Respuesta
    sesgos_distingue_regulado_no_regulado: Respuesta
    fraudes_reconoce_senales_estafa: Respuesta
    fraudes_conoce_canales_denuncia: Respuesta

class KnowledgeProfile(BaseModel):
    niveles: Dict[str, Literal["principiante", "intermedio", "avanzado"]]

class ChatRequest(BaseModel):
    message: str
    knowledge_profile: Optional[KnowledgeProfile] = None

class ChatResponse(BaseModel):
    response: str
    tema_detectado: Optional[str] = None
    nivel_aplicado: Optional[str] = None

class VerificacionRequest(BaseModel):
    tema: str
    opcion_elegida: str
    knowledge_profile: KnowledgeProfile

class VerificacionResponse(BaseModel):
    correcto: bool
    nivel_anterior: str
    nivel_nuevo: str
    knowledge_profile: KnowledgeProfile

class ResponderVerificacionRequest(BaseModel):
    thread_id: str
    opcion_elegida: str