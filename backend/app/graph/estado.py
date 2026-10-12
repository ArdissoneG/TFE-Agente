from typing import TypedDict, Optional


class EstadoChat(TypedDict):
    mensaje: str
    knowledge_profile: dict          
    tema_detectado: Optional[str]
    contexto_texto: Optional[str]
    nivel_aplicado: Optional[str]
    respuesta: Optional[str]
    pregunta_id: Optional[str]   
    respuesta_usuario: Optional[str] 
    correcto: Optional[bool]
    nivel_nuevo: Optional[str]
    reexplicacion: Optional[str]