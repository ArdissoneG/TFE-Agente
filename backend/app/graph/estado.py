from typing import TypedDict, Optional


class EstadoChat(TypedDict):
    mensaje: str
    knowledge_profile: dict          
    # Solo en evaluación (semilla, condición, etc.)
    meta: Optional[dict]
    tema_detectado: Optional[str]
    contexto_texto: Optional[str]
    chunk_ids: Optional[list]
    distancias: Optional[list]
    nivel_aplicado: Optional[str]
    respuesta: Optional[str]
    pregunta_id: Optional[str]
    respuesta_usuario: Optional[str]   
    calificacion: Optional[dict]
    correcto: Optional[bool]
    nivel_nuevo: Optional[str]
    reexplicacion: Optional[str]