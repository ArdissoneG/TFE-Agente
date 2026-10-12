from typing import TypedDict, Optional


class EstadoChat(TypedDict):
    mensaje: str
    knowledge_profile: dict          # {"tema": "nivel", ...}
    # Solo se completa al evaluar: seed, run_id, escenario_id, repeticion, condicion,
    # persistir (False = no guardar el perfil en SQLite), usuario_id, ruta_log.
    # En el uso normal de la app queda vacío.
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