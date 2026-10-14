"""
Personas ficticias de la evaluación: carga desde personas.json y comportamiento determinista
de cada una (qué perfil diagnostica el sistema y qué opción elige en la verificación).
Cada persona tiene su propio perfil: nunca se comparte ni se guarda en SQLite.
"""
import json
import sys
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION.parent / "backend"))

from app.models.knowledge_profile import clasificar_conocimiento  # noqa: E402
from app.models.schemas import DiagnosticoRequest  # noqa: E402
from app.verification import preguntas as banco  # noqa: E402

RUTA_PERSONAS = RAIZ_EVALUACION / "personas.json"
NIVELES = ["principiante", "intermedio", "avanzado"]


def cargar_personas() -> list[dict]:
    with open(RUTA_PERSONAS, encoding="utf-8") as f:
        return json.load(f)["personas"]


def perfil_diagnosticado(persona: dict) -> dict:
    """Perfil que calcula el sistema a partir de las respuestas al diagnóstico de la persona."""
    request = DiagnosticoRequest(**persona["respuestas_diagnostico"])
    return dict(clasificar_conocimiento(request).niveles)


def acierta(persona: dict, pregunta_id: str) -> bool:
    pregunta = banco._por_id(pregunta_id)
    nivel_real = persona["nivel_real"][pregunta["tema"]]
    if nivel_real == "avanzado":
        return True
    if nivel_real == "principiante":
        return False
    posicion = [p["id"] for p in banco._por_tema(pregunta["tema"])].index(pregunta_id)
    return posicion % 2 == 0


def opcion_elegida(persona: dict, pregunta_id: str) -> str:
    """Id de la opción ('a'..'d') que elige la persona en una pregunta de verificación."""
    pregunta = banco._por_id(pregunta_id)
    opciones = pregunta["opciones"]
    if acierta(persona, pregunta_id):
        return next(o["id"] for o in opciones if o["categoria_error"] is None)
    distractores = [o for o in opciones if o["categoria_error"] is not None]
    preferidos = [o for o in distractores if o["categoria_error"] == persona["error_preferido"]]
    return (preferidos or distractores)[0]["id"]