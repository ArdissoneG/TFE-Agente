"""Carga de consultas.json (preguntas de recuperación y fuera de corpus)."""
import json

from utils import RAIZ_EVALUACION

RUTA_CONSULTAS = RAIZ_EVALUACION / "consultas.json"


def cargar_consultas() -> list[dict]:
    with open(RUTA_CONSULTAS, encoding="utf-8") as f:
        return json.load(f)["consultas"]