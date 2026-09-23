from app.models.schemas import DiagnosticoRequest, KnowledgeProfile

PUNTOS = {"no": 0, "algo": 1, "si": 2}

def _nivel_desde_puntaje(puntaje: int) -> str:
    if puntaje <= 1:
        return "principiante"
    elif puntaje <= 3:
        return "intermedio"
    else:
        return "avanzado"

def clasificar_conocimiento(respuestas: DiagnosticoRequest) -> KnowledgeProfile:
    niveles = {
        "conceptos_basicos": _nivel_desde_puntaje(
            PUNTOS[respuestas.conceptos_basicos_ahorro_inversion]
            + PUNTOS[respuestas.conceptos_basicos_riesgo_rentabilidad]
        ),
        "instrumentos_inversion": _nivel_desde_puntaje(
            PUNTOS[respuestas.instrumentos_conoce_acciones_bonos]
            + PUNTOS[respuestas.instrumentos_conoce_fci_cedears]
        ),
        "sesgos_y_riesgos": _nivel_desde_puntaje(
            PUNTOS[respuestas.sesgos_conoce_sesgos_cognitivos]
            + PUNTOS[respuestas.sesgos_distingue_regulado_no_regulado]
        ),
        "fraudes_y_proteccion": _nivel_desde_puntaje(
            PUNTOS[respuestas.fraudes_reconoce_senales_estafa]
            + PUNTOS[respuestas.fraudes_conoce_canales_denuncia]
        ),
    }
    return KnowledgeProfile(niveles=niveles)