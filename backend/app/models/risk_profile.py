from app.models.schemas import UserProfile

def clasificar_perfil(profile: UserProfile) -> str:
    puntaje = 0

    puntaje += {"ninguna": 0, "basica": 1, "intermedia": 2, "avanzada": 3}[profile.experiencia]
    puntaje += {"baja": 0, "media": 2, "alta": 4}[profile.tolerancia_riesgo]
    puntaje += {"corto": 0, "mediano": 2, "largo": 3}[profile.horizonte_inversion]

    if puntaje <= 3:
        return "conservador"
    elif puntaje <= 6:
        return "moderado"
    else:
        return "agresivo"