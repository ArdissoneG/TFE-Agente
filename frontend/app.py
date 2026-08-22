import streamlit as st
import requests

st.title("Asistente de educación financiera")

BACKEND_URL = "http://127.0.0.1:8000"

PREGUNTAS_PERFIL = [
    {
        "key": "experiencia",
        "texto": "¿Cuál es tu experiencia invirtiendo?",
        "opciones": {
            "Nunca invertí": "ninguna",
            "Invertí alguna vez, pero sin mucho conocimiento": "basica",
            "Tengo conocimientos y experiencia moderada": "intermedia",
            "Invierto de forma activa y conozco bien el tema": "avanzada",
        },
    },
    {
        "key": "tolerancia_riesgo",
        "texto": "Si tu inversión bajara un 10% en un mes, ¿qué harías?",
        "opciones": {
            "La retiraría de inmediato para no perder más": "baja",
            "Esperaría un tiempo antes de decidir": "media",
            "No me preocuparía, es parte del proceso": "alta",
        },
    },
    {
        "key": "horizonte_inversion",
        "texto": "¿Por cuánto tiempo pensás mantener esta inversión?",
        "opciones": {
            "Menos de 1 año": "corto",
            "Entre 1 y 5 años": "mediano",
            "Más de 5 años": "largo",
        },
    },
]

if "perfil_completo" not in st.session_state:
    st.session_state.perfil_completo = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- ETAPA 1: ENTREVISTA DE PERFILADO ---
if st.session_state.perfil_completo is None:
    st.subheader("Antes de arrancar, contanos un poco sobre vos")

    respuestas = {}
    for pregunta in PREGUNTAS_PERFIL:
        seleccion = st.radio(
            pregunta["texto"],
            options=list(pregunta["opciones"].keys()),
            key=pregunta["key"],
        )
        respuestas[pregunta["key"]] = pregunta["opciones"][seleccion]

    monto = st.number_input("¿Con qué monto aproximado pensás invertir? (opcional)", min_value=0.0, step=1000.0)

    if st.button("Confirmar perfil"):
        payload = {
            "experiencia": respuestas["experiencia"],
            "tolerancia_riesgo": respuestas["tolerancia_riesgo"],
            "horizonte_inversion": respuestas["horizonte_inversion"],
            "monto_aproximado": monto if monto > 0 else None,
        }
        try:
            res = requests.post(f"{BACKEND_URL}/profile", json=payload)
            res.raise_for_status()
            clasificacion = res.json()["clasificacion"]
            st.session_state.perfil_completo = clasificacion
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"No pude calcular tu perfil: {e}")


# --- ETAPA 2: CHAT LIBRE ---
else:
    st.success(f"Tu perfil de riesgo es: **{st.session_state.perfil_completo}**")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def get_response(user_input: str) -> str:
        try:
            res = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input})
            res.raise_for_status()
            return res.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"⚠️ No pude conectar con el backend: {e}"

    user_input = st.chat_input("Escribí tu consulta...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        response = get_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)