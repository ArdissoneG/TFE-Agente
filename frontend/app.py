import streamlit as st
import requests

st.title("Asistente de educación financiera")

BACKEND_URL = "http://127.0.0.1:8000"

PREGUNTAS_DIAGNOSTICO = [
    {
        "tema": "Conceptos básicos",
        "preguntas": [
            {"key": "conceptos_basicos_ahorro_inversion", "texto": "¿Sabés diferenciar ahorro de inversión?"},
            {"key": "conceptos_basicos_riesgo_rentabilidad", "texto": "¿Entendés la relación entre riesgo y rentabilidad?"},
        ],
    },
    {
        "tema": "Instrumentos de inversión",
        "preguntas": [
            {"key": "instrumentos_conoce_acciones_bonos", "texto": "¿Sabés qué son las acciones y los bonos?"},
            {"key": "instrumentos_conoce_fci_cedears", "texto": "¿Conocés qué son los FCI y los CEDEARs?"},
        ],
    },
    {
        "tema": "Sesgos y riesgos",
        "preguntas": [
            {"key": "sesgos_conoce_sesgos_cognitivos", "texto": "¿Conocés los sesgos que pueden afectar tus decisiones de inversión?"},
            {"key": "sesgos_distingue_regulado_no_regulado", "texto": "¿Sabés distinguir un instrumento regulado de uno no regulado?"},
        ],
    },
    {
        "tema": "Fraudes y protección",
        "preguntas": [
            {"key": "fraudes_reconoce_senales_estafa", "texto": "¿Reconocés señales de una posible estafa de inversión?"},
            {"key": "fraudes_conoce_canales_denuncia", "texto": "¿Sabés a dónde recurrir si sospechás un fraude?"},
        ],
    },
]

OPCIONES = {
    "No, no lo sé": "no",
    "Algo sé, pero no estoy seguro/a": "algo",
    "Sí, lo tengo claro": "si",
}

if "knowledge_profile" not in st.session_state:
    st.session_state.knowledge_profile = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- ETAPA 1: DIAGNÓSTICO DE CONOCIMIENTO ---
if st.session_state.knowledge_profile is None:
    st.subheader("Antes de arrancar, veamos qué tanto sabés de cada tema")
    st.caption("No hay respuestas incorrectas — esto nos ayuda a explicarte con el nivel de detalle justo.")

    respuestas = {}
    for grupo in PREGUNTAS_DIAGNOSTICO:
        st.markdown(f"**{grupo['tema']}**")
        for pregunta in grupo["preguntas"]:
            seleccion = st.radio(
                pregunta["texto"],
                options=list(OPCIONES.keys()),
                key=pregunta["key"],
            )
            respuestas[pregunta["key"]] = OPCIONES[seleccion]

    if st.button("Ver mi diagnóstico"):
        try:
            res = requests.post(f"{BACKEND_URL}/diagnostico", json=respuestas)
            res.raise_for_status()
            st.session_state.knowledge_profile = res.json()["niveles"]
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"No pude calcular tu diagnóstico: {e}")


# --- ETAPA 2: CHAT LIBRE ---
else:
    st.success("Tu diagnóstico de conocimiento:")
    for tema, nivel in st.session_state.knowledge_profile.items():
        st.write(f"- **{tema.replace('_', ' ')}**: {nivel}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def get_response(user_input: str) -> str:
        try:
            payload = {
                "message": user_input,
                "knowledge_profile": {"niveles": st.session_state.knowledge_profile},
            }
            res = requests.post(f"{BACKEND_URL}/chat", json=payload)
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