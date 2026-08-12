import streamlit as st
import requests

st.title("Asistente de educación financiera")

# Inicializa el historial de mensajes una sola vez (al primer arranque de la sesión)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra el historial completo en cada re-ejecución del script
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

BACKEND_URL = "http://127.0.0.1:8000"

def get_response(user_input: str) -> str:
    try:
        res = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input})
        res.raise_for_status()
        return res.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"⚠️ No pude conectar con el backend: {e}"

# Input del usuario, aparece fijo abajo de la pantalla
user_input = st.chat_input("Escribí tu consulta...")

if user_input:
    # Guarda y muestra el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Genera y muestra la respuesta (mockeada por ahora)
    response = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)