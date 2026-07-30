import streamlit as st

st.title("Asistente de educación financiera")

# Inicializa el historial de mensajes una sola vez (al primer arranque de la sesión)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra el historial completo en cada re-ejecución del script
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Función mock: por ahora devuelve una respuesta fija, después la reemplazamos por Ollama
def get_mock_response(user_input: str) -> str:
    return f"(respuesta simulada) Recibí tu mensaje: '{user_input}'"

# Input del usuario, aparece fijo abajo de la pantalla
user_input = st.chat_input("Escribí tu consulta...")

if user_input:
    # Guarda y muestra el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Genera y muestra la respuesta (mockeada por ahora)
    response = get_mock_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)