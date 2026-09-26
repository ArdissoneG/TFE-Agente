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
    try:
        res = requests.get(f"{BACKEND_URL}/perfil")
        if res.status_code == 200:
            st.session_state.knowledge_profile = res.json()["niveles"]
    except requests.exceptions.RequestException:
        pass  # sin conexión al backend; sigue el flujo de diagnóstico normal

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Reiniciar diagnóstico"):
        try:
            requests.delete(f"{BACKEND_URL}/perfil")
        except requests.exceptions.RequestException:
            pass
        st.session_state.knowledge_profile = None
        st.session_state.messages = []
        st.rerun()

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

    def responder_verificacion(idx: int, thread_id: str, opcion_id: str):
        payload = {"thread_id": thread_id, "opcion_elegida": opcion_id}
        try:
            res = requests.post(f"{BACKEND_URL}/chat/verificar", json=payload)
            res.raise_for_status()
            data = res.json()
            st.session_state.knowledge_profile = data["knowledge_profile"]["niveles"]
            st.session_state.messages[idx]["verificacion"]["resultado"] = data
        except requests.exceptions.RequestException as e:
            st.session_state.messages[idx]["verificacion"]["error"] = str(e)

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("tema_detectado"):
                st.caption(
                    f"🔎 Tema detectado: {message['tema_detectado'].replace('_', ' ')} "
                    f"· Nivel aplicado: {message['nivel_aplicado']}"
                )

                verificacion = message.get("verificacion")
                if verificacion and verificacion.get("pregunta"):
                    resultado = verificacion.get("resultado")
                    if resultado is None:
                        st.markdown("**¿Entendiste bien? Probá esta pregunta:**")
                        st.write(verificacion["pregunta"]["pregunta"])
                        opciones = verificacion["pregunta"]["opciones"]
                        etiquetas = [o["texto"] for o in opciones]
                        seleccion = st.radio(
                            "Elegí una opción",
                            options=range(len(opciones)),
                            format_func=lambda i: etiquetas[i],
                            key=f"verificacion_radio_{idx}",
                        )
                        if st.button("Verificar", key=f"verificacion_btn_{idx}"):
                            opcion_id = opciones[seleccion]["id"]
                            responder_verificacion(idx, message["thread_id"], opcion_id)
                            st.rerun()
                    else:
                        tema_legible = message["tema_detectado"].replace("_", " ")
                        if resultado["correcto"]:
                            st.success(
                                f"✅ ¡Correcto! Nivel de '{tema_legible}' "
                                f"actualizado: → {resultado['nivel_nuevo']}"
                            )
                        else:
                            st.warning(
                                f"❌ No era esa. Nivel de '{tema_legible}' "
                                f"ajustado: → {resultado['nivel_nuevo']}"
                            )
                            if resultado.get("reexplicacion"):
                                st.markdown("**Probemos de otra forma:**")
                                st.write(resultado["reexplicacion"])

    def get_response(user_input: str) -> dict:
        try:
            payload = {
                "message": user_input,
                "knowledge_profile": {"niveles": st.session_state.knowledge_profile},
            }
            res = requests.post(f"{BACKEND_URL}/chat", json=payload)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            return {
                "response": f"⚠️ No pude conectar con el backend: {e}",
                "tema_detectado": None,
                "nivel_aplicado": None,
                "thread_id": None,
                "pregunta_verificacion": None,
            }

    user_input = st.chat_input("Escribí tu consulta...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        respuesta_json = get_response(user_input)
        pregunta_verificacion = respuesta_json.get("pregunta_verificacion")
        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta_json["response"],
            "tema_detectado": respuesta_json.get("tema_detectado"),
            "nivel_aplicado": respuesta_json.get("nivel_aplicado"),
            "thread_id": respuesta_json.get("thread_id"),
            "verificacion": {"pregunta": pregunta_verificacion, "resultado": None} if pregunta_verificacion else None,
        })
        st.rerun()