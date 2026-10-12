"""
Banco de preguntas de verificación: 20 preguntas de opción múltiple (5 por tema).

Cada pregunta tiene una opción correcta y tres distractores. Cada distractor lleva una
categoría de error conceptual (ver CATEGORIAS_ERROR), de modo que elegir una opción
incorrecta permite saber qué tipo de error cometió la persona. Nada de esto lo genera el
modelo de lenguaje: el enunciado, las opciones, la clave de corrección y el texto de la
explicación están escritos de antemano a partir del corpus. Cada pregunta indica en
"fuentes" los fragmentos del corpus que la respaldan y en "anclas" frases textuales de
esos fragmentos (evaluation/validar_banco.py comprueba que existan).
"""
from functools import lru_cache

import numpy as np
import ollama

from app.config import MODELO_EMBEDDINGS

CATEGORIAS_ERROR = {
    "riesgo_nulo": "Cree que existe ganancia, seguridad o protección sin riesgo ni condiciones.",
    "relacion_invertida": "Invierte o niega la relación entre dos conceptos (por ejemplo, riesgo y rentabilidad).",
    "confusion_conceptos": "Confunde el concepto con otro cercano.",
    "equivalencia_indebida": "Trata como iguales dos cosas que son distintas.",
    "generalizacion_excesiva": "Convierte una regla en absoluta (usa \"siempre\", \"nunca\" o \"solo\" sin que corresponda).",
    "senal_superficial": "Toma una señal superficial (apariencia, fama, ausencia de aviso) como prueba de legitimidad.",
}


def _pregunta(id_, tema, enunciado, posicion_correcta, correcta, distractores,
              explicacion, fuentes, anclas, contexto_semantico):
    """Arma una pregunta; la opción correcta queda en la posición indicada (0 = A ... 3 = D)."""
    textos = [(texto, categoria) for categoria, texto in distractores]
    textos.insert(posicion_correcta, (correcta, None))
    return {
        "id": id_,
        "tema": tema,
        "pregunta": enunciado,
        "opciones": [
            {"id": "abcd"[i], "texto": texto, "categoria_error": categoria}
            for i, (texto, categoria) in enumerate(textos)
        ],
        "explicacion": explicacion,
        "fuentes": fuentes,
        "anclas": anclas,
        "contexto_semantico": contexto_semantico,
    }


PREGUNTAS = [
    # ------------------------------------------------------------------ conceptos_basicos
    _pregunta(
        "conceptos_basicos_ahorro_vs_inversion", "conceptos_basicos",
        "Según el material, ¿cuál es la diferencia principal entre ahorrar e invertir?",
        2,
        "Con el ahorro se busca tener el dinero disponible en todo momento; con la inversión se busca un mayor rendimiento a cambio de asumir ciertos riesgos y de esperar un tiempo para acceder al dinero.",
        [
            ("riesgo_nulo", "Invertir rinde más que ahorrar y no implica ningún riesgo, por eso siempre conviene invertir."),
            ("equivalencia_indebida", "No hay diferencia: ahorrar e invertir son lo mismo y solo cambia el monto de dinero."),
            ("relacion_invertida", "Ahorrar rinde más que invertir, porque el dinero siempre está disponible."),
        ],
        "Mediante el ahorro se busca la disponibilidad del dinero en todo momento; con la inversión se intenta conseguir rendimientos más altos a cambio de asumir ciertos riesgos y de acceder al dinero luego de un período de tiempo.",
        ["CUA_p25_c0", "CUA_p25_c1"],
        ["la disponibilidad del dinero en todo momento", "a cambio de asumir ciertos riesgos"],
        "diferencia entre ahorrar e invertir, disponibilidad del dinero y rendimiento",
    ),
    _pregunta(
        "conceptos_basicos_riesgo_rentabilidad", "conceptos_basicos",
        "Te ofrecen una inversión \"segura\" con una rentabilidad muy superior a la del resto del mercado. ¿Qué indica el material?",
        0,
        "Que hay que desconfiar: a mayor rentabilidad, mayor riesgo, y esta regla no tiene excepción.",
        [
            ("riesgo_nulo", "Que es posible: existen inversiones seguras de alta rentabilidad si se elige bien la empresa."),
            ("relacion_invertida", "Que una rentabilidad alta es señal de que la inversión es más segura."),
            ("relacion_invertida", "Que el riesgo no tiene relación con la rentabilidad: depende solo del plazo."),
        ],
        "Cuanto mayor es la rentabilidad que ofrece una inversión, mayor es su nivel de riesgo; la regla no tiene excepción, y hay que desconfiar de las ofertas que prometen inversiones seguras con alta rentabilidad.",
        ["CNV_p6_c0", "CNV_p6_c1"],
        ["Esta regla no tiene excepción", "desconfiar siempre de ofertas que prometen inversiones seguras con alta rentabilidad"],
        "relación entre riesgo y rentabilidad, desconfiar de rendimientos muy altos prometidos como seguros",
    ),
    _pregunta(
        "conceptos_basicos_diversificacion", "conceptos_basicos",
        "¿En qué consiste la diversificación de inversiones?",
        3,
        "En repartir el dinero entre varias inversiones con distintas características, para que si una pierde las otras puedan compensar la pérdida.",
        [
            ("riesgo_nulo", "En combinar inversiones de modo que se garantice no perder dinero aunque el mercado caiga."),
            ("relacion_invertida", "En concentrar el dinero en la inversión más rentable para maximizar la ganancia."),
            ("confusion_conceptos", "En comprar muchas unidades del mismo activo para que el riesgo se reparta."),
        ],
        "La diversificación consiste en repartir el dinero entre varias inversiones (\"no poner todos los huevos en la misma canasta\"). No garantiza que no haya pérdidas si el mercado cae, pero puede mejorar las posibilidades de no perder dinero.",
        ["CNV_p9_c0", "CNV_p9_c1"],
        ["no poner todos los huevos en la misma canasta", "La diversificación no puede garantizar que las inversiones no sufran si el mercado cae"],
        "diversificación de inversiones, repartir el dinero entre varios activos para diluir el riesgo",
    ),
    _pregunta(
        "conceptos_basicos_interes_compuesto", "conceptos_basicos",
        "Depositás $100 en un plazo fijo con renovación mensual. ¿Qué diferencia hay entre el interés compuesto y el interés simple?",
        1,
        "En el compuesto los intereses se reinvierten junto con el capital y generan nuevos intereses; en el simple se retiran y el interés se calcula siempre sobre el capital inicial.",
        [
            ("relacion_invertida", "El simple rinde más, porque cada mes se cobran los intereses."),
            ("confusion_conceptos", "En el compuesto la tasa de interés nominal es más alta que en el simple."),
            ("equivalencia_indebida", "No hay diferencia en el resultado: al cabo de un año se obtiene la misma cantidad de intereses."),
        ],
        "En el interés compuesto los intereses no se retiran sino que se suman al capital inicial y se reinvierten; en el simple se retiran. En el ejemplo del material, $100 durante un año dejan $42,58 de intereses con interés compuesto y $36,00 con interés simple.",
        ["CUA_p31_c2", "CUA_p33_c0"],
        ["son reinvertidos", "$42,58", "$36,00"],
        "interés simple e interés compuesto, capitalización, reinversión de los intereses",
    ),
    _pregunta(
        "conceptos_basicos_plazo_fijo_uva", "conceptos_basicos",
        "¿Qué es un plazo fijo UVA?",
        2,
        "Un depósito en pesos expresado en Unidades de Valor Adquisitivo (UVA), cuyo capital se ajusta según la evolución de los precios (CER).",
        [
            ("confusion_conceptos", "Un depósito en pesos a tasa fija, cuyo monto no se ajusta por la evolución de los precios."),
            ("confusion_conceptos", "Un depósito en dólares cuyo rendimiento depende del tipo de cambio."),
            ("confusion_conceptos", "Una inversión de renta variable cuyo rendimiento depende de la cotización de las acciones de una empresa."),
        ],
        "Son depósitos en pesos expresados en Unidades de Valor Adquisitivo (UVA) y actualizables por el Coeficiente de Estabilización de Referencia (CER), es decir, ajustables a la evolución de precios.",
        ["CUA_p29_c2"],
        ["depósitos en pesos expresados en Unidades de Valor Adquisitivo (UVA)", "ajustable a la evolución de precios"],
        "plazo fijo UVA, unidad de valor adquisitivo, ajuste por inflación, CER",
    ),

    # ------------------------------------------------------------------ instrumentos_inversion
    _pregunta(
        "instrumentos_inversion_renta_fija_variable", "instrumentos_inversion",
        "¿Qué distingue a los valores negociables de renta fija de los de renta variable?",
        1,
        "En la renta fija el emisor se compromete a pagar intereses y devolver el capital en momentos y montos prefijados; en la renta variable la rentabilidad no está preestablecida.",
        [
            ("riesgo_nulo", "La renta fija no tiene ningún riesgo, mientras que la renta variable sí lo tiene."),
            ("relacion_invertida", "En la renta fija la rentabilidad es desconocida de antemano y en la renta variable se conoce al comprar."),
            ("generalizacion_excesiva", "La renta fija la emite solo el Estado y la renta variable solo las empresas."),
        ],
        "Los de renta fija son emitidos por el sector público o por empresas, que se comprometen a pagar intereses y devolver el capital en momentos y montos prefijados; en los de renta variable la rentabilidad no está prefijada y depende de la evolución de la sociedad emisora o del precio de mercado.",
        ["CNV_p12_c0", "CNV_p13_c0"],
        ["devolver el capital recibido en momentos y montos prefijados", "su rentabilidad no está preestablecida o prefijada"],
        "renta fija y renta variable, bonos y acciones, rentabilidad prefijada o no",
    ),
    _pregunta(
        "instrumentos_inversion_tasas_precio", "instrumentos_inversion",
        "Cuando suben las tasas de interés, ¿qué pasa en general con el precio de un título de renta fija en el mercado secundario?",
        3,
        "El precio baja (y cuando las tasas bajan, el precio sube).",
        [
            ("relacion_invertida", "El precio sube junto con las tasas."),
            ("riesgo_nulo", "No cambia, porque la renta fija tiene pagos prefijados."),
            ("confusion_conceptos", "Solo cambia al llegar el vencimiento del título."),
        ],
        "Los tipos de interés afectan los precios de la renta fija en el mercado secundario: cuando suben, el precio de la renta fija baja, y cuando bajan, el precio aumenta.",
        ["CNV_p13_c0"],
        ["cuando suben los tipos de interés, el precio de la renta fija baja"],
        "riesgo de tipo de interés, precio de los bonos y tasas de interés, mercado secundario",
    ),
    _pregunta(
        "instrumentos_inversion_riesgo_credito", "instrumentos_inversion",
        "¿A qué se refiere el riesgo de crédito (o de insolvencia) en la renta fija?",
        0,
        "A que el emisor no pueda pagar los cupones o devolver el capital, o que se retrase en hacerlo.",
        [
            ("confusion_conceptos", "A que el precio del título baje cuando suben las tasas de interés."),
            ("confusion_conceptos", "A la dificultad de vender el título rápidamente y a un precio cierto."),
            ("confusion_conceptos", "A que la inflación le quite poder de compra a los intereses cobrados."),
        ],
        "El riesgo de crédito o de insolvencia es el riesgo de que el emisor no pueda hacer frente a sus pagos (cupones o reembolso del capital) o de que se produzca un retraso en ellos.",
        ["CNV_p13_c0"],
        ["es el riesgo de que el emisor del valor no pueda hacer frente a sus pagos"],
        "riesgo de crédito, insolvencia del emisor, riesgos de la renta fija",
    ),
    _pregunta(
        "instrumentos_inversion_cedear", "instrumentos_inversion",
        "¿Qué es un CEDEAR?",
        2,
        "Un certificado que permite negociar acciones de empresas extranjeras en el mercado y en la moneda local.",
        [
            ("confusion_conceptos", "Un bono de deuda emitido por el Estado argentino."),
            ("confusion_conceptos", "Un certificado electrónico de un depósito a plazo fijo."),
            ("confusion_conceptos", "Una cuotaparte de un fondo común de inversión."),
        ],
        "Los Certificados de Depósito Argentino (CEDEAR) permiten a la persona inversora negociar acciones de empresas extranjeras en el mercado y moneda local; el material los incluye entre los activos de renta variable.",
        ["CNV_p13_c1", "CNV_p13_c2"],
        ["Certificados de Depósito Argentino (CEDEAR)", "negociar acciones de empresas extranjeras en el mercado y moneda local"],
        "qué es un CEDEAR, certificado de depósito argentino, acciones extranjeras",
    ),
    _pregunta(
        "instrumentos_inversion_renta_variable_riesgos", "instrumentos_inversion",
        "Según el material, ¿cuál de estas afirmaciones sobre la renta variable es correcta?",
        1,
        "Su rentabilidad no es conocida ni predecible, y el comportamiento pasado no garantiza su evolución futura.",
        [
            ("riesgo_nulo", "Si una acción rindió bien en el pasado, su rendimiento futuro está asegurado."),
            ("generalizacion_excesiva", "Si una empresa tiene buenos resultados, el valor de su acción siempre aumenta en el mercado."),
            ("confusion_conceptos", "Tiene una fecha de vencimiento en la que se devuelve el capital invertido."),
        ],
        "En la renta variable no hay una rentabilidad conocida ni predecible, el pasado no garantiza el futuro, el precio depende también de factores ajenos a la compañía (economía, otros mercados, tasas, inflación) y no hay plazo de vencimiento: la inversión se deshace vendiendo el activo.",
        ["CNV_p14_c0"],
        ["no garantiza su evolución futura", "No tienen plazo de vencimiento"],
        "riesgos de la renta variable, acciones, rentabilidad incierta, el pasado no garantiza el futuro",
    ),

    # ------------------------------------------------------------------ sesgos_y_riesgos
    _pregunta(
        "sesgos_y_riesgos_sesgos_generales", "sesgos_y_riesgos",
        "¿Qué plantea el material sobre los sesgos que afectan las decisiones de inversión?",
        0,
        "Que conviene conocerlos y ser consciente de que nadie está exento de ellos.",
        [
            ("generalizacion_excesiva", "Que solo afectan a quienes recién empiezan a invertir y desaparecen con la experiencia."),
            ("riesgo_nulo", "Que con una buena estrategia se pueden eliminar por completo."),
            ("generalizacion_excesiva", "Que no influyen en las decisiones, porque estas se toman con datos objetivos."),
        ],
        "Las personas inversoras deben informarse sobre los sesgos que afectan su toma de decisiones y ser conscientes de que nadie está exento de ellos.",
        ["CNV_p21_c0"],
        ["nadie está exento de ellos"],
        "sesgos cognitivos que afectan las decisiones de inversión, comportamiento del inversor",
    ),
    _pregunta(
        "sesgos_y_riesgos_instrumentos_no_regulados", "sesgos_y_riesgos",
        "¿Qué son los instrumentos o servicios no regulados?",
        3,
        "Los que no están definidos como valor negociable por la ley del mercado de capitales y no tienen una reglamentación específica y estricta que los regule en el país.",
        [
            ("equivalencia_indebida", "Los que están prohibidos por la ley."),
            ("generalizacion_excesiva", "Los que son siempre una estafa."),
            ("confusion_conceptos", "Los que emiten únicamente empresas extranjeras."),
        ],
        "Son los instrumentos que no están definidos como valor negociable según la ley del mercado de capitales y que tampoco poseen una ley o reglamentación específica y estricta que los regule en el país.",
        ["CNV_p22_c1"],
        ["tampoco poseen una ley o reglamentación específica y estricta"],
        "instrumentos regulados y no regulados, valores negociables, ley del mercado de capitales",
    ),
    _pregunta(
        "sesgos_y_riesgos_proteccion_cnv", "sesgos_y_riesgos",
        "Si invertís en un producto que no está bajo la competencia de la CNV, ¿qué ocurre con la protección de este organismo?",
        1,
        "La CNV no tiene facultades para ejercer sus funciones de regulador y de protección de las personas inversoras sobre esos productos.",
        [
            ("riesgo_nulo", "La CNV igualmente te protege y puede intervenir como si fuera un instrumento regulado."),
            ("riesgo_nulo", "La CNV garantiza la devolución de lo invertido en cualquier instrumento."),
            ("confusion_conceptos", "La protección la ejerce el Banco Central en lugar de la CNV."),
        ],
        "Al no encontrarse bajo el ámbito de competencia y autorización de la CNV, este organismo no tiene facultades para ejercer sus funciones de regulador ni de protección de quienes invierten en esos productos o servicios.",
        ["CNV_p22_c2"],
        ["no tiene facultades para ejercer sus funciones de regulador y de protección"],
        "protección de la CNV a los inversores, productos fuera del ámbito de la CNV",
    ),
    _pregunta(
        "sesgos_y_riesgos_caracteristicas_no_regulados", "sesgos_y_riesgos",
        "¿Cuál es una característica de los instrumentos no regulados que señala el material?",
        0,
        "Se negocian sin un mercado abierto ni un mecanismo centralizador, y pueden operar las 24 horas, sufrir períodos de iliquidez y tener mayores spreads.",
        [
            ("relacion_invertida", "Solo se negocian en el horario del mercado de capitales, con un mecanismo centralizador."),
            ("riesgo_nulo", "Tienen liquidez garantizada, porque se pueden vender en cualquier momento."),
            ("relacion_invertida", "Tienen spreads más chicos que los valores negociables regulados."),
        ],
        "Por lo general se negocian sin una reglamentación estricta, ni en un mercado abierto, y sin un mecanismo centralizador; muchas veces se operan las veinticuatro horas del día, con posibles períodos de iliquidez y mayores spreads.",
        ["CNV_p22_c1"],
        ["las veinticuatro horas del día", "mecanismo centralizador"],
        "características de los productos no regulados, liquidez, spreads, mercado abierto",
    ),
    _pregunta(
        "sesgos_y_riesgos_lista_no_regulados", "sesgos_y_riesgos",
        "El material lista productos no regulados (criptomonedas, CFD, Forex, entre otros). Si un producto no aparece en esa lista, ¿qué significa?",
        3,
        "Nada en particular: la lista no es exhaustiva y su ausencia no implica pronunciamiento sobre la legalidad o conformidad del producto.",
        [
            ("senal_superficial", "Que el producto está autorizado por la CNV."),
            ("generalizacion_excesiva", "Que el producto está regulado, porque la lista incluye todos los no regulados."),
            ("equivalencia_indebida", "Que el producto está prohibido por la normativa."),
        ],
        "La lista no es exhaustiva y deriva de búsquedas en Internet y consultas recibidas; la no inclusión de un producto no implica pronunciamiento alguno sobre su conformidad o legalidad.",
        ["CNV_p23_c0"],
        ["no es exhaustiva", "no implica pronunciamiento alguno"],
        "lista de productos no regulados, criptoactivos, CFD, Forex, plataformas de trading",
    ),

    # ------------------------------------------------------------------ fraudes_y_proteccion
    _pregunta(
        "fraudes_y_proteccion_senales_estafa", "fraudes_y_proteccion",
        "¿Cuál de estas es una señal de advertencia de una posible estafa de inversión?",
        2,
        "Que te presionen para decidir rápido, porque si no se pierde la oportunidad.",
        [
            ("relacion_invertida", "Que la persona o empresa esté inscripta en la CNV."),
            ("senal_superficial", "Que presente folletos y certificados de aspecto profesional, lo que prueba que la oferta es legítima."),
            ("relacion_invertida", "Que la oferta detalle los riesgos y las comisiones."),
        ],
        "Es una señal de alerta que se afirme la necesidad de tomar una decisión rápida o se pierde la oportunidad, que la persona o empresa no esté inscripta en la CNV, o que muestre folletos de aspecto profesional cuya información no está disponible en la Autopista de la Información Financiera.",
        ["CNV_p25_c0"],
        ["tome una decisión rápida", "No está inscripta/matriculada en la CNV"],
        "señales de advertencia de estafas de inversión, presión para decidir rápido",
    ),
    _pregunta(
        "fraudes_y_proteccion_ponzi_piramide", "fraudes_y_proteccion",
        "¿Qué diferencia a un esquema Ponzi de un esquema piramidal?",
        0,
        "En el Ponzi el estafador usa el dinero de nuevos inversores para pagar supuestas ganancias a los anteriores, sin que estos deban reclutar; en la pirámide el esquema solo crece si se incorporan continuamente nuevos participantes.",
        [
            ("equivalencia_indebida", "Son lo mismo: en ambos los inversores deben reclutar a otros para cobrar."),
            ("confusion_conceptos", "En el Ponzi el dinero se invierte realmente en un negocio y las ganancias son genuinas."),
            ("relacion_invertida", "En el Ponzi los inversores deben reclutar nuevos inversores; en la pirámide no hace falta."),
        ],
        "En un esquema Ponzi el estafador usa el dinero de nuevas personas inversoras para pagar supuestas ganancias a las anteriores, en lugar de invertirlo, y quienes invierten no suelen tener que reclutar; el esquema piramidal, en cambio, solo puede crecer si se incorporan continuamente nuevos participantes.",
        ["CNV_p25_c2", "CNV_p26_c0", "CNV_p26_c1"],
        ["lo utiliza para pagar supuestas ganancias", "no suelen tener que reclutar nuevos inversores", "se incorporan, continuamente, nuevos participantes"],
        "esquemas Ponzi y piramidales, tipos de fraude en las inversiones",
    ),
    _pregunta(
        "fraudes_y_proteccion_pump_and_dump", "fraudes_y_proteccion",
        "¿En qué consiste el fraude llamado \"pump-and-dump\" (inflar y desechar)?",
        1,
        "Alguien compra a bajo precio un activo poco cotizado, difunde información falsa para subir su precio y lo vende caro, dejando a otros con un activo sin valor.",
        [
            ("equivalencia_indebida", "Una estrategia legítima de compra de acciones de empresas pequeñas para mantenerlas a largo plazo."),
            ("confusion_conceptos", "Un engaño en el que se pide una cuota por adelantado para comprar acciones sin valor a un precio alto."),
            ("confusion_conceptos", "Una estafa que se aprovecha de la confianza dentro de un grupo, como una comunidad religiosa o profesional."),
        ],
        "En el pump-and-dump alguien compra deliberadamente acciones de una empresa pequeña o criptoactivos a un precio muy bajo, difunde información falsa para aumentar el interés y el precio, y luego se deshace de los activos y desaparece.",
        ["CNV_p26_c1", "CNV_p26_c2"],
        ["difunde información falsa", "se deshace de los activos al precio elevado y desaparece"],
        "pump and dump, inflar y desechar, fraude con acciones de empresas pequeñas y criptoactivos",
    ),
    _pregunta(
        "fraudes_y_proteccion_personas_famosas", "fraudes_y_proteccion",
        "Una persona famosa recomienda públicamente una inversión. ¿Qué indica el material?",
        2,
        "Que su respaldo no significa que la inversión sea legítima ni apropiada, y que no conviene decidir solo porque alguien la promocione.",
        [
            ("senal_superficial", "Que si la recomienda alguien con reputación es porque verificó que cumple con la ley."),
            ("senal_superficial", "Que es confiable cuando quien la recomienda pertenece a tu mismo grupo (profesional, religioso, etc.)."),
            ("generalizacion_excesiva", "Que solo hay riesgo si la recomienda un desconocido."),
        ],
        "El respaldo de una persona famosa no significa que una inversión sea legítima o apropiada: los famosos pueden ser engañados y no suelen tener experiencia para garantizar que cumpla las leyes. Nunca es buena idea decidir solo porque un promotor diga que es una buena inversión.",
        ["CNV_p27_c0", "CNV_p27_c1", "CNV_p27_c2"],
        ["no significa que una inversión sea legítima", "Nunca es buena idea tomar una decisión de inversión sólo porque"],
        "fraude avalado por personas famosas o influencers, fraude por afinidad",
    ),
    _pregunta(
        "fraudes_y_proteccion_denuncias_cnv", "fraudes_y_proteccion",
        "¿Cuál de estas afirmaciones sobre las denuncias ante la CNV es correcta según el material?",
        3,
        "Antes de denunciar se sugiere plantear el reclamo al responsable de relaciones con el público (RRP) del agente.",
        [
            ("generalizacion_excesiva", "La denuncia debe hacerse siempre con datos personales: no se aceptan denuncias anónimas."),
            ("riesgo_nulo", "La CNV resuelve los conflictos patrimoniales y hace que te devuelvan el dinero."),
            ("confusion_conceptos", "Las denuncias contra agentes del mercado de capitales se presentan ante el Banco Central."),
        ],
        "Antes de formular una denuncia se sugiere contactar al responsable de relaciones con el público (RRP) del agente. La denuncia es reservada y puede ser anónima, y la CNV no resuelve conflictos patrimoniales ni cuestiones resarcitorias.",
        ["CNV_p32_c0", "CNV_p32_c1"],
        ["responsable de relaciones con el público (RRP) del agente en cuestión", "es posible denunciar en carácter de anónimo", "excede la competencia de la CNV la resolución de conflictos de índole patrimonial"],
        "canales de denuncia ante la CNV, responsable de relaciones con el público, qué puede y qué no puede resolver la CNV",
    ),
]

NIVELES_ORDEN = ["principiante", "intermedio", "avanzado"]


def _por_tema(tema: str) -> list[dict]:
    return [p for p in PREGUNTAS if p["tema"] == tema]


def _por_id(pregunta_id: str) -> dict | None:
    return next((p for p in PREGUNTAS if p["id"] == pregunta_id), None)


@lru_cache(maxsize=512)
def _embed(texto: str) -> tuple:
    return tuple(ollama.embed(model=MODELO_EMBEDDINGS, input=texto)["embeddings"][0])


def _similitud_coseno(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def elegir_pregunta(tema: str, mensaje: str) -> str | None:
    """
    De las preguntas fijas del banco para ese tema, devuelve el id de la más relacionada
    semánticamente con el mensaje del usuario (similitud coseno contra 'contexto_semantico',
    con los mismos embeddings que el RAG). Es determinista: mismo mensaje, misma pregunta.
    """
    candidatas = _por_tema(tema)
    if not candidatas:
        return None
    if len(candidatas) == 1:
        return candidatas[0]["id"]

    embedding_mensaje = _embed(mensaje)
    mejor_id, mejor_similitud = candidatas[0]["id"], -1.0
    for candidata in candidatas:
        similitud = _similitud_coseno(embedding_mensaje, _embed(candidata["contexto_semantico"]))
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_id = candidata["id"]
    return mejor_id


def armar_pregunta(pregunta_id: str) -> dict | None:
    """Enunciado y opciones para mostrar. No incluye la clave ni la categoría de error."""
    pregunta = _por_id(pregunta_id)
    if pregunta is None:
        return None
    return {
        "tema": pregunta["tema"],
        "pregunta": pregunta["pregunta"],
        "opciones": [{"id": o["id"], "texto": o["texto"]} for o in pregunta["opciones"]],
    }


def calificar_respuesta(pregunta_id: str, opcion_id: str) -> dict | None:
    """
    Corrección determinista: compara la opción elegida con la clave del banco. No usa el
    modelo de lenguaje. Si eligió un distractor, devuelve también su categoría de error
    conceptual. Devuelve None si la pregunta o la opción no existen.
    """
    pregunta = _por_id(pregunta_id)
    if pregunta is None:
        return None
    elegida = next((o for o in pregunta["opciones"] if o["id"] == opcion_id), None)
    if elegida is None:
        return None
    correcta = next(o for o in pregunta["opciones"] if o["categoria_error"] is None)
    return {
        "correcta": elegida["categoria_error"] is None,
        "opcion_elegida": elegida["id"],
        "texto_elegido": elegida["texto"],
        "categoria_error": elegida["categoria_error"],
        "opcion_correcta": correcta["id"],
        "texto_correcto": correcta["texto"],
        "explicacion": pregunta["explicacion"],
    }


def siguiente_nivel(nivel_actual: str, acierto: bool | None) -> str:
    if acierto is None:
        return nivel_actual  # no se pudo corregir (opción inválida): el nivel no cambia
    idx = NIVELES_ORDEN.index(nivel_actual)
    idx = min(idx + 1, len(NIVELES_ORDEN) - 1) if acierto else max(idx - 1, 0)
    return NIVELES_ORDEN[idx]