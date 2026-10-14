"""
Valida las personas y las consultas de la evaluación contra el corpus congelado:
  personas:  8 respuestas válidas, 4 temas con nivel real, 2 por nivel nominal (una calibrada y una
             descalibrada); las calibradas coinciden con el diagnóstico y las descalibradas no.
  consultas: 20 de recuperación (5 por tema) y 10 fuera de corpus; los fragmentos relevantes se
             recalculan desde las frases ancla y deben coincidir con los guardados; los términos
             que deben estar ausentes del corpus realmente lo están; cada persona tiene 5 escenarios.

Uso (con el venv activo, desde la raíz del repo):  python evaluation/validar_instrumentos.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from consultas_util import cargar_consultas  # noqa: E402
from personas import NIVELES, cargar_personas, perfil_diagnosticado  # noqa: E402
from utils import cargar_corpus, normalizar  # noqa: E402

TEMAS = ["conceptos_basicos", "instrumentos_inversion", "sesgos_y_riesgos", "fraudes_y_proteccion"]
CLAVES_DIAGNOSTICO = [
    "conceptos_basicos_ahorro_inversion", "conceptos_basicos_riesgo_rentabilidad",
    "instrumentos_conoce_acciones_bonos", "instrumentos_conoce_fci_cedears",
    "sesgos_conoce_sesgos_cognitivos", "sesgos_distingue_regulado_no_regulado",
    "fraudes_reconoce_senales_estafa", "fraudes_conoce_canales_denuncia",
]


def main() -> int:
    errores = []
    corpus = cargar_corpus()
    texto_corpus = " ".join(normalizar(f["texto"]).lower() for f in corpus.values())

    # ---------------------------------------------------------------- personas
    personas = cargar_personas()
    ids_personas = [p["id"] for p in personas]
    if len(ids_personas) != 6 or len(set(ids_personas)) != 6:
        errores.append("tiene que haber 6 personas con ids únicos")
    for p in personas:
        if sorted(p["respuestas_diagnostico"]) != sorted(CLAVES_DIAGNOSTICO):
            errores.append(f"[{p['id']}] las respuestas al diagnóstico no son las 8 esperadas")
        if any(v not in ("no", "algo", "si") for v in p["respuestas_diagnostico"].values()):
            errores.append(f"[{p['id']}] hay respuestas al diagnóstico inválidas")
        if sorted(p["nivel_real"]) != sorted(TEMAS) or any(v not in NIVELES for v in p["nivel_real"].values()):
            errores.append(f"[{p['id']}] el nivel real no cubre los 4 temas con niveles válidos")
            continue
        diagnosticado = perfil_diagnosticado(p)
        coinciden = sum(diagnosticado[t] == p["nivel_real"][t] for t in TEMAS)
        if p["autopercepcion"] == "calibrada" and coinciden != 4:
            errores.append(f"[{p['id']}] es calibrada pero el diagnóstico solo coincide en {coinciden} de 4 temas")
        if p["autopercepcion"].startswith("descalibrada") and coinciden > 2:
            errores.append(f"[{p['id']}] es descalibrada pero el diagnóstico coincide en {coinciden} de 4 temas")
    por_nivel = Counter((p["nivel_nominal"], p["autopercepcion"].split(" ")[0]) for p in personas)
    for nivel in NIVELES:
        for tipo in ("calibrada", "descalibrada"):
            if por_nivel[(nivel, tipo)] != 1:
                errores.append(f"nivel {nivel}: se esperaba 1 persona {tipo}, hay {por_nivel[(nivel, tipo)]}")

    # ---------------------------------------------------------------- consultas
    consultas = cargar_consultas()
    ids = [c["id"] for c in consultas]
    if len(ids) != len(set(ids)):
        errores.append("hay ids de consulta repetidos")
    recuperacion = [c for c in consultas if c["tipo"] == "recuperacion"]
    fuera = [c for c in consultas if c["tipo"] == "fuera_de_corpus"]
    if len(recuperacion) != 20 or len(fuera) != 10:
        errores.append(f"se esperaban 20 de recuperación y 10 fuera de corpus (hay {len(recuperacion)} y {len(fuera)})")
    for tema in TEMAS:
        n = sum(c["tema_esperado"] == tema for c in recuperacion)
        if n != 5:
            errores.append(f"el tema {tema} tiene {n} preguntas de recuperación (se esperaban 5)")

    for c in recuperacion:
        derivados = sorted(
            k for k, v in corpus.items() if any(normalizar(a) in normalizar(v["texto"]) for a in c["anclas"])
        )
        if derivados != sorted(c["relevantes"]):
            errores.append(f"[{c['id']}] los relevantes guardados {c['relevantes']} no coinciden con los derivados {derivados}")
        for a in c["anclas"]:
            if not any(normalizar(a) in normalizar(v["texto"]) for v in corpus.values()):
                errores.append(f"[{c['id']}] la ancla no aparece en ningún fragmento: «{a}»")
        temas = {corpus[k]["tema"] for k in c["relevantes"] if k in corpus}
        if temas != {c["tema_esperado"]}:
            errores.append(f"[{c['id']}] los relevantes son de los temas {temas}, no solo de {c['tema_esperado']}")
        if not (1 <= len(c["relevantes"]) <= 4):
            errores.append(f"[{c['id']}] tiene {len(c['relevantes'])} fragmentos relevantes")
    for c in fuera:
        for t in c["terminos_ausentes"]:
            if normalizar(t).lower() in texto_corpus:
                errores.append(f"[{c['id']}] el término «{t}» SÍ aparece en el corpus: la pregunta no es fuera de corpus")
    for c in consultas:
        if c["persona_id"] not in ids_personas:
            errores.append(f"[{c['id']}] persona_id desconocida: {c['persona_id']}")
    por_persona = Counter(c["persona_id"] for c in consultas)
    if any(por_persona[i] != 5 for i in ids_personas):
        errores.append(f"los escenarios no están repartidos 5 por persona: {dict(por_persona)}")

    print(f"{len(personas)} personas, {len(recuperacion)} preguntas de recuperación, {len(fuera)} fuera de corpus.")
    print("Fragmentos relevantes por pregunta:", {c["id"]: len(c["relevantes"]) for c in recuperacion})
    print("Escenarios por persona:", dict(sorted(por_persona.items())))
    if errores:
        print("\nPROBLEMAS:")
        for e in errores:
            print(" -", e)
        return 1
    print("Instrumentos válidos: personas, consultas y fragmentos relevantes verificados contra el corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())