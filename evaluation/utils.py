"""Utilidades compartidas por los scripts de evaluación (sin dependencias del backend)."""
import json
import re
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
RUTA_CORPUS = RAIZ_EVALUACION / "corpus_chunks.jsonl"

_PATRON_ID = re.compile(
    r"(?P<doc>Educ\. Financiera_agosto_2023|texto_guia_de_inversoras_v2_0_2)\.pdf"
    r"_pagina(?P<pagina>\d+)_(?P<transcripcion>transcripcion_)?chunk(?P<n>\d+)$"
)
_PREFIJO_DOC = {
    "Educ. Financiera_agosto_2023": "CUA",   # Cuaderno BCRA - Ministerio de Educación de Santa Fe
    "texto_guia_de_inversoras_v2_0_2": "CNV",  # Guía de inversoras e inversores de la CNV
}


def id_corto(id_largo: str) -> str:
    """
    '../data/raw/Educ. Financiera_agosto_2023.pdf_pagina25_chunk1' -> 'CUA_p25_c1'
    '.../texto_guia_de_inversoras_v2_0_2.pdf_pagina28_transcripcion_chunk0' -> 'CNV_p28_t0'
    Si el id ya es corto, lo devuelve igual.
    """
    m = _PATRON_ID.search(id_largo)
    if not m:
        return id_largo
    marca = "t" if m["transcripcion"] else "c"
    return f"{_PREFIJO_DOC[m['doc']]}_p{m['pagina']}_{marca}{m['n']}"


def cargar_corpus(ruta: Path = RUTA_CORPUS) -> dict[str, dict]:
    """Devuelve {id_corto: fila} con el corpus congelado de evaluation/corpus_chunks.jsonl."""
    corpus = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            fila = json.loads(linea)
            corpus[id_corto(fila["id"])] = fila
    return corpus


def normalizar(texto: str) -> str:
    return " ".join(texto.split())