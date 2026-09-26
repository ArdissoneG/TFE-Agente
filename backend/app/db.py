import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "perfiles.db"


def _conectar():
    return sqlite3.connect(DB_PATH)


def inicializar_db():
    with _conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS perfiles (
                usuario_id TEXT PRIMARY KEY,
                niveles_json TEXT NOT NULL,
                actualizado_en TEXT NOT NULL
            )
        """)


def guardar_perfil(usuario_id: str, niveles: dict):
    with _conectar() as conn:
        conn.execute(
            """
            INSERT INTO perfiles (usuario_id, niveles_json, actualizado_en)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(usuario_id) DO UPDATE SET
                niveles_json = excluded.niveles_json,
                actualizado_en = excluded.actualizado_en
            """,
            (usuario_id, json.dumps(niveles)),
        )


def obtener_perfil(usuario_id: str):
    with _conectar() as conn:
        fila = conn.execute(
            "SELECT niveles_json FROM perfiles WHERE usuario_id = ?", (usuario_id,)
        ).fetchone()
    return json.loads(fila[0]) if fila else None


def borrar_perfil(usuario_id: str):
    with _conectar() as conn:
        conn.execute("DELETE FROM perfiles WHERE usuario_id = ?", (usuario_id,))