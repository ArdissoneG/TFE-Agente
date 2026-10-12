import json
from datetime import datetime
from pathlib import Path

RUTA_LOG_POR_DEFECTO = Path(__file__).resolve().parent.parent / "registros" / "interacciones.jsonl"


def registrar(evento: dict, ruta: str | Path | None = None) -> None:
    """Agrega una línea JSON al log de interacciones (una línea = un evento)."""
    destino = Path(ruta) if ruta else RUTA_LOG_POR_DEFECTO
    destino.parent.mkdir(parents=True, exist_ok=True)
    registro = {"timestamp": datetime.now().isoformat(timespec="seconds"), **evento}
    with open(destino, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")