"""Fonctions partagées par toutes les étapes : normalisation de texte, JSON sur disque, cache."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, List, Optional


def norm_text(s: Optional[str]) -> str:
    """Forme canonique d'un libellé, utilisée pour toutes les comparaisons exactes."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def first(lst: List[Any], default=None):
    return lst[0] if lst else default


def ensure_list(x) -> list:
    """Les sorties du LLM ne respectent pas toujours le schéma : tout ce qui n'est pas une liste devient []."""
    return x if isinstance(x, list) else []


def shorten(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("..." if len(s) > n else "")


# =========================
# JSON sur disque
# =========================


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> Optional[dict]:
    """Comme read_json, mais renvoie None si le fichier est absent ou corrompu."""
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_json(dir_path: Path) -> List[Path]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.glob("*.json") if p.is_file())


def to_json_str(x) -> str:
    return json.dumps(x, ensure_ascii=False)


# =========================
# Cache des appels LLM
# =========================
# Chaque résultat LLM est écrit avec la signature de ses entrées. Si les entrées
# changent (nouvelle note, nouvelle reconstruction), la signature ne correspond
# plus et le résultat est recalculé ; sinon on relit le disque sans appeler le modèle.


def signature(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_cached(path: Path, sig: str) -> Optional[dict]:
    obj = safe_read_json(path)
    if isinstance(obj, dict) and obj.get("signature") == sig:
        return obj
    return None
