"""Petits utilitaires partagés (texte, JSON sur disque)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional


def norm_text(s: Optional[str]) -> str:
    """Normalisation utilisée partout pour les comparaisons exactes."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def first(lst: List[Any], default=None):
    return lst[0] if lst else default


def shorten(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("..." if len(s) > n else "")


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_json(dir_path: Path) -> List[Path]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.json") if p.is_file()])


def to_json_str(x) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return ""
