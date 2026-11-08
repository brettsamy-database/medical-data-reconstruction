"""Accès au modèle MedGemma et utilitaires de parsing JSON des sorties LLM.

Repris des cellules 0.1) (authentification Hugging Face) et 4C.1) (chargement
de MedGemma) du notebook original. Les fonctions de nettoyage JSON, qui
existaient en trois copies dans le notebook (filtre gold, reconstruction,
scoring), sont regroupées ici.

`torch` et `transformers` ne sont importés qu'au chargement du modèle :
on peut donc importer ce module (et lancer les tests) sans GPU.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

from config import settings

# Réduit les avertissements transformers sur max_length quand max_new_tokens est déjà contrôlé.
warnings.filterwarnings("ignore", message=r"Both `max_new_tokens`.*`max_length`.*")


# =========================
# Authentification Hugging Face
# =========================


def get_hf_token() -> Optional[str]:
    """Cherche le token dans : variables d'environnement, fichier .env, secrets Colab."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(settings.ROOT_DIR / ".env")
    except Exception:  # noqa: BLE001
        pass

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:  # Google Colab
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


def hf_login() -> bool:
    """Connexion Hugging Face (nécessaire car MedGemma est un modèle « gated »)."""
    token = get_hf_token()
    if not token:
        print("No HF_TOKEN found. Put HF_TOKEN=... in a .env file at the repo root "
              "(or add a Colab secret named HF_TOKEN) if the model is gated.")
        return False
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    print("Hugging Face: authenticated.")
    return True


# =========================
# Modèle MedGemma
# =========================


@dataclass
class LLMResult:
    note: str
    used_llm: bool
    model_name: str


class MedGemma:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipe = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        token = os.environ.get("HF_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                device_map="auto",
            )
        except Exception:  # noqa: BLE001
            # repli 4-bit
            bnb = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                quantization_config=bnb,
                device_map="auto",
            )

        try:
            model.generation_config.max_length = None
        except Exception:  # noqa: BLE001
            pass

        self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, do_sample: bool) -> str:
        if self._pipe is None:
            self.load()
        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False,
        )
        return out[0]["generated_text"]


_LLM: Optional[MedGemma] = None


def get_llm() -> MedGemma:
    """Instance partagée (le modèle n'est chargé qu'au premier `generate`)."""
    global _LLM
    if _LLM is None or _LLM.model_name != settings.MODEL_NAME:
        _LLM = MedGemma(settings.MODEL_NAME)
    return _LLM


def set_llm(llm) -> None:
    """Remplace l'instance partagée (utile pour les tests avec un faux modèle)."""
    global _LLM
    _LLM = llm


# =========================
# Parsing JSON des sorties LLM
# =========================


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json(text: str) -> Optional[str]:
    """Renvoie le premier objet JSON `{...}` équilibré trouvé dans le texte."""
    text = strip_fences(text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def try_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def repair_json(s: str) -> str:
    """Réparations simples : virgules finales, guillemets simples."""
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    return s


def load_json_lenient(text: str) -> Optional[dict]:
    """extract_first_json -> json.loads -> repair_json -> json.loads, sans appel LLM."""
    js = extract_first_json(text)
    if not js:
        return None
    obj = try_load(js)
    if obj is not None:
        return obj
    return try_load(repair_json(js))
