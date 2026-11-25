"""Accès à MedGemma et lecture des réponses JSON du modèle.

Toutes les étapes qui appellent le LLM passent par `get_llm()`, qui renvoie une
instance partagée chargée au premier appel. Les tests la remplacent par un faux
modèle avec `set_llm()`, ce qui permet de tester tout le pipeline sans GPU :
`torch` et `transformers` ne sont importés qu'au moment du chargement.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

from config import settings

# max_new_tokens est fixé à chaque appel : l'avertissement de transformers sur max_length n'apporte rien.
warnings.filterwarnings("ignore", message=r"Both `max_new_tokens`.*`max_length`.*")


# =========================
# Authentification Hugging Face
# =========================


def get_hf_token() -> Optional[str]:
    """Cherche le token dans l'environnement, puis dans le fichier .env, puis dans les secrets Colab."""
    try:
        from dotenv import load_dotenv

        load_dotenv(settings.ROOT_DIR / ".env")
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:  # hors Colab, ou secret absent : pas de token
        return None


def hf_login() -> bool:
    """MedGemma est un modèle « gated » : sans token, le téléchargement est refusé."""
    token = get_hf_token()
    if not token:
        print("No HF_TOKEN found. Put HF_TOKEN=... in a .env file at the repo root "
              "(or add a Colab secret named HF_TOKEN).")
        return False
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    print("Hugging Face: authenticated.")
    return True


# =========================
# Modèle
# =========================


@dataclass
class LLMResult:
    """Une note générée, et si elle vient du modèle ou du gabarit de secours."""

    note: str
    used_llm: bool
    model_name: str


class MedGemma:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipe = None
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is None:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        token = os.environ.get("HF_TOKEN") or None
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)

        # bf16 tient sur un GPU de 12 Go ou plus (environ 8,6 Go pour le modèle 4B).
        # En dessous, on retombe sur une quantification 4 bits.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                device_map="auto",
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto",
            )

        model.generation_config.max_length = None
        self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stop_at_json: Optional[bool] = None,
    ) -> str:
        """Génère une réponse au prompt.

        Les appels déterministes (`do_sample=False`) attendent tous un objet JSON, et
        on n'en lit que le premier. Sans arrêt explicite, le modèle continue jusqu'à
        `max_new_tokens`, souvent en répétant sa réponse. `stop_at_json` (actif par
        défaut pour ces appels) coupe la génération dès que le premier objet est
        fermé. En décodage glouton, le JSON obtenu est identique, et l'appel est
        jusqu'à 9 fois plus rapide.
        """
        if self._pipe is None:
            self.load()
        if stop_at_json is None:
            stop_at_json = not do_sample

        extra = {}
        if stop_at_json:
            from transformers import StoppingCriteriaList

            extra["stopping_criteria"] = StoppingCriteriaList([_StopAfterFirstJson(self._pipe.tokenizer)])

        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False,
            **extra,
        )
        return out[0]["generated_text"]


class _StopAfterFirstJson:
    """Critère d'arrêt : vrai dès que le texte généré contient un objet JSON complet.

    Il s'appuie sur `extract_first_json`, la fonction qu'utilisent les parseurs :
    la génération s'arrête exactement là où la lecture du JSON s'arrêterait.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompt_len: Optional[int] = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if self.prompt_len is None:  # premier appel : un seul token généré après le prompt
            self.prompt_len = input_ids.shape[1] - 1
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        done = extract_first_json(text) is not None
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


_LLM: Optional[MedGemma] = None


def get_llm() -> MedGemma:
    global _LLM
    if _LLM is None or _LLM.model_name != settings.MODEL_NAME:
        _LLM = MedGemma(settings.MODEL_NAME)
    return _LLM


def set_llm(llm) -> None:
    """Remplace le modèle partagé, par exemple par un faux modèle dans les tests."""
    global _LLM
    _LLM = llm


# =========================
# Lecture des réponses JSON
# =========================
# MedGemma entoure souvent son JSON de balises markdown, le répète, ou laisse
# une virgule finale. On extrait le premier objet équilibré et on corrige les
# défauts les plus courants. Les étapes qui le justifient (reconstruction,
# vérification) peuvent ensuite demander au modèle de réparer sa réponse.


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json(text: str) -> Optional[str]:
    """Renvoie le premier objet `{...}` aux accolades équilibrées, ou None."""
    text = strip_fences(text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _try_load(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return None


def repair_json(s: str) -> str:
    """Corrige les deux défauts les plus fréquents : virgules finales et guillemets simples."""
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    return s


def load_json_lenient(text: str):
    """Lit le premier objet JSON de la réponse, avec une passe de réparation si besoin."""
    js = extract_first_json(text)
    if not js:
        return None
    obj = _try_load(js)
    if obj is not None:
        return obj
    return _try_load(repair_json(js))
