"""Étape 5 : génération des notes cliniques non structurées à partir du `gold_source`.

Repris de la cellule 5) du notebook original.

Nomenclature :
- `raw_note_text` : sortie brute du LLM de génération
- `note_text`     : note nettoyée utilisée pour la reconstruction
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from src.gold import save_gold
from src.llm import LLMResult, get_llm
from src.utils import norm_text, shorten

NOTE_STYLE_CATALOG = {
    "consult_note_short": {
        "label": "Short consultation note",
        "style_instructions": [
            "Write a short consultation note.",
            "Use natural clinical prose.",
            "Keep the text compact.",
        ],
    },
    "medical_history_note": {
        "label": "Medical history note",
        "style_instructions": [
            "Write the note as a brief medical history summary.",
            "Focus on known conditions first, then medications.",
            "Use a slightly retrospective tone.",
        ],
    },
    "telegraphic_note": {
        "label": "Telegraphic note",
        "style_instructions": [
            "Write like a quick doctor note.",
            "Use short fragments and abbreviated style.",
            "The text can feel compressed and not fully polished.",
        ],
    },
    "health_check_summary": {
        "label": "Health check summary",
        "style_instructions": [
            "Write the note as a brief health check conclusion.",
            "Use a summarizing tone.",
            "Keep it concise and readable.",
        ],
    },
}

NOTE_STYLE_NAMES = list(NOTE_STYLE_CATALOG.keys())


def assign_note_styles(patients_in: List[dict], note_styles: Optional[List[str]] = None) -> List[dict]:
    """Attribue un style de note à chaque patient, en tournant sur la liste des styles."""
    note_styles = note_styles or NOTE_STYLE_NAMES
    out = []
    for i, patient in enumerate(patients_in):
        style_name = note_styles[i % len(note_styles)]
        style_cfg = NOTE_STYLE_CATALOG[style_name]
        p2 = dict(patient)
        p2["note_style"] = style_name
        p2["note_style_label"] = style_cfg["label"]
        out.append(p2)
    return out


def note_prompt(patient: dict) -> str:
    note_style = patient.get("note_style", NOTE_STYLE_NAMES[0])
    style_cfg = NOTE_STYLE_CATALOG[note_style]
    style_block = "\n".join([f"- {x}" for x in style_cfg["style_instructions"]])

    name = patient.get("full_name") or ""
    dob = patient.get("birthDate") or ""
    gender = patient.get("gender") or ""

    cond_lines = "\n".join([f"- {c}" for c in patient.get("conditions", [])]) or "- None"
    med_lines = "\n".join([f"- {m}" for m in patient.get("medication_requests", [])]) or "- None"

    return f"""You are a doctor writing a free-text clinical note.

Style requirements:
{style_block}

Hard constraints:
- The note must explicitly mention the patient name in a natural way.
- The note must explicitly mention the date of birth in a natural way.
- The note must explicitly mention the gender in a natural way.
- Use all listed conditions and all listed medication requests somewhere in the note.
- Do not omit any listed item.
- Do not add any new diagnosis, symptom, medication, treatment, or contextual detail.
- Keep the note non-structured and natural.

Patient:
- Name: {name}
- Date of birth: {dob}
- Gender: {gender}

Conditions:
{cond_lines}

Medication requests:
{med_lines}

Clinical note:"""


def template_note(patient: dict) -> str:
    """Note de repli quand le LLM n'est pas disponible ou échoue."""
    note_style = patient.get("note_style", NOTE_STYLE_NAMES[0])
    name = patient.get("full_name") or ""
    dob = patient.get("birthDate") or ""
    gender = patient.get("gender") or ""

    cond_txt = "; ".join(patient.get("conditions", [])) or "None"
    med_txt = "; ".join(patient.get("medication_requests", [])) or "None"
    prefix = {
        "consult_note_short": "Consultation note",
        "medical_history_note": "Medical history",
        "telegraphic_note": "Quick note",
        "health_check_summary": "Health check summary",
    }.get(note_style, "Clinical note")
    return (
        f"{prefix}: {name} (DOB: {dob}, gender: {gender}) with {cond_txt}. "
        f"Medication requests include {med_txt}."
    )


# =========================
# Nettoyage des sorties brutes du LLM
# =========================


def _literal_string_or_none(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, str):
            return val
    except Exception:  # noqa: BLE001
        pass
    return None


def _dedupe_sentences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out, seen = [], set()
    for part in parts:
        p = re.sub(r"\s+", " ", (part or "").strip())
        if not p:
            continue
        k = norm_text(p)
        if k and k not in seen:
            out.append(p)
            seen.add(k)
    return " ".join(out).strip() if out else re.sub(r"\s+", " ", text).strip()


def _normalize_candidate(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""

    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s, flags=re.S)
    s = s.replace("\\n", " ").replace("\\t", " ")
    s = re.sub(r"^Clinical note:\s*", "", s, flags=re.I)
    s = re.sub(r"^Final Answer:\s*", "", s, flags=re.I)
    s = re.sub(r"^The final answer is\s*", "", s, flags=re.I)
    s = s.strip().strip('"').strip("'").strip()

    cut_patterns = [
        r"\n\s*Final Answer\s*:",
        r"\n\s*```",
        r"\n\s*print\s*\(",
        r"\n\s*Patient\s*:",
        r"\n\s*Conditions\s*:",
        r"\n\s*Medication requests\s*:",
        r"\n\s*Diagnoses\s*\(Condition\)\s*:",
        r"\n\s*Medications\s*\(MedicationRequest\)\s*:",
    ]
    cut_idx = len(s)
    for pat in cut_patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            cut_idx = min(cut_idx, m.start())
    s = s[:cut_idx].strip()

    s = re.sub(r"\s+", " ", s).strip()
    s = _dedupe_sentences(s)

    lower = s.lower()
    if any(tok in lower for tok in ["```", "print(", "final answer", "\\boxed{"]):
        return ""
    return s


def _candidate_score(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return -1e9
    score = float(len(s))
    lower = s.lower()
    bad_tokens = [
        "final answer", "```", "print(", "\\boxed{",
        "patient:", "conditions:", "medication requests:",
        "diagnoses (condition):", "medications (medicationrequest):",
    ]
    score -= 200.0 * sum(tok in lower for tok in bad_tokens)
    score += 25.0 * bool(re.search(r"[A-Za-z]", s))
    score += 25.0 * (len(s.split()) >= 8)
    score += 25.0 * bool(re.search(r"[.!?]", s))
    return score


def clean_generated_note(raw_text: str) -> str:
    """Extrait la meilleure note « propre » d'une sortie brute (code, \\boxed{}, Final Answer...)."""
    raw = (raw_text or "").strip()
    if not raw:
        return ""

    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    candidates = [raw]

    if re.search(r"Clinical note\s*:", raw, flags=re.I):
        candidates.append(re.split(r"Clinical note\s*:", raw, flags=re.I)[-1])

    for pat in [r"\n\s*Final Answer\s*:", r"\n\s*```", r"\n\s*print\s*\("]:
        m = re.search(pat, raw, flags=re.I)
        if m:
            candidates.append(raw[:m.start()])

    for m in re.finditer(r"\\boxed\{([^}]*)\}", raw, flags=re.S):
        candidates.append(m.group(1))

    for pat in [r'print\((?P<q>"(?:[^"\\]|\\.)*")\)', r"print\((?P<q>'(?:[^'\\]|\\.)*')\)"]:
        for m in re.finditer(pat, raw, flags=re.S):
            lit = _literal_string_or_none(m.group("q"))
            if lit:
                candidates.append(lit)

    for pat in [r'"""(.*?)"""', r"'{3}(.*?)'{3}"]:
        for m in re.finditer(pat, raw, flags=re.S):
            block = (m.group(1) or "").strip()
            if block:
                candidates.append(block)

    cleaned = []
    seen = set()
    for cand in candidates:
        c = _normalize_candidate(cand)
        k = norm_text(c)
        if c and k and k not in seen:
            cleaned.append(c)
            seen.add(k)

    if not cleaned:
        return re.sub(r"\s+", " ", raw).strip()

    return max(cleaned, key=_candidate_score)


# =========================
# Pipeline de génération
# =========================


def generate_notes(patients: List[dict], llm=None) -> Tuple[Dict[str, str], Dict[str, LLMResult]]:
    """Génère une note par patient (repli sur `template_note` si le LLM échoue) et les écrit sur disque."""
    llm = llm or get_llm()
    settings.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    settings.RAW_NOTES_DIR.mkdir(parents=True, exist_ok=True)

    raw_notes: Dict[str, str] = {}
    notes: Dict[str, LLMResult] = {}
    for p in patients:
        pid = p["patient_id"]
        if llm.available():
            try:
                txt = llm.generate(
                    note_prompt(p),
                    max_new_tokens=settings.NOTE_GENERATION_MAX_NEW_TOKENS,
                    temperature=settings.NOTE_GENERATION_TEMPERATURE,
                    do_sample=settings.NOTE_GENERATION_DO_SAMPLE,
                )
                raw_txt = (txt or "").strip()
                clean_txt = clean_generated_note(raw_txt) or raw_txt
                raw_notes[pid] = raw_txt
                notes[pid] = LLMResult(note=clean_txt, used_llm=True, model_name=settings.MODEL_NAME)
            except Exception:  # noqa: BLE001
                fallback = template_note(p)
                raw_notes[pid] = fallback
                notes[pid] = LLMResult(note=fallback, used_llm=False, model_name="template")
        else:
            fallback = template_note(p)
            raw_notes[pid] = fallback
            notes[pid] = LLMResult(note=fallback, used_llm=False, model_name="template")

    for pid, res in notes.items():
        (settings.RAW_NOTES_DIR / f"{pid}.txt").write_text(raw_notes.get(pid, res.note), encoding="utf-8")
        (settings.NOTES_DIR / f"{pid}.txt").write_text(res.note, encoding="utf-8")

    return raw_notes, notes


def assign_styles_and_save(patients: List[dict]) -> List[dict]:
    """Attribue les styles et met à jour le gold sur disque (le style fait partie de la source canonique)."""
    patients = assign_note_styles(patients, NOTE_STYLE_NAMES)
    save_gold(patients)
    return patients


def load_notes() -> Tuple[Dict[str, str], Dict[str, LLMResult]]:
    """Relit les notes depuis data/processed/notes/."""
    raw_notes: Dict[str, str] = {}
    notes: Dict[str, LLMResult] = {}
    if not settings.NOTES_DIR.exists():
        return raw_notes, notes
    for p in sorted(settings.NOTES_DIR.glob("*.txt")):
        pid = p.stem
        note = p.read_text(encoding="utf-8")
        raw_path = settings.RAW_NOTES_DIR / f"{pid}.txt"
        raw_notes[pid] = raw_path.read_text(encoding="utf-8") if raw_path.exists() else note
        notes[pid] = LLMResult(note=note, used_llm=raw_notes[pid] != note, model_name=settings.MODEL_NAME)
    return raw_notes, notes


def notes_dataframe(patients: List[dict], raw_notes: Dict[str, str], notes: Dict[str, LLMResult]) -> pd.DataFrame:
    gold_by_pid = {p["patient_id"]: p for p in patients}
    df = pd.DataFrame([{
        "patient_id": pid,
        "full_name": gold_by_pid.get(pid, {}).get("full_name"),
        "birthDate": gold_by_pid.get(pid, {}).get("birthDate"),
        "gender": gold_by_pid.get(pid, {}).get("gender"),
        "note_style": gold_by_pid.get(pid, {}).get("note_style"),
        "note_style_label": gold_by_pid.get(pid, {}).get("note_style_label"),
        "used_llm": res.used_llm,
        "model": res.model_name,
        "raw_note_preview": shorten(raw_notes.get(pid, "")),
        "note_text": res.note,
        "note_preview": shorten(res.note),
        "raw_chars": len(raw_notes.get(pid, "")),
        "clean_chars": len(res.note or ""),
    } for pid, res in notes.items()])
    if df.empty:
        return df
    return df.sort_values(["note_style", "patient_id"]).reset_index(drop=True)
