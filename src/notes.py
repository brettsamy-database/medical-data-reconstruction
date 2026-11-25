"""Génération des notes cliniques à partir du gold source.

Chaque patient reçoit l'un de quatre styles de note, attribués à tour de rôle pour
que les styles soient équilibrés dans l'échantillon. L'idée est de voir si la forme
de la note (prose, liste télégraphique…) change ce que la reconstruction retrouve.

On garde deux versions de chaque note :
- `raw_note_text` : la sortie brute du modèle ;
- `note_text`     : la note nettoyée, la seule que voit la reconstruction.
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
        "fallback_prefix": "Consultation note",
        "style_instructions": [
            "Write a short consultation note.",
            "Use natural clinical prose.",
            "Keep the text compact.",
        ],
    },
    "medical_history_note": {
        "label": "Medical history note",
        "fallback_prefix": "Medical history",
        "style_instructions": [
            "Write the note as a brief medical history summary.",
            "Focus on known conditions first, then medications.",
            "Use a slightly retrospective tone.",
        ],
    },
    "telegraphic_note": {
        "label": "Telegraphic note",
        "fallback_prefix": "Quick note",
        "style_instructions": [
            "Write like a quick doctor note.",
            "Use short fragments and abbreviated style.",
            "The text can feel compressed and not fully polished.",
        ],
    },
    "health_check_summary": {
        "label": "Health check summary",
        "fallback_prefix": "Health check summary",
        "style_instructions": [
            "Write the note as a brief health check conclusion.",
            "Use a summarizing tone.",
            "Keep it concise and readable.",
        ],
    },
}

NOTE_STYLE_NAMES = list(NOTE_STYLE_CATALOG)


def assign_note_styles(patients_in: List[dict], note_styles: Optional[List[str]] = None) -> List[dict]:
    note_styles = note_styles or NOTE_STYLE_NAMES
    out = []
    for i, patient in enumerate(patients_in):
        style = note_styles[i % len(note_styles)]
        out.append({**patient, "note_style": style, "note_style_label": NOTE_STYLE_CATALOG[style]["label"]})
    return out


def assign_styles_and_save(patients: List[dict]) -> List[dict]:
    """Le style fait partie du gold : on le réécrit sur disque pour que le scoring puisse grouper par style."""
    patients = assign_note_styles(patients, NOTE_STYLE_NAMES)
    save_gold(patients)
    return patients


def note_prompt(patient: dict) -> str:
    style = NOTE_STYLE_CATALOG[patient.get("note_style", NOTE_STYLE_NAMES[0])]
    style_block = "\n".join(f"- {x}" for x in style["style_instructions"])
    cond_lines = "\n".join(f"- {c}" for c in patient.get("conditions", [])) or "- None"
    med_lines = "\n".join(f"- {m}" for m in patient.get("medication_requests", [])) or "- None"

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
- Name: {patient.get("full_name") or ""}
- Date of birth: {patient.get("birthDate") or ""}
- Gender: {patient.get("gender") or ""}

Conditions:
{cond_lines}

Medication requests:
{med_lines}

Clinical note:"""


def template_note(patient: dict) -> str:
    """Note minimale construite sans LLM, pour faire tourner le pipeline sans GPU."""
    style = NOTE_STYLE_CATALOG.get(patient.get("note_style", NOTE_STYLE_NAMES[0]), {})
    prefix = style.get("fallback_prefix", "Clinical note")
    cond_txt = "; ".join(patient.get("conditions", [])) or "None"
    med_txt = "; ".join(patient.get("medication_requests", [])) or "None"
    return (
        f"{prefix}: {patient.get('full_name') or ''} (DOB: {patient.get('birthDate') or ''}, "
        f"gender: {patient.get('gender') or ''}) with {cond_txt}. "
        f"Medication requests include {med_txt}."
    )


# =========================
# Nettoyage des sorties du modèle
# =========================
# Sans format de chat, MedGemma ne s'arrête pas toujours à la fin de la note. On a
# observé : un « Final Answer: \boxed{...} » ajouté à la fin, la note réécrite dans
# un bloc de code Python (print("...")), le prompt recopié (« Patient: »,
# « Conditions: »), ou la même phrase répétée plusieurs fois.
# On construit plusieurs versions candidates de la note et on garde la plus plausible.

# Tout ce qui suit l'un de ces motifs ne fait plus partie de la note.
CUT_PATTERNS = [
    r"\n\s*Final Answer\s*:",
    r"\n\s*```",
    r"\n\s*print\s*\(",
    r"\n\s*Patient\s*:",
    r"\n\s*Conditions\s*:",
    r"\n\s*Medication requests\s*:",
    r"\n\s*Diagnoses\s*\(Condition\)\s*:",
    r"\n\s*Medications\s*\(MedicationRequest\)\s*:",
]
# Une candidate qui contient encore l'un de ces marqueurs est rejetée.
CODE_MARKERS = ["```", "print(", "final answer", "\\boxed{"]
# Indices qu'une candidate contient encore du prompt ou du code : chacun la pénalise.
PROMPT_LEAK_MARKERS = CODE_MARKERS + [
    "patient:", "conditions:", "medication requests:",
    "diagnoses (condition):", "medications (medicationrequest):",
]


def _literal_string_or_none(s: str) -> Optional[str]:
    try:
        val = ast.literal_eval((s or "").strip())
    except (ValueError, TypeError, SyntaxError):
        return None
    return val if isinstance(val, str) else None


def _dedupe_sentences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    out, seen = [], set()
    for part in re.split(r"(?<=[.!?])\s+", text):
        p = re.sub(r"\s+", " ", part.strip())
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
    for prefix in (r"^Clinical note:\s*", r"^Final Answer:\s*", r"^The final answer is\s*"):
        s = re.sub(prefix, "", s, flags=re.I)
    s = s.strip().strip('"').strip("'").strip()

    cuts = [m.start() for m in (re.search(p, s, flags=re.I) for p in CUT_PATTERNS) if m]
    s = s[:min(cuts, default=len(s))].strip()

    s = _dedupe_sentences(re.sub(r"\s+", " ", s).strip())
    if any(tok in s.lower() for tok in CODE_MARKERS):
        return ""
    return s


def _candidate_score(text: str) -> float:
    """Plus la note est longue et ressemble à de la prose, mieux elle est notée ; chaque trace de prompt coûte cher."""
    s = (text or "").strip()
    if not s:
        return -1e9
    lower = s.lower()
    score = float(len(s))
    score -= 200.0 * sum(tok in lower for tok in PROMPT_LEAK_MARKERS)
    score += 25.0 * bool(re.search(r"[A-Za-z]", s))
    score += 25.0 * (len(s.split()) >= 8)
    score += 25.0 * bool(re.search(r"[.!?]", s))
    return score


def clean_generated_note(raw_text: str) -> str:
    raw = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    candidates = [raw]
    if re.search(r"Clinical note\s*:", raw, flags=re.I):
        candidates.append(re.split(r"Clinical note\s*:", raw, flags=re.I)[-1])
    for pat in CUT_PATTERNS[:3]:  # texte avant la réponse finale ou le code
        m = re.search(pat, raw, flags=re.I)
        if m:
            candidates.append(raw[:m.start()])
    candidates += [m.group(1) for m in re.finditer(r"\\boxed\{([^}]*)\}", raw, flags=re.S)]
    for pat in (r'print\((?P<q>"(?:[^"\\]|\\.)*")\)', r"print\((?P<q>'(?:[^'\\]|\\.)*')\)"):
        candidates += [lit for m in re.finditer(pat, raw, flags=re.S) if (lit := _literal_string_or_none(m.group("q")))]
    for pat in (r'"""(.*?)"""', r"'{3}(.*?)'{3}"):
        candidates += [b.strip() for b in (m.group(1) for m in re.finditer(pat, raw, flags=re.S)) if b and b.strip()]

    cleaned, seen = [], set()
    for cand in candidates:
        c = _normalize_candidate(cand)
        k = norm_text(c)
        if k and k not in seen:
            cleaned.append(c)
            seen.add(k)

    if not cleaned:
        return re.sub(r"\s+", " ", raw).strip()
    return max(cleaned, key=_candidate_score)


# =========================
# Génération
# =========================


def generate_notes(patients: List[dict], llm=None) -> Tuple[Dict[str, str], Dict[str, LLMResult]]:
    """Génère une note par patient et écrit les deux versions (brute et nettoyée) sur disque."""
    llm = llm or get_llm()
    raw_notes: Dict[str, str] = {}
    notes: Dict[str, LLMResult] = {}

    for p in patients:
        pid = p["patient_id"]
        if llm.available():
            raw_txt = (llm.generate(
                note_prompt(p),
                max_new_tokens=settings.NOTE_GENERATION_MAX_NEW_TOKENS,
                temperature=settings.NOTE_GENERATION_TEMPERATURE,
                do_sample=settings.NOTE_GENERATION_DO_SAMPLE,
            ) or "").strip()
            raw_notes[pid] = raw_txt
            notes[pid] = LLMResult(note=clean_generated_note(raw_txt) or raw_txt, used_llm=True,
                                   model_name=settings.MODEL_NAME)
        else:
            raw_notes[pid] = template_note(p)
            notes[pid] = LLMResult(note=raw_notes[pid], used_llm=False, model_name="template")

    for d in (settings.NOTES_DIR, settings.RAW_NOTES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for pid, res in notes.items():
        (settings.RAW_NOTES_DIR / f"{pid}.txt").write_text(raw_notes[pid], encoding="utf-8")
        (settings.NOTES_DIR / f"{pid}.txt").write_text(res.note, encoding="utf-8")
    return raw_notes, notes


def load_notes() -> Tuple[Dict[str, str], Dict[str, LLMResult]]:
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
    rows = []
    for pid, res in notes.items():
        g = gold_by_pid.get(pid, {})
        raw = raw_notes.get(pid, "")
        rows.append({
            "patient_id": pid,
            "full_name": g.get("full_name"),
            "birthDate": g.get("birthDate"),
            "gender": g.get("gender"),
            "note_style": g.get("note_style"),
            "note_style_label": g.get("note_style_label"),
            "used_llm": res.used_llm,
            "model": res.model_name,
            "raw_note_preview": shorten(raw),
            "note_text": res.note,
            "note_preview": shorten(res.note),
            "raw_chars": len(raw),
            "clean_chars": len(res.note or ""),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["note_style", "patient_id"]).reset_index(drop=True)
