# ml/column_mapper.py
"""
Multilingualer, KI-gestützter Spalten-Mapper.
- Verwendet sentence-transformers (multilingual) für semantische Ähnlichkeit.
- Fallback: difflib Sequenz-Ähnlichkeit, falls Modell nicht vorhanden.
- Kombiniert semantische Scores mit einfachen Heuristiken (Regel-Boosts).
- Liefert:
    mapping:    {original_col -> internal_key}
    confidences:{original_col -> 0..1}
    unknown:    [original_col]  # nicht sicher zuordenbar
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import difflib
import re

# ------------------------------
# Internes Zielschema (multilingual)
# key -> (anzeige_titel, beschreibung, synonyme_liste)
# Synonyme in EN/DE/ES/FR/ZH (weitere lassen sich leicht ergänzen)
# ------------------------------
INTERNAL_SCHEMA: Dict[str, Tuple[str, str, List[str]]] = {
    "Application": (
        "Application",
        "Maschinen-/Prozesstyp bzw. Einsatzgebiet",
        [
            # EN
            "application", "use case", "machine type", "category", "process type",
            # DE
            "anwendung", "einsatz", "einsatzgebiet", "maschinenart", "kategorie",
            # ES
            "aplicación", "tipo de máquina", "categoría", "tipo de proceso",
            # FR
            "application", "type de machine", "catégorie", "type de procédé",
            # ZH
            "应用", "用途", "机器类型", "类别"
        ],
    ),
    "Sub Application": (
        "Sub Application",
        "Unterkategorie / feinere Einordnung",
        [
            # EN
            "sub application", "sub-category", "sub category", "subcategory", "segment",
            # DE
            "unteranwendung", "unterkategorie", "bereich", "teilbereich",
            # ES
            "subaplicación", "subcategoría", "segmento",
            # FR
            "sous-application", "sous-catégorie", "segment",
            # ZH
            "子应用", "子类别", "细分"
        ],
    ),
    "Purchase Price": (
        "Purchase Price",
        "Anschaffungs-/Listenpreis der Maschine",
        [
            # EN
            "purchase price", "capex", "list price", "investment", "cost price", "unit price",
            # DE
            "kaufpreis", "anschaffungspreis", "listenpreis", "investition",
            # ES
            "precio de compra", "precio lista", "inversión", "capex",
            # FR
            "prix d'achat", "prix catalogue", "investissement",
            # ZH
            "购买价格", "采购价", "清单价", "投资"
        ],
    ),
    "Power (kW)": (
        "Power (kW)",
        "Leistungsaufnahme / Motorleistung in kW",
        [
            # EN
            "power", "motor power", "rated power", "kw", "electric power", "consumption (kw)",
            # DE
            "leistung", "motorleistung", "nennleistung", "kw", "stromverbrauch", "leistung (kw)",
            # ES
            "potencia", "potencia del motor", "kW", "consumo eléctrico",
            # FR
            "puissance", "puissance moteur", "kW", "consommation électrique",
            # ZH
            "功率", "电机功率", "千瓦", "kW", "功率消耗"
        ],
    ),
    "Maintenance Cost": (
        "Maintenance Cost",
        "Regelmäßige Service-/Wartungskosten pro Jahr",
        [
            # EN
            "maintenance", "maintenance cost", "service cost", "o&m", "upkeep",
            # DE
            "wartung", "wartungskosten", "servicekosten", "instandhaltung",
            # ES
            "mantenimiento", "coste de mantenimiento", "coste de servicio",
            # FR
            "maintenance", "coût de maintenance", "coût de service",
            # ZH
            "维护", "维护成本", "服务成本", "运维"
        ],
    ),
    "Flow Rate (L/s)": (
        "Flow Rate (L/s)",
        "Durchflussrate/Volumenstrom (z. B. Wasser, Prozessmedium)",
        [
            # EN
            "flow", "flow rate", "l/s", "throughput", "capacity (l/s)", "inlet flow",
            # DE
            "durchfluss", "durchflussrate", "l/s", "förderstrom", "volumenstrom",
            # ES
            "caudal", "caudal (l/s)", "flujo", "tasa de flujo",
            # FR
            "débit", "débit (l/s)", "flux",
            # ZH
            "流量", "流速", "L/s", "体积流量"
        ],
    ),
    "Drum Size (mm)": (
        "Drum Size (mm)",
        "Trommeldurchmesser / Baugröße",
        [
            # EN
            "drum size", "bowl diameter", "diameter", "mm",
            # DE
            "trommeldurchmesser", "trommelgröße", "durchmesser", "mm",
            # ES
            "diámetro", "tamaño del tambor",
            # FR
            "diamètre", "taille du tambour",
            # ZH
            "滚筒直径", "直径", "毫米"
        ],
    ),
    "Weight (kg)": (
        "Weight (kg)",
        "Gewicht / Masse der Maschine",
        [
            # EN
            "weight", "mass", "total weight", "kg", "machine weight",
            # DE
            "gewicht", "gesamtgewicht", "masse", "kg",
            # ES
            "peso", "kg",
            # FR
            "poids", "kg",
            # ZH
            "重量", "千克", "kg"
        ],
    ),
}

# Welche Felder brauchst du mindestens fürs UI-Filtering
REQUIRED_KEYS: List[str] = ["Application", "Sub Application"]

# ------------------------------
# Utility: Normalisierung & Cosine
# ------------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _cos(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return (dot / (na * nb)) if na and nb else 0.0

# ------------------------------
# Optional: Multilinguales Embedding-Modell
# ------------------------------
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        # Multilingual, klein & schnell, gut für Spaltennamen
        _MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        _MODEL = None
    return _MODEL

# ------------------------------
# Heuristik-Boosts (Regeln)
# Erhöht Score bei passenden Mustern (z. B. kW -> Power)
# ------------------------------
def _heuristic_boost(col: str, key: str) -> float:
    c = _norm(col)

    # Power Hinweise
    if key == "Power (kW)":
        if re.search(r"\bkw\b", c) or "motor" in c or "leistung" in c or "power" in c:
            return 0.15
        if "consumption" in c or "verbrauch" in c:
            return 0.10

    # Purchase Price
    if key == "Purchase Price":
        if "price" in c or "preis" in c or "capex" in c or "invest" in c or "list" in c or "成本" in c:
            return 0.15

    # Flow
    if key == "Flow Rate (L/s)":
        if "l/s" in c or "flow" in c or "durchfluss" in c or "caudal" in c or "débit" in c or "流量" in c:
            return 0.15

    # Weight
    if key == "Weight (kg)":
        if "kg" in c or "weight" in c or "gewicht" in c or "poids" in c or "重量" in c:
            return 0.15

    # Drum Size
    if key == "Drum Size (mm)":
        if "mm" in c or "diameter" in c or "durchmesser" in c or "直径" in c:
            return 0.15

    # Application / Sub Application – häufige Wörter
    if key in ("Application", "Sub Application"):
        if any(w in c for w in ["application", "anwendung", "aplicación", "sous", "unter", "子应用", "类别", "category"]):
            return 0.10

    return 0.0

# ------------------------------
# String-Ähnlichkeit (Fallback)
# ------------------------------
def _string_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_norm(a), b=_norm(b)).ratio()

# ------------------------------
# Prompts für Ziel-Felder
# ------------------------------
def _build_target_prompts() -> Tuple[List[str], List[str]]:
    keys = list(INTERNAL_SCHEMA.keys())
    prompts = []
    for k in keys:
        title, desc, syns = INTERNAL_SCHEMA[k]
        # Prompt kombiniert Titel + Beschreibung + Synonyme
        text = " | ".join([title, desc] + syns)
        prompts.append(text)
    return keys, prompts

# ------------------------------
# Hauptfunktion: Mapping vorschlagen
# ------------------------------
def suggest_mapping(columns: List[str], use_embeddings: bool = True, fast_mode: bool = False):
    """
    columns: Liste hochgeladener Spaltennamen
    returns:
        mapping     {original_col -> internal_key}
        confidences {original_col -> 0..1}
        unknown     [original_col]
    """
    mapping: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    unknown: List[str] = []

    keys, prompts = _build_target_prompts()

    # --- FAST MODE: Nur String + Heuristik, kein KI-Modell ---
    if fast_mode:
        for col in columns:
            best_key, best_score = None, 0.0
            for k, (title, desc, syns) in INTERNAL_SCHEMA.items():
                score = max(
                    _string_sim(col, k),
                    _string_sim(col, title),
                    *[_string_sim(col, s) for s in syns]
                )
                score += _heuristic_boost(col, k)
                if score > best_score:
                    best_key, best_score = k, score
            best_score = max(0.0, min(1.0, best_score))
            mapping[col] = best_key
            confidences[col] = best_score
        return mapping, confidences, unknown

    # --- Normal Mode (ggf. mit Embeddings) ---
    model = _get_model() if use_embeddings else None
    target_vecs = None
    if model:
        try:
            target_vecs = model.encode(prompts, convert_to_numpy=False)
        except Exception:
            target_vecs, model = None, None

    for col in columns:
        best_key, best_score = None, 0.0

        if model and target_vecs is not None:
            try:
                vec = model.encode([col], convert_to_numpy=False)[0]
                for i, tv in enumerate(target_vecs):
                    score = _cos(vec, tv) + _heuristic_boost(col, keys[i])
                    if score > best_score:
                        best_key, best_score = keys[i], score
            except Exception:
                pass

        # Fallback: String-Sim
        if best_key is None:
            for k, (title, desc, syns) in INTERNAL_SCHEMA.items():
                score = max(
                    _string_sim(col, k),
                    _string_sim(col, title),
                    *[_string_sim(col, s) for s in syns]
                )
                score += _heuristic_boost(col, k)
                if score > best_score:
                    best_key, best_score = k, score

        best_score = max(0.0, min(1.0, best_score))
        mapping[col] = best_key
        confidences[col] = best_score

    return mapping, confidences, unknown

# ------------------------------
# Pflichtfelder prüfen
# ------------------------------
def ensure_required(mapped_df_columns: List[str]) -> List[str]:
    missing = []
    for req in REQUIRED_KEYS:
        if req not in mapped_df_columns:
            missing.append(req)
    return missing
