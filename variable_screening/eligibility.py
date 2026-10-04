"""Conservative, reproducible first-stage screen of every Form I field.

This is an eligibility screen, not a prediction screen. It deliberately keeps
ordinary discharge fields for later penalized screening; it never looks at CHART.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .source import ID

ROLE_FIELDS = {ID: "patient key", "AInjDt": "injury-year split", "AInjAge": "age and predictor"}
NON_PREDICTORS = {ID, "AInjDt", "AAdmDt", "ARbAdmDt", "ADisDt", "ASFDate", "ANSysAdD"}
SOURCE_EXCLUSIONS = {
    "AASAC6RL": "rehabilitation-admission C6 right motor score; structural collection change after November 2000 makes cross-era prediction use unsuitable",
    "AASAL3RL": "rehabilitation-admission L3 left motor score; structural collection change after November 2000 makes cross-era prediction use unsuitable",
    "AASAS1RR": "rehabilitation-admission S1 right motor score; structural collection change after November 2000 makes cross-era prediction use unsuitable",
    "AHispnic": "detailed dictionary starts November 1995, unlike variable list; year-only public dates cannot identify structurally uncollected early records",
    "ASpkEngl": "variable list says February 1996 but detailed dictionary says the current English-ability item was added in 2011 and older language codes were converted; comparable ability cannot be established",
    "APResDis": "2000–2004 definition may refer to end of outpatient treatment rather than initial rehabilitation discharge",
}
# Verified from the public dictionary and the existing field-level feasibility audit.
# These are code semantics, not a preselected candidate list.
SPECIAL_CODES = {
    "ASex": {"9"}, "AHispnic": {"7", "9"}, "ARace": {"7", "9", "99"},
    "AMarStIj": {"9"}, "AEducLvl": {"9", "99"},
    "ATrmEtio": {"99"}, "APResInj": {"9", "99"}, "APResDis": {"9", "99"},
    "ABdMMDis": {"99"},
    "AI2RhADa": {"888", "999"}, "ASDDAc2R": {"888", "999"},
    "ASDDDurR": {"888", "999"}, "AHDaSyAc": {"8888", "9999"},
    "AHDaSyRb": {"8888", "9999"}, "AASATotD": {"888", "999"},
    "ANurLvlD": {"X99"}, "AASAImDs": {"U", "9"},
    "ANCatDis": {"9"}, "AFScorDs": {"99", "999"},
    "APrimPay": {"99"}, "AUMVAdm": {"9"}, "AUMVDis": {"9"},
    "AVeteran": {"9"}, "APrLvlSt": {"99"}, "AJobCnCd": {"99"},
    "AFmIncLv": {"9"},
}
CATEGORICAL = {
    "ASex", "AHispnic", "ARace", "AMarStIj", "AEducLvl", "ATrmEtio",
    "APResInj", "APResDis", "ABdMMDis", "ANurLvlD", "AASAImDs", "ANCatDis",
    "APrimPay", "AUMVAdm", "AUMVDis", "AVeteran", "APrLvlSt", "AJobCnCd",
}
CONTINUOUS = {
    "AInjAge", "AI2ADays", "AI2RhADa", "ASDDAc2R", "ASDDDurR",
    "AHDaSyAc", "AHDaSyRb", "AASATotD", "AFScorDs",
}
KNOWN_BOUNDS = {"AInjAge": (18, 88), "AASATotD": (0, 100), "AFScorDs": (13, 91)}
SOURCE_DIR = Path(__file__).resolve().parent / "reference"
LEVEL_FIELDS = re.compile(r"^(?:ANurLvl|ASL|AML|APNF)")
MOTOR_SCORE = re.compile(r"^AASA(?:[CTLS]\d|St)")
FIM_SCORE = re.compile(r"^AF(?:Scor|Eatg|Grmg|Bthg|DrUp|DrLo|Tltg|BdMg|BwMg|MBCW|MTlt|MTSh|LWWc|LMod|LStr)")
ANATOMICAL_LEVELS = {
    "X00",
    *(f"C{x:02d}" for x in range(1, 9)),
    *(f"T{x:02d}" for x in range(1, 13)),
    *(f"L{x:02d}" for x in range(1, 6)),
    *(f"S{x:02d}" for x in range(1, 6)),
}
UNKNOWN_LEVELS = {"C99", "T99", "L99", "S99", "X99"}


def _harmonize(field: str, values: pd.Series) -> pd.Series:
    if field != "ATrmEtio":
        return values
    def broad(raw: object) -> object:
        if pd.isna(raw):
            return pd.NA
        code = int(raw)
        if 1 <= code <= 9:
            return "vehicular"
        if 10 <= code <= 19 or code in (30, 31):
            # The March 1996 fall/assault distinction cannot be recovered
            # consistently, so these categories share one broad level.
            return "violence_or_fall"
        if 20 <= code <= 29 or 70 <= code <= 78:
            return "sports_recreation"
        return {40: "pedestrian", 50: "medical_surgical", 60: "other"}.get(code, pd.NA)
    mapped = values.map(broad).astype("string")
    if (values.notna() & mapped.isna()).any():
        raise ValueError("ATrmEtio: raw code without documented broad-group mapping")
    return mapped


def _field_line(field: str, lines: list[str]) -> str:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(field) + r"(?![A-Za-z0-9_])")
    for line in lines:
        if pattern.search(line):
            return " ".join(line.split())
    return ""


def _year_from_list(line: str, field: str) -> int | None:
    if not line:
        return None
    suffix = line.split(field, 1)[-1]
    match = re.search(r"(?:19|20)\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})", suffix)
    if not match:
        return None
    if match.group(1):
        yr = int(match.group(1))
        return 2000 + yr if yr <= 21 else 1900 + yr
    return int(match.group())


def _dictionary_page(field: str, dictionary: str) -> str:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(field) + r"(?![A-Za-z0-9_])")
    hits = list(pattern.finditer(dictionary))
    if not hits:
        return ""
    for hit in reversed(hits):
        pages = re.findall(r"--- PAGE (\d+) ---", dictionary[:hit.start()])
        if pages and int(pages[-1]) > 20:
            return pages[-1]
    return ""


def _generic_specials(field: str, dictionary: str, page: str) -> set[str]:
    """Only accept an unambiguous 'CODE Unknown/Not Done' dictionary entry."""
    if not page:
        return set()
    marker = f"--- PAGE {page} ---"
    start = dictionary.rfind(marker, 0, dictionary.find("8-Character Description:", dictionary.find(marker)))
    block = dictionary[start:dictionary.find("8-Character Description:", start) + 250]
    found = set()
    for code in ("9", "99", "999", "888", "8888", "9999"):
        if re.search(rf"(?m)^\s*{code}\s+(?:Unknown|Not Done|Not Applicable|No System|Unable)", block, re.I):
            found.add(code)
    return found


def screen_fields(cohort: pd.DataFrame, all_fields: list[str],
                  min_observed_fraction: float = .05) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return inventory, one final decision per raw field, and retained pool.

    Source evidence is recorded for exclusions too. Nothing is chosen using an
    outcome, internal validation, or temporal test distribution.
    """
    list_lines = (SOURCE_DIR / "Variable List 2016-2021_for public.txt").read_text(encoding="utf-8").splitlines()
    dictionary = (SOURCE_DIR / "Data_Dictionary_Public_AR2021.txt").read_text(encoding="utf-8")
    train = cohort.loc[cohort.split.eq("train")]
    early = train.AInjDt.between(1995, 1997)
    late = train.AInjDt.between(1998, 2000)
    rows = []
    for field in all_fields:
        line = _field_line(field, list_lines)
        page = _dictionary_page(field, dictionary)
        start_year = _year_from_list(line, field)
        raw = train[field].astype("string").str.strip().replace("", pd.NA)
        if LEVEL_FIELDS.match(field):
            raw = raw.str.upper()
        raw_missing = raw.isna()
        observed_fraction = float((~raw_missing).mean())
        early_fraction = float((~raw_missing.loc[early]).mean())
        late_fraction = float((~raw_missing.loc[late]).mean())
        codes = set(SPECIAL_CODES.get(field, _generic_specials(field, dictionary, page)))
        if MOTOR_SCORE.match(field):
            codes |= ({"88", "99"} if field.startswith("AASASt") else {"8", "9"})
        if FIM_SCORE.match(field):
            codes |= ({"99", "999"} if field.startswith("AFScor") else {"9"})
        if field.startswith("AASAIm"):
            codes |= {"U", "9"}
        if field.startswith("ANCat"):
            codes |= {"9"}
        if LEVEL_FIELDS.match(field):
            codes |= UNKNOWN_LEVELS
        cleaned = _harmonize(field, raw.mask(raw.isin(codes)))
        numeric = pd.to_numeric(cleaned, errors="coerce")
        numeric_ok = bool((cleaned.notna() & numeric.isna()).sum() == 0)
        n_levels = int(cleaned.nunique(dropna=True))
        if field in CATEGORICAL:
            kind = "categorical"
        elif field in CONTINUOUS or MOTOR_SCORE.match(field) or FIM_SCORE.match(field):
            kind = "continuous"
        elif not numeric_ok or n_levels <= 12:
            kind = "categorical"
        else:
            kind = "continuous"
        reason = ""
        if field in NON_PREDICTORS:
            reason = "join, split, or calendar date role; not a Form I main effect"
        elif field in SOURCE_EXCLUSIONS:
            reason = SOURCE_EXCLUSIONS[field]
        elif not line or not page:
            reason = "source definition or alias could not be matched to the public dictionary"
        elif start_year is None:
            reason = "collection start could not be established from the variable list"
        elif start_year > 1995:
            reason = "not collected across the full 1995–2000 development era"
        elif min(early_fraction, late_fraction) < min_observed_fraction:
            reason = "severe or structural absence in a development-era stratum (<5% raw observed)"
        elif kind == "continuous" and not numeric_ok:
            reason = "continuous values include undefined nonnumeric raw codes"
        elif kind == "categorical" and n_levels > 60:
            reason = "more than 60 raw levels without a documented stable grouping"
        bounds = KNOWN_BOUNDS.get(field)
        if MOTOR_SCORE.match(field):
            bounds = (0, 50) if field.startswith("AASASt") else (0, 5)
        if FIM_SCORE.match(field):
            bounds = (13, 91) if field.startswith("AFScor") else (1, 7)
        if not reason and bounds is not None and (numeric.dropna().lt(bounds[0]) |
                                                  numeric.dropna().gt(bounds[1])).any():
            reason = "values outside dictionary-supported bounds remain after special-code removal"
        # Suspicious sentinel-shaped values are not guessed to be scores or levels.
        if not reason and field not in SPECIAL_CODES:
            suspicious = set(cleaned.dropna().unique()) & {"888", "999", "8888", "9999"}
            if suspicious:
                reason = f"special-code meaning not verified for observed {sorted(suspicious)}"
        decision = "exclude" if reason else "retain"
        valid = cleaned.dropna()
        levels = sorted(ANATOMICAL_LEVELS) if kind == "categorical" and LEVEL_FIELDS.match(field) else (
            sorted(valid.unique().tolist()) if kind == "categorical" else [])
        rows.append({
            "raw_field": field, "source": "Form I 2021 public CSV",
            "variable_list_entry": line, "dictionary_page": page,
            "collection_start_year": start_year, "predictor_type": kind,
            "available_by_discharge": field not in NON_PREDICTORS,
            "n_train_raw_observed": int((~raw_missing).sum()),
            "n_train_clean_observed": int(cleaned.notna().sum()),
            "raw_missing_fraction_train": round(1 - observed_fraction, 6),
            "raw_observed_fraction_1995_1997": round(early_fraction, 6),
            "raw_observed_fraction_1998_2000": round(late_fraction, 6),
            "n_clean_levels_train": n_levels,
            "special_codes": "|".join(sorted(codes)),
            "legal_levels_train": "|".join(levels),
            "bound_min": bounds[0] if bounds else None,
            "bound_max": bounds[1] if bounds else None,
            "harmonization_rule": ("broad etiology groups; merge pre-1996 violence/fall ambiguity"
                                    if field == "ATrmEtio" else
                                    "special codes to missing; otherwise preserve raw scale or code"),
            "decision": decision, "reason": reason or "pre-discharge definition and development-era availability supported",
            "review_version": "first-stage-2026-10-04-v7-exclude-rehab3",
        })
    results = pd.DataFrame(rows)
    if len(results) != len(all_fields) or results.raw_field.nunique() != len(all_fields):
        raise ValueError("First-stage field inventory is incomplete")
    inventory = results[["raw_field", "source", "variable_list_entry", "dictionary_page",
                         "collection_start_year", "n_train_raw_observed", "raw_missing_fraction_train"]].copy()
    pool = results.loc[results.decision.eq("retain")].copy()
    return inventory, results, pool


def clean_candidates(frame: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning only; learned parameters live in BaselineProcessor."""
    columns = {ID: frame[ID].astype("string")}
    for row in pool.itertuples():
        field = row.raw_field
        raw = frame[field].astype("string").str.strip().replace("", pd.NA)
        if LEVEL_FIELDS.match(field):
            raw = raw.str.upper()
        codes = set(row.special_codes.split("|")) if row.special_codes else set()
        val = _harmonize(field, raw.mask(raw.isin(codes)))
        if row.predictor_type == "continuous":
            number = pd.to_numeric(val, errors="coerce")
            if (val.notna() & number.isna()).any():
                raise ValueError(f"{field}: undeclared nonnumeric code")
            lo, hi = row.bound_min, row.bound_max
            if pd.notna(lo) and number.dropna().lt(float(lo)).any():
                raise ValueError(f"{field}: below legal minimum")
            if pd.notna(hi) and number.dropna().gt(float(hi)).any():
                raise ValueError(f"{field}: above legal maximum")
            columns[field] = number.astype(float)
        else:
            legal = set(row.legal_levels_train.split("|"))
            unknown = val.notna() & ~val.isin(legal)
            if unknown.any():
                raise ValueError(f"{field}: undeclared raw codes {val[unknown].unique()[:8].tolist()}")
            columns[field] = val
        columns[field + "__raw_missing"] = raw.isna() | raw.isin(codes)
    return pd.DataFrame(columns, index=frame.index)
