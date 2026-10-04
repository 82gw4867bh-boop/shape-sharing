"""Read the public archive once and define patient-level cohorts and outcomes."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ID = "UniID"
NODES = (1, 5, 10, 15)
OUTCOMES = {
    "physical_independence": "BCHPITot",
    "mobility": "BCHMbTot",
    "occupation": "BCHOpTot",
    "social_integration": "BCHSocIn",
}
SEED = 20261003
ROOT = "NSCISC_Public_Data_2021/"


def _key(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame[ID] = frame[ID].astype("string").str.strip()
    if frame[ID].isna().any() or frame[ID].eq("").any():
        raise ValueError(f"{label}: blank UniID")
    return frame


def _unique(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    bad = frame.duplicated(cols, keep=False)
    if bad.any():
        raise ValueError(f"{label}: duplicate keys {frame.loc[bad, cols].head(8).to_dict('records')}")


def read_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """No other pipeline module opens the raw archive."""
    with zipfile.ZipFile(path) as archive:
        f1 = pd.read_csv(archive.open(ROOT + "f1_public_2021.csv"), dtype="string", low_memory=False)
        f2 = pd.read_csv(archive.open(ROOT + "f2_public_2021.csv"),
                         usecols=[ID, "BYear", *OUTCOMES.values(), "BFolUpCt", "BLostRsn", "BIntCond"],
                         dtype="string", low_memory=False)
        rec = pd.read_csv(archive.open(ROOT + "rec_public_2021.csv"),
                          usecols=lambda c: c in {ID, "SDthDt", "SStatCur"},
                          dtype="string", low_memory=False)
    for label, frame in (("Form I", f1), ("Form II", f2), ("Record Status", rec)):
        _key(frame, label)
    node = pd.to_numeric(f2["BYear"], errors="coerce")
    if node.isna().any() or not node.mod(1).eq(0).all():
        raise ValueError("Form II has missing or noninteger BYear")
    f2["BYear"] = node.astype(int).astype("string")
    _unique(f1, [ID], "Form I")
    _unique(f2, [ID, "BYear"], "Form II")
    _unique(rec, [ID], "Record Status")
    if not set(f2[ID]).issubset(set(f1[ID])) or not set(rec[ID]).issubset(set(f1[ID])):
        raise ValueError("Form II or Record Status contains an unknown UniID")
    if not set(f1[ID]).issubset(set(rec[ID])):
        raise ValueError("Record Status is missing Form I patients")
    return f1, f2, rec


def cohort_and_splits(f1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    year = pd.to_numeric(f1["AInjDt"], errors="coerce")
    raw_age = f1["AInjAge"].astype("string").str.strip()
    grouped = raw_age.isin(["0-5y", "6-12y", "13-14y", "88+y"])
    age = pd.to_numeric(raw_age.where(~grouped), errors="coerce")
    unrecognized = raw_age.notna() & ~grouped & age.isna()
    if unrecognized.any():
        raise ValueError(f"AInjAge: unrecognized codes {raw_age[unrecognized].unique()[:8].tolist()}")
    within = year.between(1995, 2019)
    adult = age.between(18, 88)
    cohort = f1.loc[within & adult].copy()
    cohort["AInjDt"] = year.loc[cohort.index].astype(int)
    cohort["AInjAge"] = age.loc[cohort.index].astype(float).astype("string")
    cohort = cohort.sort_values(ID).reset_index(drop=True)
    cohort["split"] = "outside_primary"
    dev_ids = cohort.loc[cohort.AInjDt.between(1995, 2000), ID].to_numpy()
    if len(dev_ids) < 10:
        raise ValueError("Development cohort has fewer than 10 patients")
    held = set(np.random.default_rng(SEED).permutation(dev_ids)[:round(.2 * len(dev_ids))])
    cohort.loc[cohort[ID].isin(held), "split"] = "internal_validation"
    cohort.loc[cohort.AInjDt.between(1995, 2000) & ~cohort[ID].isin(held), "split"] = "train"
    cohort.loc[cohort.AInjDt.between(2001, 2004), "split"] = "temporal_test_primary"
    counts = {"raw_form1": len(f1), "outside_injury_year": int((~within).sum()),
              "grouped_age_in_era": int((within & grouped).sum()),
              "underage_or_invalid_in_era": int((within & ~adult & ~grouped).sum()),
              "cohort": len(cohort), "seed": SEED,
              "split_counts": cohort["split"].value_counts().to_dict()}
    return cohort, counts


def outcome_long(cohort: pd.DataFrame, f2: pd.DataFrame, rec: pd.DataFrame,
                 nodes: tuple[int, ...] = NODES) -> pd.DataFrame:
    """Create a complete prediction grid; observed CHART is never imputed."""
    base = cohort[[ID, "AInjDt", "split"]]
    grid = base.merge(pd.DataFrame({"BYear": nodes}), how="cross")
    grid["t"] = (grid.BYear - 1) / 5
    follow = f2.copy()
    follow["BYear"] = pd.to_numeric(follow.BYear, errors="coerce")
    if (follow.BYear.notna() & ~follow.BYear.mod(1).eq(0)).any():
        raise ValueError("Form II BYear is not integer")
    follow = follow.loc[follow.BYear.isin(nodes)].copy()
    follow["BYear"] = follow.BYear.astype(int)
    death = rec[[ID, "SDthDt"]].copy()
    death["SDthDt"] = pd.to_numeric(death.SDthDt, errors="coerce")
    illegal_death = death.SDthDt.notna() & ~death.SDthDt.between(1900, 2021)
    if illegal_death.any():
        raise ValueError(f"Record Status: invalid death year {death.loc[illegal_death, ID].head().tolist()}")
    grid = grid.merge(death, on=ID, validate="many_to_one")
    grid["death_timing"] = np.select(
        [grid.SDthDt.lt(grid.AInjDt + grid.BYear).fillna(False).to_numpy(dtype=bool),
         grid.SDthDt.eq(grid.AInjDt + grid.BYear).fillna(False).to_numpy(dtype=bool)],
        ["after_death", "same_year_uncertain"], default="not_known_dead_by_node")
    joined = grid.merge(follow[[ID, "BYear", *OUTCOMES.values(), "BFolUpCt", "BLostRsn", "BIntCond"]],
                        on=[ID, "BYear"], how="left", validate="one_to_one", indicator="_form2")
    rows = []
    for domain, col in OUTCOMES.items():
        part = joined[[ID, "split", "AInjDt", "BYear", "t", "death_timing", "_form2", col]].copy()
        value = pd.to_numeric(part[col], errors="coerce")
        raw = part[col].astype("string").str.strip()
        bad_text = raw.notna() & raw.ne("") & value.isna()
        if bad_text.any():
            raise ValueError(f"{col}: unrecognized outcome code {raw[bad_text].unique()[:8].tolist()}")
        illegal = value.notna() & ~value.between(0, 100) & ~value.eq(999)
        if illegal.any():
            raise ValueError(f"{col}: score outside 0–100 or 999")
        part["observed"] = value.mask(value.eq(999))
        part["domain"] = domain
        part["has_form2"] = part["_form2"].eq("both")
        part["evaluable"] = part.observed.notna() & part.death_timing.eq("not_known_dead_by_node")
        rows.append(part.drop(columns=[col, "_form2"]))
    result = pd.concat(rows, ignore_index=True)
    _unique(result, [ID, "BYear", "domain"], "model-ready outcome grid")
    return result
