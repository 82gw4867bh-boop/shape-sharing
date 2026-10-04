"""Descriptive audit of three selected rehabilitation-admission motor items.

Form II is read only to explain follow-up patterns, never to construct an
imputer or modify discharge-time predictors.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .eligibility import _harmonize
from .source import ID, ROOT, cohort_and_splits


SPLITS = ("train", "internal_validation", "temporal_test_primary")
TARGETS = ("AASAC6RL", "AASAL3RL", "AASAS1RR")
PAIRS = (("C6", "AASAC6RL", "AASAC6RR"),
         ("L3", "AASAL3RL", "AASAL3RR"),
         ("S1", "AASAS1RL", "AASAS1RR"))
DISCHARGE_PAIRS = (("C6", "AASAC6DL", "AASAC6DR"),
                   ("L3", "AASAL3DL", "AASAL3DR"),
                   ("S1", "AASAS1DL", "AASAS1DR"))
UPPER = ("C5", "C6", "C7", "C8", "T1")
LOWER = ("L2", "L3", "L4", "L5", "S1")
CHART = ("BCHPITot", "BCHMbTot", "BCHOpTot", "BCHSocIn")


def motor_value(s: pd.Series) -> pd.Series:
    value = pd.to_numeric(s, errors="coerce")
    return value.where(value.between(0, 5))


def main(archive: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        f1 = pd.read_csv(z.open(ROOT + "f1_public_2021.csv"), dtype="string", low_memory=False)
        f2 = pd.read_csv(z.open(ROOT + "f2_public_2021.csv"),
                         usecols=[ID, "BYear", "BFolUpCt", "BLostRsn", "BIntCond", *CHART],
                         dtype="string", low_memory=False)
        rec = pd.read_csv(z.open(ROOT + "rec_public_2021.csv"),
                          usecols=[ID, "SDthDt"], dtype="string", low_memory=False)
    cohort, _ = cohort_and_splits(f1)
    cohort = cohort.loc[cohort.split.isin(SPLITS)].copy()
    cohort["era"] = np.where(cohort.AInjDt.le(2000), "1995-2000", "2001-2004")
    cohort["day1_system_admit"] = cohort.AI2ADays.eq("1")
    cohort["rehab_status"] = np.where(cohort.ARbAdDtM.eq("8"), "no_system_rehab",
                                       np.where(cohort.ARbAdDtM.eq("1"), "system_rehab", "unclear"))
    cohort["triad_all_9"] = cohort[list(TARGETS)].eq("9").all(axis=1)
    cohort["triad_all_valid"] = pd.concat([motor_value(cohort[x]).notna() for x in TARGETS],
                                           axis=1).all(axis=1)
    cohort["triad_pattern"] = np.where(cohort.triad_all_9, "all_9",
                                         np.where(cohort.triad_all_valid, "all_valid", "mixed"))
    cohort["etiology_broad"] = _harmonize("ATrmEtio", cohort.ATrmEtio.mask(cohort.ATrmEtio.eq("99")))
    for field in TARGETS:
        cohort[field + "_raw_group"] = np.select(
            [cohort[field].eq("9").fillna(False).to_numpy(dtype=bool),
             cohort[field].eq("8").fillna(False).to_numpy(dtype=bool),
             motor_value(cohort[field]).notna().to_numpy(dtype=bool)],
            ["9_unknown_or_no_rehab", "8_not_testable", "valid_0_to_5"], default="blank_or_other")

    year1 = f2.loc[f2.BYear.eq("1")].copy()
    year1["has_form2_year1"] = True
    year1["year1_any_valid_chart"] = pd.concat(
        [pd.to_numeric(year1[x], errors="coerce").between(0, 100) for x in CHART], axis=1
    ).any(axis=1)
    year1 = year1[[ID, "has_form2_year1", "BFolUpCt", "BLostRsn", "BIntCond",
                   "year1_any_valid_chart"]]
    cohort = cohort.merge(year1, on=ID, how="left", validate="one_to_one")
    cohort["has_form2_year1"] = cohort.has_form2_year1.astype("boolean").fillna(False).astype(bool)
    cohort["year1_any_valid_chart"] = cohort.year1_any_valid_chart.astype("boolean").fillna(False).astype(bool)
    death = rec[[ID, "SDthDt"]].copy()
    death["death_year"] = pd.to_numeric(death.SDthDt, errors="coerce")
    cohort = cohort.merge(death[[ID, "death_year"]], on=ID, how="left", validate="one_to_one")
    cohort["dead_by_year1"] = cohort.death_year.le(cohort.AInjDt + 1).fillna(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for keys, part in cohort.groupby(["split", "rehab_status", "day1_system_admit", "triad_pattern"],
                                     dropna=False):
        summaries.append(dict(zip(["split", "rehab_status", "day1_system_admit", "triad_pattern"], keys))
                         | {"n_patients": len(part),
                            "n_year1_form2": int(part.has_form2_year1.sum()),
                            "n_year1_valid_chart": int(part.year1_any_valid_chart.sum()),
                            "n_dead_by_year1": int(part.dead_by_year1.sum()),
                            "n_rehab_admit_date_recorded": int(part.ARbAdmDt.notna().sum()),
                            "n_injury_to_rehab_888": int(part.AI2RhADa.eq("888").sum()),
                            "n_rehab_days_8888": int(part.AHDaSyRb.eq("8888").sum()),
                            "n_rehab_fim_total_valid": int(pd.to_numeric(part.AFScorRb,
                               errors="coerce").between(13, 91).sum()),
                            "n_rehab_motor_total_valid": int(pd.to_numeric(part.AASATotR,
                               errors="coerce").between(0, 100).sum()),
                            "n_discharge_motor_total_valid": int(pd.to_numeric(part.AASATotD,
                               errors="coerce").between(0, 100).sum()),
                            "n_discharge_fim_total_valid": int(pd.to_numeric(part.AFScorDs,
                               errors="coerce").between(13, 91).sum())})
    pd.DataFrame(summaries).to_csv(output_dir / "triad_by_rehab_day1_and_form2.csv",
                                   index=False, encoding="utf-8-sig")

    year_rows = []
    for (year, day1, status), part in cohort.groupby(["AInjDt", "day1_system_admit", "rehab_status"]):
        year_rows.append({"injury_year": year, "day1_system_admit": day1,
                          "rehab_status": status, "n_patients": len(part),
                          "n_triad_all_9": int(part.triad_all_9.sum()),
                          "fraction_triad_all_9": float(part.triad_all_9.mean()),
                          "n_triad_all_valid": int(part.triad_all_valid.sum())})
    pd.DataFrame(year_rows).to_csv(output_dir / "triad_by_injury_year.csv", index=False,
                                   encoding="utf-8-sig")

    for field in TARGETS:
        table = (cohort.groupby(["split", "rehab_status", "day1_system_admit",
                                 field + "_raw_group"], dropna=False).size()
                 .rename("n_patients").reset_index())
        table.to_csv(output_dir / f"{field}_raw_code_context.csv", index=False,
                     encoding="utf-8-sig")

    follow_rows = []
    for (split, pattern), part in cohort.groupby(["split", "triad_pattern"]):
        for code in ("1", "4", "5", "8", "9", "__NO_FORM2__"):
            n = int(part.BFolUpCt.fillna("__NO_FORM2__").eq(code).sum())
            follow_rows.append({"split": split, "triad_pattern": pattern,
                                "BFolUpCt": code, "n_patients": n,
                                "fraction": n / len(part)})
    pd.DataFrame(follow_rows).to_csv(output_dir / "form2_year1_followup_by_triad.csv",
                                     index=False, encoding="utf-8-sig")

    pair_rows = []
    for timing, pairs in (("rehab_admit", PAIRS), ("discharge", DISCHARGE_PAIRS)):
        for segment, left, right in pairs:
            for split in SPLITS:
                part = cohort.loc[cohort.split.eq(split)]
                l, r = motor_value(part[left]), motor_value(part[right])
                both = l.notna() & r.notna()
                gap = (l.loc[both] - r.loc[both]).abs()
                pair_rows.append({"timing": timing, "segment": segment, "split": split,
                                  "n_patients": len(part), "n_both_observed": int(both.sum()),
                                  "n_only_left_observed": int((l.notna() & r.isna()).sum()),
                                  "n_only_right_observed": int((l.isna() & r.notna()).sum()),
                                  "n_neither_observed": int((l.isna() & r.isna()).sum()),
                                  "fraction_equal_among_both": float(gap.eq(0).mean()),
                                  "fraction_gap_ge_2_among_both": float(gap.ge(2).mean()),
                                  "mean_abs_gap_among_both": float(gap.mean()),
                                  "median_abs_gap_among_both": float(gap.median()),
                                  "correlation_among_both": float(l.loc[both].corr(r.loc[both]))})
    pd.DataFrame(pair_rows).to_csv(output_dir / "left_right_motor_pairs.csv", index=False,
                                   encoding="utf-8-sig")

    asymmetry_rows = []
    for timing, pairs in (("rehab_admit", PAIRS), ("discharge", DISCHARGE_PAIRS)):
        for segment, left, right in pairs:
            for (split, etiology), part in cohort.groupby(["split", "etiology_broad"], dropna=False):
                l, r = motor_value(part[left]), motor_value(part[right])
                both = l.notna() & r.notna()
                gap = (l.loc[both] - r.loc[both]).abs()
                asymmetry_rows.append({"timing": timing, "segment": segment, "split": split,
                                       "etiology_broad": etiology, "n_patients": len(part),
                                       "n_both_observed": int(both.sum()),
                                       "fraction_gap_ge_2": float(gap.ge(2).mean()) if len(gap) else np.nan,
                                       "mean_abs_gap": float(gap.mean()) if len(gap) else np.nan})
    pd.DataFrame(asymmetry_rows).to_csv(output_dir / "motor_asymmetry_by_etiology.csv",
                                        index=False, encoding="utf-8-sig")

    aggregate_rows = []
    for timing, suffix in (("rehab_admit", "R"), ("discharge", "D")):
        upper_cols = [f"AASA{segment}{suffix}{side}" for segment in UPPER for side in ("L", "R")]
        lower_cols = [f"AASA{segment}{suffix}{side}" for segment in LOWER for side in ("L", "R")]
        all_cols = upper_cols + lower_cols
        for split in SPLITS:
            part = cohort.loc[cohort.split.eq(split)]
            upper = pd.concat([motor_value(part[x]) for x in upper_cols], axis=1)
            lower = pd.concat([motor_value(part[x]) for x in lower_cols], axis=1)
            complete_upper = upper.notna().all(axis=1)
            complete_lower = lower.notna().all(axis=1)
            complete_all = complete_upper & complete_lower
            source_total = pd.to_numeric(part[f"AASATot{suffix}"], errors="coerce")
            valid_total = source_total.between(0, 100)
            matches_total = source_total.loc[complete_all & valid_total].eq(
                upper.loc[complete_all & valid_total].sum(axis=1)
                + lower.loc[complete_all & valid_total].sum(axis=1))
            aggregate_rows.append({"timing": timing, "split": split, "n_patients": len(part),
                                   "n_complete_upper_10_items": int(complete_upper.sum()),
                                   "n_complete_lower_10_items": int(complete_lower.sum()),
                                   "n_complete_all_20_items": int(complete_all.sum()),
                                   "n_source_total_valid": int(valid_total.sum()),
                                   "n_complete_and_total_valid": int(matches_total.size),
                                   "n_complete_total_matches": int(matches_total.sum())})
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "motor_aggregate_coverage.csv",
                                        index=False, encoding="utf-8-sig")

    print("triad by context")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("left right")
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.archive, args.output_dir)
