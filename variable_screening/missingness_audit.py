"""Patient-level missingness audit for the selected Form I baseline fields.

This audit is descriptive. It does not fit an imputer or inspect follow-up outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .eligibility import LEVEL_FIELDS, _harmonize
from .source import ID, ROOT, cohort_and_splits


SPLITS = ("train", "internal_validation", "temporal_test_primary")
SEX_CODES = {"1": "male", "2": "female", "3": "other"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(archive: Path, screening_dir: Path, output_dir: Path) -> None:
    selected = pd.read_csv(screening_dir / "selected_variables_by_domain.csv", dtype="string")
    selected = selected.loc[selected.decision.eq("selected")]
    fields = selected.raw_field.drop_duplicates().tolist()
    pool = pd.read_csv(screening_dir / "candidate_variable_pool.csv", dtype="string")
    pool = pool.loc[pool.raw_field.isin(fields)].set_index("raw_field").loc[fields].reset_index()
    with zipfile.ZipFile(archive) as z:
        f1 = pd.read_csv(z.open(ROOT + "f1_public_2021.csv"), dtype="string", low_memory=False)
    cohort, _ = cohort_and_splits(f1)
    cohort = cohort.loc[cohort.split.isin(SPLITS)].copy()
    cleaned = pd.DataFrame({ID: cohort[ID], "split": cohort.split,
                            "injury_year": cohort.AInjDt})
    for row in pool.itertuples():
        raw = cohort[row.raw_field].astype("string").str.strip().replace("", pd.NA)
        if LEVEL_FIELDS.match(row.raw_field):
            raw = raw.str.upper()
        special = set(row.special_codes.split("|")) if pd.notna(row.special_codes) else set()
        value = _harmonize(row.raw_field, raw.mask(raw.isin(special)))
        if row.predictor_type == "continuous":
            value = pd.to_numeric(value, errors="raise")
        cleaned[row.raw_field] = value
    raw_sex = cohort.ASex.astype("string").str.strip().replace("", pd.NA)
    cleaned["sex_group"] = raw_sex.map(SEX_CODES).fillna("unknown_or_missing")

    rows = []
    for field in fields:
        info = pool.set_index("raw_field").loc[field]
        for split in SPLITS:
            part = cleaned.loc[cleaned.split.eq(split)]
            n = len(part)
            missing = int(part[field].isna().sum())
            rows.append({"field": field, "type": info.predictor_type,
                         "dictionary_page": info.dictionary_page,
                         "selected_domains": "|".join(selected.loc[selected.raw_field.eq(field), "domain"]),
                         "split": split, "n_patients": n, "n_missing": missing,
                         "missing_fraction": missing / n, "n_observed": n - missing})
    rates = pd.DataFrame(rows)
    wide = rates.pivot(index=["field", "type", "dictionary_page", "selected_domains"],
                       columns="split", values=["n_missing", "missing_fraction", "n_observed"])
    wide.columns = [f"{split}_{metric}" for metric, split in wide.columns]
    wide = wide.reset_index()
    wide["temporal_minus_train_pp"] = 100 * (wide.temporal_test_primary_missing_fraction
                                             - wide.train_missing_fraction)
    wide = wide.sort_values(["train_missing_fraction", "field"], ascending=[False, True])

    sex_rows = []
    for split in SPLITS:
        part = cleaned.loc[cleaned.split.eq(split)]
        for sex in ("male", "female", "other", "unknown_or_missing"):
            group = part.loc[part.sex_group.eq(sex)]
            for field in fields:
                n = len(group)
                sex_rows.append({"split": split, "sex_group": sex, "field": field,
                                 "n_patients": n, "n_missing": int(group[field].isna().sum()),
                                 "missing_fraction": float(group[field].isna().mean()) if n else np.nan,
                                 "n_observed": int(group[field].notna().sum())})
    sex_rates = pd.DataFrame(sex_rows)

    observed_rows = []
    for field in fields:
        if pool.set_index("raw_field").at[field, "predictor_type"] != "continuous":
            continue
        for split in SPLITS:
            for sex in ("male", "female"):
                vals = cleaned.loc[cleaned.split.eq(split) & cleaned.sex_group.eq(sex), field].dropna()
                observed_rows.append({"split": split, "sex_group": sex, "field": field,
                                      "n_observed": len(vals),
                                      "mean": float(vals.mean()) if len(vals) else np.nan,
                                      "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                                      "median": float(vals.median()) if len(vals) else np.nan,
                                      "q1": float(vals.quantile(.25)) if len(vals) else np.nan,
                                      "q3": float(vals.quantile(.75)) if len(vals) else np.nan})
    observed = pd.DataFrame(observed_rows)

    continuous_rows = []
    for field in fields:
        if pool.set_index("raw_field").at[field, "predictor_type"] != "continuous":
            continue
        stats = {}
        for split in SPLITS:
            vals = cleaned.loc[cleaned.split.eq(split), field].dropna().astype(float)
            stats[split] = vals
        train_vals, test_vals = stats["train"], stats["temporal_test_primary"]
        pooled_sd = np.sqrt((train_vals.var(ddof=1) + test_vals.var(ddof=1)) / 2)
        row = {"field": field,
               "definition": pool.set_index("raw_field").at[field, "variable_list_entry"],
               "mean_difference_test_minus_train": float(test_vals.mean() - train_vals.mean()),
               "standardized_mean_difference": float((test_vals.mean() - train_vals.mean()) / pooled_sd)
               if pooled_sd > 0 else np.nan,
               "mean_ratio_test_over_train": float(test_vals.mean() / train_vals.mean())
               if train_vals.mean() != 0 else np.nan}
        for split, vals in stats.items():
            row.update({f"{split}_n_observed": len(vals),
                        f"{split}_mean": float(vals.mean()),
                        f"{split}_sd": float(vals.std(ddof=1)),
                        f"{split}_median": float(vals.median()),
                        f"{split}_q1": float(vals.quantile(.25)),
                        f"{split}_q3": float(vals.quantile(.75))})
        continuous_rows.append(row)
    continuous_comparison = pd.DataFrame(continuous_rows).sort_values(
        "standardized_mean_difference", key=lambda s: s.abs(), ascending=False)

    category_rows = []
    for field in fields:
        if pool.set_index("raw_field").at[field, "predictor_type"] != "categorical":
            continue
        for split in SPLITS:
            part = cleaned.loc[cleaned.split.eq(split), field]
            for level, n in part.value_counts(dropna=False).items():
                category_rows.append({"field": field, "split": split,
                                      "level": "__MISSING__" if pd.isna(level) else str(level),
                                      "n_patients": len(part), "n_level": int(n),
                                      "fraction": int(n) / len(part)})
    category_distribution = pd.DataFrame(category_rows)

    completeness = []
    for domain, group in selected.groupby("domain", sort=False):
        domain_fields = group.raw_field.tolist()
        for split in SPLITS:
            part = cleaned.loc[cleaned.split.eq(split)]
            count_missing = part[domain_fields].isna().sum(axis=1)
            completeness.append({"domain": domain, "split": split, "n_patients": len(part),
                                 "n_complete": int(count_missing.eq(0).sum()),
                                 "complete_fraction": float(count_missing.eq(0).mean()),
                                 "n_at_least_one_missing": int(count_missing.gt(0).sum()),
                                 "median_missing_fields": float(count_missing.median()),
                                 "p90_missing_fields": float(count_missing.quantile(.9))})
    completeness = pd.DataFrame(completeness)

    completeness_by_sex = []
    for domain, group in selected.groupby("domain", sort=False):
        domain_fields = group.raw_field.tolist()
        for split in SPLITS:
            for sex in ("male", "female"):
                part = cleaned.loc[cleaned.split.eq(split) & cleaned.sex_group.eq(sex)]
                is_complete = part[domain_fields].notna().all(axis=1)
                completeness_by_sex.append({"domain": domain, "split": split,
                                            "sex_group": sex, "n_patients": len(part),
                                            "n_complete": int(is_complete.sum()),
                                            "complete_fraction": float(is_complete.mean())})
    completeness_by_sex = pd.DataFrame(completeness_by_sex)

    year_rows = []
    year_means = []
    for year, group in cleaned.groupby("injury_year"):
        if 1995 <= year <= 2004:
            for field in fields:
                year_rows.append({"injury_year": year, "field": field,
                                  "n_patients": len(group), "n_missing": int(group[field].isna().sum()),
                                  "missing_fraction": float(group[field].isna().mean())})
                if pool.set_index("raw_field").at[field, "predictor_type"] == "continuous":
                    vals = group[field].dropna().astype(float)
                    year_means.append({"injury_year": year, "field": field,
                                       "n_observed": len(vals), "mean": float(vals.mean()),
                                       "median": float(vals.median())})
    year_rates = pd.DataFrame(year_rows)
    year_means = pd.DataFrame(year_means)

    sex_counts = (cleaned.groupby(["split", "sex_group"]).size().rename("n_patients")
                  .reset_index())
    sex_counts["fraction_in_split"] = sex_counts.n_patients / sex_counts.groupby("split").n_patients.transform("sum")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("missingness_by_field.csv", wide),
                       ("missingness_by_sex.csv", sex_rates),
                       ("observed_continuous_by_sex.csv", observed),
                       ("continuous_cross_era_comparison.csv", continuous_comparison),
                       ("categorical_cross_era_distribution.csv", category_distribution),
                       ("complete_cases_by_domain.csv", completeness),
                       ("complete_cases_by_domain_and_sex.csv", completeness_by_sex),
                       ("missingness_by_injury_year.csv", year_rates),
                       ("continuous_means_by_injury_year.csv", year_means),
                       ("sex_composition.csv", sex_counts)):
        data.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    metadata = {"source_archive_sha256": sha256(archive),
                "selection_sha256": sha256(screening_dir / "selected_variables_by_domain.csv"),
                "pool_sha256": sha256(screening_dir / "candidate_variable_pool.csv"),
                "n_distinct_selected_fields": len(fields),
                "split_n": cleaned.split.value_counts().to_dict(),
                "unit": "one Form I row per adult patient",
                "missing_definition": "blank or declared special code after the existing deterministic harmonization",
                "outcome_data_used": False,
                "temporal_status": "exploratory, same-database time holdout opened before freeze"}
    (output_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(wide[["field", "type", "train_missing_fraction",
                "temporal_test_primary_missing_fraction", "temporal_minus_train_pp"]].to_string(index=False))
    print("\nSEX\n", sex_counts.to_string(index=False))
    print("\nCOMPLETE\n", completeness.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("screening_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.archive, args.screening_dir, args.output_dir)
