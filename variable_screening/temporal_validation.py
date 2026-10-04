"""Evaluate the saved Lasso 0.5 -> M0 fixed effects on the 2001-2004 cohort."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from m0.mixed_effects_m0_template import load_contract
from .eligibility import clean_candidates
from .preprocess import BaselineProcessor
from .source import ID, NODES, OUTCOMES, cohort_and_splits, outcome_long, read_archive


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Official 2021 public data dictionary: marital status at injury, PDF page 170;
# neurologic impairment category at discharge, PDF page 255. These are valid
# source codes absent from the development patients, not estimated new levels.
DOCUMENTED_UNSEEN = {"AMarStIj": {"7"}, "ANCatDis": {"8"}}


def prepare_processor(pool: pd.DataFrame, saved: dict, selected: set[str]) -> tuple[BaselineProcessor, list[dict]]:
    """Keep trained parameters intact; extend only the official raw-code vocabulary."""
    subset = pool.loc[pool.raw_field.isin(selected)].copy()
    if set(subset.raw_field) != selected:
        raise ValueError("selected field missing from saved candidate pool")
    obj = copy.deepcopy(saved)
    obj["parameters"] = {k: v for k, v in obj["parameters"].items() if k in selected}
    obj["groups"] = {k: v for k, v in obj["groups"].items() if k in selected}
    obj["columns"] = [c for c in obj["columns"] if obj["column_info"][c]["raw_field"] in selected]
    obj["column_info"] = {c: obj["column_info"][c] for c in obj["columns"]}
    if set(obj["parameters"]) != selected or set(obj["groups"]) != selected:
        raise ValueError("frozen processor does not cover selected fields")
    extensions = []
    for field, codes in DOCUMENTED_UNSEEN.items():
        if field not in selected:
            continue
        row = subset.loc[subset.raw_field.eq(field)].iloc[0]
        if row.predictor_type != "categorical":
            raise ValueError(f"{field}: expected categorical field")
        original = set(row.legal_levels_train.split("|"))
        learned = set(obj["parameters"][field]["legal_raw"])
        if original != learned:
            raise ValueError(f"{field}: legal vocabulary differs between pool and processor")
        added = codes - original
        subset.loc[subset.raw_field.eq(field), "legal_levels_train"] = "|".join(sorted(original | codes))
        obj["parameters"][field]["legal_raw"] = sorted(learned | codes)
        for code in sorted(added):
            extensions.append({"field": field, "code": code, "dictionary_page": 170 if field == "AMarStIj" else 255,
                               "mapped_to": "__OTHER__" if "__OTHER__" in obj["parameters"][field]["levels"]
                               else obj["parameters"][field]["reference"]})
    return BaselineProcessor.from_dict(subset, obj), extensions


def score_fixed_effects(long: pd.DataFrame, clinical: pd.DataFrame, manifest: dict, models: list[dict]) -> pd.DataFrame:
    """Score every eligible patient-node with saved fixed coefficients only."""
    if set(long.domain) != set(OUTCOMES) or not long.BYear.isin(NODES).all():
        raise ValueError("temporal evaluation requires four domains and main nodes only")
    if long.duplicated([ID, "BYear", "domain"]).any() or clinical[ID].duplicated().any():
        raise ValueError("duplicate temporal key")
    if set(long[ID]) != set(clinical[ID]):
        raise ValueError("temporal baseline and outcome patients differ")
    if not long.split.eq("temporal_test_primary").all():
        raise ValueError("non-temporal patient entered temporal evaluation")
    by_domain = {m["domain"]: m for m in models}
    if set(by_domain) != set(OUTCOMES) or len(models) != len(OUTCOMES):
        raise ValueError("M0 must contain four unique domain models")
    all_columns = sorted({c for cols in manifest["domain_design_columns"].values() for c in cols})
    if not set(all_columns).issubset(clinical.columns):
        raise ValueError("temporal design column missing")
    if not np.isfinite(clinical[all_columns].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite temporal design")
    scored = long.merge(clinical[[ID, *all_columns]], on=ID, validate="many_to_one")
    scored["prediction"] = np.nan
    scored["prediction_applicable"] = scored.death_timing.eq("not_known_dead_by_node")
    for domain in OUTCOMES:
        model = by_domain[domain]
        if not model["converged"]:
            raise ValueError(f"{domain}: M0 fit did not converge")
        columns = manifest["domain_design_columns"][domain]
        names = ["const", "t", *columns]
        if set(model["fixed_effects"]) != set(names):
            raise ValueError(f"{domain}: fixed-effect design differs from manifest")
        mask = scored.domain.eq(domain) & scored.prediction_applicable
        x = np.column_stack([np.ones(mask.sum()), scored.loc[mask, ["t", *columns]].to_numpy(dtype=float)])
        beta = np.asarray([model["fixed_effects"][name] for name in names], dtype=float)
        if not np.isfinite(beta).all():
            raise ValueError(f"{domain}: nonfinite fixed effect")
        scored.loc[mask, "prediction"] = x @ beta
    if not scored.evaluable.eq(scored.observed.notna() & scored.prediction_applicable).all():
        raise ValueError("outcome evaluability or death rule mismatch")
    if scored.loc[scored.evaluable, "prediction"].isna().any():
        raise ValueError("evaluable record lacks prediction")
    return scored


def metric_rows(scored: pd.DataFrame, by_node: bool) -> pd.DataFrame:
    keys = ["domain", "BYear"] if by_node else ["domain"]
    rows = []
    for key, group in scored.groupby(keys, sort=True):
        valid = group.loc[group.evaluable]
        error = valid.prediction - valid.observed
        if len(valid) >= 3 and valid.prediction.std(ddof=0) > 0:
            slope, intercept = np.polyfit(valid.prediction.to_numpy(dtype=float),
                                          valid.observed.to_numpy(dtype=float), 1)
        else:
            slope, intercept = np.nan, np.nan
        applicable = group.loc[group.prediction_applicable]
        record = {"domain": key[0],
                  "n_patients": group[ID].nunique(), "n_grid_rows": len(group),
                  "n_prediction_applicable": len(applicable), "n_has_form2": int(group.has_form2.sum()),
                  "n_evaluable": len(valid), "n_evaluable_patients": valid[ID].nunique(),
                  "n_after_death": int(group.death_timing.eq("after_death").sum()),
                  "n_same_year_uncertain": int(group.death_timing.eq("same_year_uncertain").sum()),
                  "mae": error.abs().mean(), "rmse": np.sqrt(np.mean(error ** 2)),
                  "bias_pred_minus_obs": error.mean(),
                  "mean_prediction_evaluable": valid.prediction.mean(),
                  "mean_observed_evaluable": valid.observed.mean(),
                  "calibration_intercept": intercept, "calibration_slope": slope,
                  "fraction_prediction_outside_0_100": (~applicable.prediction.between(0, 100)).mean(),
                  "fraction_observed_at_100": valid.observed.eq(100).mean()}
        if by_node:
            record["BYear"] = key[1]
        rows.append(record)
    return pd.DataFrame(rows)


def change_rows(scored: pd.DataFrame) -> pd.DataFrame:
    """Adjacent-node changes among patients evaluable at both nodes."""
    records = []
    for domain, part in scored.groupby("domain", sort=True):
        valid = part.loc[part.evaluable].pivot(index=ID, columns="BYear", values=["observed", "prediction"])
        for first, second in zip(NODES[:-1], NODES[1:]):
            pair = valid[[('observed', first), ('observed', second),
                          ('prediction', first), ('prediction', second)]].dropna()
            observed_change = pair['observed', second] - pair['observed', first]
            predicted_change = pair['prediction', second] - pair['prediction', first]
            error = predicted_change - observed_change
            records.append({"domain": domain, "from_node": first, "to_node": second,
                            "n_paired_evaluable": len(pair),
                            "mean_observed_change": observed_change.mean(),
                            "mean_predicted_change": predicted_change.mean(),
                            "change_mae": error.abs().mean(),
                            "change_rmse": np.sqrt(np.mean(error ** 2)),
                            "change_bias_pred_minus_obs": error.mean()})
    return pd.DataFrame(records)


def run(archive: Path, upstream: Path, m0_dir: Path, output: Path,
        allow_development: bool = False) -> None:
    manifest, _ = load_contract(upstream, allow_development=allow_development)
    if sha256(archive) != manifest["source_archive_sha256"]:
        raise ValueError("raw archive hash differs from development source")
    for filename in ("source.py", "eligibility.py", "preprocess.py"):
        if sha256(Path(__file__).with_name(filename)) != manifest["source_code_sha256"][filename]:
            raise ValueError(f"{filename}: upstream source code changed")
    if tuple(manifest["main_nodes"]) != NODES or manifest["outcomes"] != OUTCOMES:
        raise ValueError("upstream nodes or outcomes differ")
    metadata = json.loads((m0_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if (metadata["upstream_manifest_sha256"] != sha256(upstream / "feature_manifest.json")
            or metadata["analysis_status"] != manifest["analysis_status"]
            or metadata["screening_method_by_domain"] != {d: "lasso" for d in OUTCOMES}
            or metadata["screening_alpha_by_domain"] != {d: 0.5 for d in OUTCOMES}
            or set(metadata["evaluated_splits"]) != {"train", "internal_validation"}):
        raise ValueError("M0 metadata does not match frozen selection contract")
    f1, f2, rec = read_archive(archive)
    cohort, _ = cohort_and_splits(f1)
    saved_splits = pd.read_csv(upstream / "patient_splits.csv", dtype={ID: "string"})
    comparison = cohort[[ID, "AInjDt", "split"]].merge(saved_splits[[ID, "AInjDt", "split"]],
             on=ID, how="outer", suffixes=("_new", "_saved"), indicator=True, validate="one_to_one")
    if not comparison._merge.eq("both").all() or not comparison.AInjDt_new.eq(comparison.AInjDt_saved).all() or not comparison.split_new.eq(comparison.split_saved).all():
        raise ValueError("recomputed patient split differs from upstream")
    temporal = cohort.loc[cohort.split.eq("temporal_test_primary")].copy()
    if not temporal.AInjDt.between(2001, 2004).all() or temporal[ID].isin(
            cohort.loc[cohort.split.isin(["train", "internal_validation"]), ID]).any():
        raise ValueError("temporal cohort boundary violated")
    selected = {field for fields in manifest["domain_selected_raw_fields"].values() for field in fields}
    selected.add("ASex")  # frozen MICE auxiliary, including when not a domain main effect
    pool = pd.read_csv(upstream / "candidate_variable_pool.csv", dtype="string")
    pool["special_codes"] = pool["special_codes"].fillna("")
    saved_processor = json.loads((upstream / "preprocessor.json").read_text(encoding="utf-8"))
    processor, extensions = prepare_processor(pool, saved_processor, selected)
    if manifest.get("continuous_imputation") != "mice":
        raise ValueError("temporal evaluation expects the MICE development contract")
    processor.imputer = joblib.load(upstream / "mice_imputer.joblib")
    # No outcome is used to determine whether a source code is documented.
    code_counts = []
    for extension in extensions:
        mask = temporal[extension["field"]].astype("string").str.strip().eq(extension["code"])
        code_counts.append({**extension, "n_temporal_patients": int(mask.sum())})
    cleaned = clean_candidates(temporal, processor.pool)
    long = outcome_long(temporal, f2, rec, nodes=NODES)
    draw_scores = []
    for draw in range(manifest["mice_imputations"]):
        source_draw = manifest.get("selected_mice_draw_index", draw)
        clinical = processor.transform(cleaned, draw=source_draw).clinical
        models = [m["imputation_models"][draw] for m in metadata["models"]]
        draw_scores.append(score_fixed_effects(long, clinical, manifest, models))
    scored = draw_scores[0].copy()
    scored["prediction"] = np.mean(
        np.stack([part.prediction.to_numpy(dtype=float) for part in draw_scores]), axis=0)
    scored.loc[~scored.prediction_applicable, "prediction"] = np.nan
    keep = ["split", ID, "AInjDt", "BYear", "t", "domain", "observed", "prediction",
            "prediction_applicable", "evaluable", "death_timing", "has_form2"]
    by_node = metric_rows(scored, True)
    pooled = metric_rows(scored, False)
    changes = change_rows(scored)
    output.mkdir(parents=True, exist_ok=True)
    scored[keep].to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    by_node.to_csv(output / "metrics_by_node.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(output / "metrics_pooled.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(output / "paired_changes.csv", index=False, encoding="utf-8-sig")
    audit = {"analysis_status": "official_m0_baseline_external_time_validation",
             "baseline_designation": manifest.get("baseline_designation"),
             "temporal_set_status": "patient_disjoint_injury_year_holdout",
             "data_independence": "same public database; temporal cohort is not a separate database",
             "cohort": "adult SCI, injury years 2001-2004", "n_temporal_patients": len(temporal),
             "nodes": list(NODES), "domains": list(OUTCOMES), "source_archive_sha256": sha256(archive),
             "upstream_manifest_sha256": sha256(upstream / "feature_manifest.json"),
             "m0_run_metadata_sha256": sha256(m0_dir / "run_metadata.json"),
             "temporal_validation_code_sha256": sha256(Path(__file__)),
             "documented_codes_unseen_in_development": code_counts,
             "prediction_rule": "saved M0 fixed effects only; no refit or patient random effect",
             "imputation_rule": "one previously saved MICE completion; development-train-fitted imputer and selected source draw index",
             "selected_mice_draw_index": manifest.get("selected_mice_draw_index", 0),
             "outcome_rule": "observed 0-100 only; after death and same-year uncertainty excluded",
             "no_20_year_evaluation": True}
    (output / "run_metadata.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False))
    print(f"Saved temporal validation to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("m0_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-development", action="store_true")
    args = parser.parse_args()
    run(args.archive, args.upstream, args.m0_dir, args.output, args.allow_development)


if __name__ == "__main__":
    main()
