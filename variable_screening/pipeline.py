"""CLI for the single-owner Form I -> M0 upstream workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib

from . import CONTRACT_VERSION
from .eligibility import clean_candidates, screen_fields
from .preprocess import BaselineProcessor
from .relations import all_pairs, empirical_associations, pair_review
from .screening import DEFAULT_CONFIG, screen_domain
from .source import ID, NODES, OUTCOMES, cohort_and_splits, outcome_long, read_archive


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run(archive: Path, output_dir: Path, config_path: Path,
        prior_pair_review: Path | None = None) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("screening_config.yaml keys differ from the supported configuration")
    if (config["primary_method"] != "lasso" or float(config["primary_alpha"]) != 0.5
            or [float(value) for value in config["alphas"]] != [0.5]
            or config["max_original_variables"] is not None
            or config["max_design_columns"] is not None):
        raise ValueError("formal screening requires uncapped Lasso alpha=0.5")
    output_dir.mkdir(parents=True, exist_ok=True)
    f1, f2, rec = read_archive(archive)
    cohort, cohort_counts = cohort_and_splits(f1)
    inventory, rules, pool = screen_fields(cohort, list(f1.columns))
    if rules.decision.isin(["retain", "exclude"]).sum() != len(f1.columns):
        raise ValueError("first-stage decisions are incomplete")
    if pool.empty:
        raise ValueError("first-stage candidate pool is empty")
    save_csv(inventory, output_dir / "raw_variable_inventory.csv")
    save_csv(rules, output_dir / "eligibility_screening_results.csv")
    save_csv(pool, output_dir / "candidate_variable_pool.csv")
    save_csv(rules, output_dir / "variable_dictionary.csv")
    save_csv(rules.loc[rules.decision.eq("exclude"),
                       ["raw_field", "decision", "reason", "variable_list_entry", "dictionary_page"]],
             output_dir / "decision_log.csv")
    (output_dir / "eligibility_rules.md").write_text(
        "# First-stage eligibility rules\n\n"
        "Every Form I raw field receives a final retain/exclude decision before outcome-based screening. "
        "This first stage excludes join/date roles, fields unavailable across 1995–2000, severe "
        "development-era absence (<5% observed in either 1995–1997 or 1998–2000), "
        "and raw codes that cannot be interpreted conservatively. It does not use CHART associations, "
        "internal-validation or temporal-test distributions. The public variable list and dictionary "
        "location are recorded for every field in the CSV. Retained fields are working candidates, "
        "not independently confirmed clinical effects. Revisions require a new rule version and log.\n",
        encoding="utf-8")
    split = cohort[[ID, "AInjDt", "split"]].copy()
    split["rule_version"] = rules.review_version.iloc[0]
    save_csv(split, output_dir / "patient_splits.csv")
    save_csv(split.groupby("split", as_index=False).agg(n_patients=(ID, "nunique")),
             output_dir / "split_counts.csv")
    dev = cohort.loc[cohort.split.isin(["train", "internal_validation"])].copy()
    train = dev.loc[dev.split.eq("train")]
    validation = dev.loc[dev.split.eq("internal_validation")]
    clean_train = clean_candidates(train, pool)
    clean_validation = clean_candidates(validation, pool)
    outcome = outcome_long(dev, f2, rec)
    method_frames, method_diagnostics = [], []
    kind_by_field = pool.set_index("raw_field").predictor_type.to_dict()
    for domain in OUTCOMES:
        selected, diagnostic = screen_domain(domain, clean_train, outcome, pool,
                                             config, method="lasso", fixed_alpha=0.5)
        method_frames.append(selected)
        method_diagnostics.append(diagnostic)
    method_results = pd.concat(method_frames, ignore_index=True)
    save_csv(method_results, output_dir / "screening_method_results.csv")
    (output_dir / "screening_method_diagnostics.json").write_text(
        json.dumps(method_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = pd.DataFrame([{
        "domain": item["domain"], "method": item["method"],
        "cv_patient_rmse": item["cv_patient_rmse"], "alpha": item["alpha"],
        "l1_ratio": item["l1_ratio"],
        "n_selected_raw_fields": len(item["selected_raw_fields"]),
        "n_categorical": sum(kind_by_field[f] == "categorical"
                             for f in item["selected_raw_fields"]),
        "n_continuous": sum(kind_by_field[f] == "continuous"
                            for f in item["selected_raw_fields"]),
        "n_selected_design_columns": item["selected_design_columns"],
    } for item in method_diagnostics])
    # Revised after development-data comparison; the method is now fixed for this run.
    selected_method = {domain: "lasso" for domain in OUTCOMES}
    comparison["chosen_for_m0"] = comparison.method.eq("lasso")
    comparison["role"] = "primary_revised_after_development"
    save_csv(comparison, output_dir / "screening_method_comparison.csv")
    selection = method_results.loc[[selected_method[d] == m for d, m in
                                    zip(method_results.domain, method_results.method)]].copy()
    screening_diagnostics = [item for item in method_diagnostics
                             if selected_method[item["domain"]] == item["method"]]
    save_csv(selection, output_dir / "selected_variables_by_domain.csv")
    def pair_count(raw_fields: set[str]) -> dict:
        subset = pool.loc[pool.raw_field.isin(raw_fields)]
        continuous = int(subset.predictor_type.eq("continuous").sum())
        categorical = int(subset.predictor_type.eq("categorical").sum())
        return {"n_distinct_raw_fields": len(raw_fields), "n_continuous": continuous,
                "n_categorical": categorical,
                "n_same_type_pairs": continuous * (continuous - 1) // 2 +
                                     categorical * (categorical - 1) // 2}
    active_fields = set(selection.loc[selection.decision.eq("selected"), "raw_field"])
    selection_counts = {
        "by_domain": {item["domain"]: {"n_raw_fields": len(item["selected_raw_fields"]),
                                      "n_design_columns": item["selected_design_columns"]}
                      for item in screening_diagnostics},
        **pair_count(active_fields),
    }
    (output_dir / "selection_counts.json").write_text(
        json.dumps(selection_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_raw = selection.loc[selection.decision.eq("selected"), "raw_field"].unique().tolist()
    selected_pool = pool.loc[pool.raw_field.isin(selected_raw)].copy()
    pairs = all_pairs(selected_pool, rules.review_version.iloc[0])
    prior = pd.read_csv(prior_pair_review, dtype="string") if prior_pair_review else None
    if prior is not None:
        prior = prior.merge(pairs[["variable_a", "variable_b"]],
                            on=["variable_a", "variable_b"], how="inner", validate="one_to_one")
    review, relation = pair_review(pairs, prior)
    save_csv(review, output_dir / "variable_pair_review.csv")
    save_csv(relation, output_dir / "variable_relation_table.csv")
    empirical = empirical_associations(pairs, clean_train)
    save_csv(empirical, output_dir / "variable_empirical_associations.csv")
    alerts = empirical.loc[
        empirical.method.isin(["Pearson", "Cramers V"]) &
        empirical.association_value.abs().ge(config["correlation_alert_threshold"]),
        ["variable_a", "variable_b", "method", "n_pair_observed", "association_value"]].copy()
    substitute_rows = []
    for domain in OUTCOMES:
        lookup = selection.loc[selection.domain.eq(domain)].set_index("raw_field")
        for pair in alerts.itertuples(index=False):
            a, b = pair.variable_a, pair.variable_b
            if lookup.at[a, "decision"] != "selected" or lookup.at[b, "decision"] != "selected":
                continue
            substitute_rows.append({"domain": domain, "variable_a": a, "variable_b": b,
                                    "method": pair.method, "n_pair_observed": pair.n_pair_observed,
                                    "association_value": pair.association_value,
                                    "frequency_a": lookup.at[a, "selection_frequency"],
                                    "frequency_b": lookup.at[b, "selection_frequency"],
                                    "selected_a": lookup.at[a, "decision"] == "selected",
                                    "selected_b": lookup.at[b, "decision"] == "selected",
                                    "interpretation": "descriptive correlation alert; not automatic substitution"})
    save_csv(pd.DataFrame(substitute_rows, columns=["domain", "variable_a", "variable_b",
                                                   "method", "n_pair_observed", "association_value",
                                                   "frequency_a", "frequency_b", "selected_a",
                                                   "selected_b", "interpretation"]),
             output_dir / "correlated_variable_alerts.csv")
    selected_rows = selection.loc[selection.decision.eq("selected")].copy()
    selected_raw = selected_rows.raw_field.unique().tolist()
    final_pool = pool.loc[pool.raw_field.isin(set(selected_raw) | {"ASex"})].copy()
    processor = BaselineProcessor(final_pool, config["rare_min_count"],
                                  config["include_missing_indicators"],
                                  config["continuous_imputation"], config["seed"],
                                  config["mice_max_iter"], config["mice_nearest_features"]).fit(clean_train)
    union_columns = sorted({column for encoded in selected_rows.expanded_columns
                            for column in encoded.split("|") if column})
    draw_files = []
    for draw in range(config["mice_imputations"]):
        train_prepared = processor.transform(clean_train, draw=draw)
        validation_prepared = processor.transform(clean_validation, draw=draw)
        prepared = pd.concat([train_prepared.clinical, validation_prepared.clinical], ignore_index=True)
        flags = pd.concat([train_prepared.missing, validation_prepared.missing], ignore_index=True)
        prepared = prepared[[ID, *union_columns]].merge(flags[[ID, *[f + "__raw_missing" for f in selected_raw]]],
                                                         on=ID, validate="one_to_one")
        prepared = prepared.merge(split[[ID, "split"]], on=ID, validate="one_to_one")
        baseline_name = "prepared_baseline.csv" if draw == 0 else f"prepared_baseline_imputation_{draw + 1}.csv"
        long_name = "model_ready_long.csv" if draw == 0 else f"model_ready_long_imputation_{draw + 1}.csv"
        save_csv(prepared, output_dir / baseline_name)
        long = outcome.merge(prepared[[ID, *union_columns]], on=ID, validate="many_to_one")
        save_csv(long, output_dir / long_name)
        draw_files.append({"draw": draw, "baseline": baseline_name, "long": long_name})
    joblib.dump(processor.imputer, output_dir / "mice_imputer.joblib")
    (output_dir / "preprocessor.json").write_text(json.dumps(processor.to_dict(), ensure_ascii=False,
                                                               indent=2), encoding="utf-8")
    (output_dir / "screening_diagnostics.json").write_text(json.dumps(screening_diagnostics,
                                                                       ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
    domain_design = {domain: sorted({column for encoded in selected_rows.loc[
        selected_rows.domain.eq(domain), "expanded_columns"]
                                     for column in encoded.split("|") if column}) for domain in OUTCOMES}
    counts = {"cohort": cohort_counts,
              "first_stage": {"n_raw_fields": len(f1.columns), "n_retained": len(pool),
                              "n_excluded": int(rules.decision.eq("exclude").sum()),
                              "n_continuous": int(pool.predictor_type.eq("continuous").sum()),
                              "n_categorical": int(pool.predictor_type.eq("categorical").sum())},
              "relations": {"n_pairs": len(pairs),
                            "n_pending_clinical_review": int(review.relation_level.eq("待核查").sum())},
              "observations": outcome.groupby(["split", "domain", "BYear"], as_index=False).agg(
                  n_grid_rows=(ID, "size"), n_real_valid=("observed", "count"),
                  n_evaluable=("evaluable", "sum")).to_dict("records")}
    (output_dir / "pipeline_counts.json").write_text(json.dumps(counts, ensure_ascii=False,
                                                                indent=2), encoding="utf-8")
    manifest = {
        "contract_version": CONTRACT_VERSION, "analysis_status": "development_not_frozen",
        "baseline_designation": "official_m0_reference_baseline_v1",
        "source_archive_sha256": sha256(archive), "rule_version": rules.review_version.iloc[0],
        "seed": config["seed"], "main_nodes": list(NODES), "outcomes": OUTCOMES,
        "time_formula": "(BYear-1)/5", "id_column": ID, "score_column": "observed",
        "domain_design_columns": domain_design,
        "continuous_imputation": "mice", "mice_imputations": config["mice_imputations"],
        "imputation_draw_files": draw_files,
        "imputation_fit_split": "train_only",
        "screening_method_by_domain": selected_method,
        "screening_alpha_by_domain": {domain: 0.5 for domain in OUTCOMES},
        "selection_revision": "method_changed_after_development_comparison_not_prespecified",
        "domain_selected_raw_fields": {domain: selection.loc[
            selection.domain.eq(domain) & selection.decision.eq("selected"), "raw_field"].tolist()
                                       for domain in OUTCOMES},
        "column_info": {column: processor.column_info[column] for column in union_columns},
        "groups": {raw: sorted({column for encoded in selected_rows.loc[
            selected_rows.raw_field.eq(raw), "expanded_columns"]
                                for column in encoded.split("|") if column}) for raw in selected_raw},
        "software_versions": {"python": platform.python_version(), "pandas": pd.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "scikit_learn": sklearn.__version__},
        "source_code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in (
            "source.py", "eligibility.py", "preprocess.py", "relations.py", "screening.py", "pipeline.py")},
        "files_sha256": {name: sha256(output_dir / name) for name in (
            "patient_splits.csv", "candidate_variable_pool.csv", "prepared_baseline.csv",
            "model_ready_long.csv", "selected_variables_by_domain.csv", "screening_method_results.csv",
            "screening_method_comparison.csv", "screening_method_diagnostics.json",
            "selection_counts.json",
            "variable_pair_review.csv", "variable_empirical_associations.csv", "preprocessor.json",
            "mice_imputer.joblib",
            "screening_config.yaml") if (output_dir / name).exists()},
    }
    # Copy the exact configuration beside the artifacts before hashing it.
    (output_dir / "screening_config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest["files_sha256"]["screening_config.yaml"] = sha256(output_dir / "screening_config.yaml")
    for item in draw_files:
        for name in (item["baseline"], item["long"]):
            manifest["files_sha256"][name] = sha256(output_dir / name)
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                                  indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir.resolve()), "n_raw": len(f1.columns),
                      "n_candidates": len(pool), "n_pairs": len(pairs),
                      "domain_selected": {d: len(v) for d, v in domain_design.items()}},
                     ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("screening_config.yaml"))
    parser.add_argument("--prior-pair-review", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output_dir, args.config, args.prior_pair_review)


if __name__ == "__main__":
    main()
