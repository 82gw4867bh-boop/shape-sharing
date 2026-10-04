"""Fit four independent random-intercept M0 models from upstream artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from variable_screening import CONTRACT_VERSION
from variable_screening.source import ID, NODES, OUTCOMES


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(directory: Path, allow_development: bool = False) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("upstream contract version mismatch")
    if (manifest.get("analysis_status") != "frozen"
            and manifest.get("baseline_designation") != "official_m0_reference_baseline_v2_selected_existing_draw"
            and not allow_development):
        raise ValueError("unfrozen selection requires --allow-development")
    for name, expected in manifest["files_sha256"].items():
        if _hash(directory / name) != expected:
            raise ValueError(f"upstream hash mismatch: {name}")
    splits = pd.read_csv(directory / "patient_splits.csv", dtype={ID: "string"})
    baseline = pd.read_csv(directory / "prepared_baseline.csv", dtype={ID: "string"})
    long = pd.read_csv(directory / "model_ready_long.csv", dtype={ID: "string"})
    selection = pd.read_csv(directory / "selected_variables_by_domain.csv", dtype="string")
    pool = pd.read_csv(directory / "candidate_variable_pool.csv", dtype="string")
    if splits[ID].duplicated().any() or baseline[ID].duplicated().any():
        raise ValueError("duplicate patient in split or baseline")
    if long.duplicated([ID, "BYear", "domain"]).any():
        raise ValueError("duplicate patient × node × domain key")
    if not long.groupby(ID).size().eq(len(NODES) * len(OUTCOMES)).all():
        raise ValueError("incomplete patient × node × domain prediction grid")
    if not long[ID].isin(splits[ID]).all() or not long[ID].isin(baseline[ID]).all():
        raise ValueError("model-ready row has no patient-level record")
    true_split = splits.set_index(ID).split
    if not long.split.eq(long[ID].map(true_split)).all() or not baseline.split.eq(
            baseline[ID].map(true_split)).all():
        raise ValueError("patient split mismatch")
    if not long.split.isin(["train", "internal_validation"]).all() or not long.BYear.isin(NODES).all():
        raise ValueError("sealed temporal or 20-year data entered M0")
    if set(long.domain) != set(OUTCOMES):
        raise ValueError("all four CHART domains required")
    if not np.allclose(long.t.to_numpy(dtype=float), (long.BYear.to_numpy(dtype=float) - 1) / 5):
        raise ValueError("incorrect M0 time column")
    evaluable = long.evaluable.astype("string").str.lower()
    if not evaluable.isin(["true", "false"]).all():
        raise ValueError("evaluable must be boolean")
    long["evaluable"] = evaluable.eq("true")
    long["observed"] = pd.to_numeric(long.observed, errors="coerce")
    if (long.observed.notna() & ~long.observed.between(0, 100)).any():
        raise ValueError("CHART outcome outside 0–100")
    if not long.evaluable.eq(long.observed.notna() &
                            long.death_timing.eq("not_known_dead_by_node")).all():
        raise ValueError("evaluable disagrees with observed score or death status")
    if not selection.raw_field.isin(pool.raw_field).all():
        raise ValueError("selected variable outside first-stage pool")
    method_by_domain = manifest.get("screening_method_by_domain")
    alpha_by_domain = manifest.get("screening_alpha_by_domain")
    if method_by_domain != {domain: "lasso" for domain in OUTCOMES}:
        raise ValueError("M0 requires Lasso selection in all domains")
    if alpha_by_domain != {domain: 0.5 for domain in OUTCOMES}:
        raise ValueError("M0 requires Lasso alpha=0.5 in all domains")
    if "alpha" not in selection:
        raise ValueError("selected-variable alpha is missing")
    for domain, method in method_by_domain.items():
        selected_rows = selection.loc[selection.domain.eq(domain)]
        if not selected_rows.method.eq(method).all():
            raise ValueError(f"{domain}: screening method mismatch")
        if not pd.to_numeric(selected_rows.alpha, errors="coerce").eq(0.5).all():
            raise ValueError(f"{domain}: screening alpha mismatch")
    for domain, cols in manifest["domain_design_columns"].items():
        if domain not in OUTCOMES or not set(cols).issubset(long.columns):
            raise ValueError(f"{domain}: missing design columns")
        if any(c.startswith("domain") or c in OUTCOMES.values() for c in cols):
            raise ValueError(f"{domain}: forbidden domain or outcome term")
        if cols and not np.isfinite(long[cols].to_numpy(dtype=float)).all():
            raise ValueError(f"{domain}: nonfinite design")
        selected = set(selection.loc[selection.domain.eq(domain) & selection.decision.eq("selected"),
                                     "raw_field"])
        if selected != set(manifest["domain_selected_raw_fields"][domain]):
            raise ValueError(f"{domain}: selected list mismatch")
        if "expanded_columns" in selection:
            mapped = {column for value in selection.loc[
                selection.domain.eq(domain) & selection.decision.eq("selected"), "expanded_columns"].fillna("")
                      for column in value.split("|") if column}
            if mapped != set(cols):
                raise ValueError(f"{domain}: selected expanded columns differ from manifest")
    all_design = sorted({c for cols in manifest["domain_design_columns"].values() for c in cols})
    if all_design:
        expected = long[ID].map(baseline.set_index(ID)[all_design].to_dict("index"))
        for col in all_design:
            reference = expected.map(lambda record: record[col]).to_numpy(dtype=float)
            if not np.allclose(long[col].to_numpy(dtype=float), reference):
                raise ValueError(f"{col}: model-ready design differs from prepared baseline")
    for field, cols in manifest["groups"].items():
        onehot = [c for c in cols if manifest["column_info"][c]["kind"] == "one_hot"]
        if onehot:
            values = baseline[onehot].to_numpy(dtype=float)
            if not np.isin(values, [0.0, 1.0]).all() or (values.sum(axis=1) > 1).any():
                raise ValueError(f"{field}: invalid one-hot group")
    return manifest, long


def metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (split, node, domain), group in frame.groupby(["split", "BYear", "domain"], sort=True):
        valid = group.loc[group.evaluable]
        error = valid.prediction - valid.observed
        if len(valid) >= 3 and valid.prediction.std(ddof=0) > 0:
            slope, intercept = np.polyfit(valid.prediction.to_numpy(dtype=float),
                                          valid.observed.to_numpy(dtype=float), 1)
        else:
            slope, intercept = np.nan, np.nan
        records.append({"split": split, "BYear": node, "domain": domain,
                        "n_predicted": len(group), "n_evaluable": len(valid),
                        "mae": error.abs().mean(), "rmse": np.sqrt(np.mean(error ** 2)),
                        "bias_pred_minus_obs": error.mean(),
                        "mean_prediction": valid.prediction.mean(), "mean_observed": valid.observed.mean(),
                        "calibration_intercept": intercept, "calibration_slope": slope,
                        "fraction_pred_outside_0_100": (~group.prediction.between(0, 100)).mean(),
                        "fraction_observed_at_100": valid.observed.eq(100).mean()})
    return pd.DataFrame(records)


def run(input_dir: Path, output_dir: Path, allow_development: bool = False) -> None:
    manifest, long = load_contract(input_dir, allow_development)
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_longs = [long]
    for item in manifest.get("imputation_draw_files", [])[1:]:
        other = pd.read_csv(input_dir / item["long"], dtype={ID: "string"})
        if not other[[ID, "BYear", "domain", "split"]].equals(
                long[[ID, "BYear", "domain", "split"]]):
            raise ValueError("imputation draw changed the patient-node grid")
        if not pd.to_numeric(other.observed, errors="coerce").equals(long.observed):
            raise ValueError("imputation draw changed observed outcomes")
        other["evaluable"] = other.evaluable.astype("string").str.lower().eq("true")
        draw_longs.append(other)
    if len(draw_longs) != manifest.get("mice_imputations", 1):
        raise ValueError("missing imputation draw")
    predictions, summaries, changes = [], [], []
    for domain in OUTCOMES:
        cols = manifest["domain_design_columns"][domain]
        names = ["const", "t", *cols]
        draw_predictions, draw_models, betas, covariances = [], [], [], []
        for draw, draw_long in enumerate(draw_longs):
            part = draw_long.loc[draw_long.domain.eq(domain)].copy()
            train = part.loc[part.split.eq("train") & part.evaluable].copy()
            if train[ID].nunique() < 3 or train.BYear.nunique() < 2:
                raise ValueError(f"{domain}: insufficient training patients/nodes")
            x_train = np.column_stack([np.ones(len(train)), train[["t", *cols]].to_numpy(dtype=float)])
            if np.linalg.matrix_rank(x_train) < x_train.shape[1]:
                raise ValueError(f"{domain}: selected design rank deficient in imputation {draw + 1}")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = sm.MixedLM(train.observed.to_numpy(dtype=float),
                                 pd.DataFrame(x_train, columns=names),
                                 groups=train[ID].to_numpy()).fit(reml=False, method="bfgs",
                                                                   maxiter=500, disp=False)
            random_var = float(np.asarray(fit.cov_re)[0, 0])
            residual_var = float(fit.scale)
            if not fit.converged or random_var <= 1e-6 or random_var / residual_var <= 1e-4:
                raise RuntimeError(f"{domain}: nonconvergence or boundary variance in imputation {draw + 1}")
            beta = np.asarray(fit.fe_params, dtype=float)
            betas.append(beta)
            covariances.append(np.asarray(fit.cov_params(), dtype=float)[:len(names), :len(names)])
            x_all = np.column_stack([np.ones(len(part)), part[["t", *cols]].to_numpy(dtype=float)])
            draw_predictions.append(x_all @ beta)
            draw_models.append({"domain": domain, "draw": draw, "converged": bool(fit.converged),
                                "fixed_effects": dict(zip(names, beta.tolist())),
                                "random_intercept_variance": random_var,
                                "residual_variance": residual_var,
                                "warnings": [str(w.message) for w in caught]})
            (output_dir / f"summary_{domain}_imputation_{draw + 1}.txt").write_text(
                str(fit.summary()), encoding="utf-8")
        part = long.loc[long.domain.eq(domain)].copy()
        part["prediction"] = np.mean(draw_predictions, axis=0)
        part["imputation_prediction_sd"] = (np.std(draw_predictions, axis=0, ddof=1)
                                           if len(draw_predictions) > 1 else np.nan)
        predictions.append(part[["split", ID, "BYear", "domain", "observed", "prediction",
                                 "imputation_prediction_sd", "evaluable", "death_timing", "has_form2"]])
        wide = part.pivot(index=["split", ID], columns="BYear", values="prediction")
        change = wide.reset_index()[["split", ID]].copy()
        change["domain"] = domain
        for first, second in zip(NODES[:-1], NODES[1:]):
            change[f"change_{first}_to_{second}"] = (wide[second] - wide[first]).to_numpy()
        changes.append(change)
        beta_mean = np.mean(betas, axis=0)
        within = np.mean(covariances, axis=0)
        between = np.cov(np.stack(betas).T) if len(betas) > 1 else np.zeros_like(within)
        total = within + (1 + 1 / len(betas)) * between
        entry = {"domain": domain, "formula": "observed ~ t + " + " + ".join(cols) + " + (1|UniID)",
                 "train_patients": int(train[ID].nunique()), "train_rows": len(train),
                 "converged": True,
                 "random_intercept_variance": float(np.mean([m["random_intercept_variance"] for m in draw_models])),
                 "residual_variance": float(np.mean([m["residual_variance"] for m in draw_models])),
                 "fixed_effects": dict(zip(names, beta_mean.tolist())),
                 "fixed_effects_se": dict(zip(names, np.sqrt(np.diag(total)).tolist())),
                 "imputation_models": draw_models}
        summaries.append(entry)
        (output_dir / f"summary_{domain}.txt").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics(prediction).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(changes, ignore_index=True).to_csv(output_dir / "predicted_changes.csv",
                                                  index=False, encoding="utf-8-sig")
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "contract_version": CONTRACT_VERSION, "upstream_manifest_sha256": _hash(input_dir / "feature_manifest.json"),
        "analysis_status": manifest["analysis_status"], "training_nodes": NODES,
        "baseline_designation": manifest.get("baseline_designation"),
        "screening_method_by_domain": manifest["screening_method_by_domain"],
        "screening_alpha_by_domain": manifest["screening_alpha_by_domain"],
        "mice_imputations": len(draw_longs),
        "evaluated_splits": ["train", "internal_validation"], "models": summaries,
        "prediction_rule": "fixed effects only; no new-patient outcome or random effect",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved four independent M0 fits to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-development", action="store_true")
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.allow_development)


if __name__ == "__main__":
    main()
