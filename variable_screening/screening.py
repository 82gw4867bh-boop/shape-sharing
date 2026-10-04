"""Patient-grouped penalized screening with fold-fitted preprocessing."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import GroupKFold

from .preprocess import BaselineProcessor
from .source import ID, OUTCOMES, SEED


DEFAULT_CONFIG = {
    "seed": SEED, "folds": 5, "cv_repeats": 1, "stability_subsamples": 20,
    "subsample_fraction": 0.8, "rare_min_count": 20,
    "include_missing_indicators": False,
    "l1_ratios": [0.5, 0.9], "alphas": [0.5],
    "primary_method": "lasso", "primary_alpha": 0.5,
    "selection_frequency_threshold": 0.6, "max_original_variables": None,
    "max_design_columns": None,
    "correlation_alert_threshold": 0.7,
    "coef_threshold": 0.1, "relative_group_threshold": 0.1,
    "max_iter": 2000, "tol": 0.001,
    "continuous_imputation": "mice", "mice_imputations": 5,
    "mice_max_iter": 3, "mice_nearest_features": 15,
}


def _processor(pool: pd.DataFrame, config: dict, frame: pd.DataFrame) -> BaselineProcessor:
    return BaselineProcessor(pool, config["rare_min_count"],
                             config["include_missing_indicators"],
                             config["continuous_imputation"], config["seed"],
                             config["mice_max_iter"], config["mice_nearest_features"]).fit(frame)


def _design(rows: pd.DataFrame, baseline: pd.DataFrame, processor: BaselineProcessor,
            draw: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformed = processor.transform(baseline, draw=draw)
    order = pd.Index(baseline[ID].astype(str))
    ix = order.get_indexer(rows[ID].astype(str))
    if (ix < 0).any():
        raise ValueError("outcome row has no baseline patient")
    x = transformed.screening[processor.columns].to_numpy(dtype=float)[ix]
    y = rows.observed.to_numpy(dtype=float)
    t = rows.t.to_numpy(dtype=float)
    ids = rows[ID].astype(str).to_numpy()
    count = pd.Series(ids).map(pd.Series(ids).value_counts()).to_numpy(dtype=float)
    w = 1.0 / count
    return x, y, t, ids, w


def _fit_adjusted(x: np.ndarray, y: np.ndarray, t: np.ndarray, w: np.ndarray,
                  alpha: float, l1_ratio: float, config: dict) -> tuple[ElasticNet | Lasso, np.ndarray, np.ndarray]:
    time = np.column_stack([np.ones(len(t)), t])
    weighted = time * np.sqrt(w)[:, None]
    target = np.column_stack([y, x]) * np.sqrt(w)[:, None]
    coefficients = np.linalg.lstsq(weighted, target, rcond=None)[0]
    residual_x = x - time @ coefficients[:, 1:]
    residual_y = y - time @ coefficients[:, 0]
    shared = dict(alpha=alpha, fit_intercept=False, max_iter=config["max_iter"],
                  tol=config["tol"], selection="random", random_state=config["seed"])
    model = Lasso(**shared) if l1_ratio == 1.0 else ElasticNet(l1_ratio=l1_ratio, **shared)
    model.fit(residual_x, residual_y, sample_weight=w)
    return model, coefficients[:, 0], coefficients[:, 1:]


def _predict(model: ElasticNet | Lasso, y_time: np.ndarray, x_time: np.ndarray,
             x: np.ndarray, t: np.ndarray) -> np.ndarray:
    time = np.column_stack([np.ones(len(t)), t])
    return time @ y_time + (x - time @ x_time) @ model.coef_


def _patient_rmse(observed: np.ndarray, prediction: np.ndarray, ids: np.ndarray) -> float:
    error = pd.DataFrame({ID: ids, "squared": (observed - prediction) ** 2})
    return float(np.sqrt(error.groupby(ID).squared.mean().mean()))


def _group_strengths(model: ElasticNet | Lasso, processor: BaselineProcessor) -> dict[str, float]:
    index = {col: i for i, col in enumerate(processor.columns)}
    return {field: max((abs(model.coef_[index[col]]) for col in cols), default=0.0)
            for field, cols in processor.groups.items()}


def _selected_groups(model: ElasticNet | Lasso, processor: BaselineProcessor,
                     config: dict) -> set[str]:
    strength = _group_strengths(model, processor)
    cutoff = max(config["coef_threshold"],
                 config["relative_group_threshold"] * max(strength.values(), default=0))
    return {field for field, value in strength.items() if value >= cutoff and value > 0}


def _assemble_fields(ranked_fields: list[str], processor: BaselineProcessor,
                     x: np.ndarray, t: np.ndarray, max_original: int | None,
                     max_columns: int | None) -> tuple[list[str], dict[str, list[str]],
                                                       dict[str, list[str]], dict[str, str], int, set[str]]:
    selected, used_indices, used_columns, rank_skipped = [], [], 0, set()
    selected_columns: dict[str, list[str]] = {}
    zero_variance_omitted: dict[str, list[str]] = {}
    exclusion_reason: dict[str, str] = {}
    column_index = {column: index for index, column in enumerate(processor.columns)}
    for field in ranked_fields:
        if max_original is not None and len(selected) >= max_original:
            exclusion_reason[field] = "raw_variable_limit"
            continue
        group = processor.groups[field]
        estimable = [column for column in group
                     if np.ptp(x[:, column_index[column]]) > 0]
        omitted = [column for column in group if column not in estimable]
        if not estimable:
            exclusion_reason[field] = "no_estimable_design_columns"
            continue
        if max_columns is not None and used_columns + len(estimable) > max_columns:
            exclusion_reason[field] = "design_column_limit"
            continue
        proposal = used_indices + [column_index[column] for column in estimable]
        fixed = np.column_stack([np.ones(len(t)), t, x[:, proposal]])
        if np.linalg.matrix_rank(fixed) < fixed.shape[1]:
            rank_skipped.add(field)
            exclusion_reason[field] = "rank_deficient_group"
            continue
        selected.append(field)
        selected_columns[field] = estimable
        if omitted:
            zero_variance_omitted[field] = omitted
        used_columns += len(estimable)
        used_indices = proposal
    return (selected, selected_columns, zero_variance_omitted, exclusion_reason,
            used_columns, rank_skipped)


def screen_domain(domain: str, clean_train: pd.DataFrame, outcomes: pd.DataFrame,
                  pool: pd.DataFrame, config: dict,
                  method: str = "elastic_net",
                  fixed_alpha: float | None = None) -> tuple[pd.DataFrame, dict]:
    rows = outcomes.loc[outcomes.domain.eq(domain) & outcomes.split.eq("train") & outcomes.evaluable].copy()
    if not rows.BYear.isin((1, 5, 10, 15)).all():
        raise ValueError("screening received a sealed time node")
    if rows[ID].nunique() < config["folds"]:
        raise ValueError(f"{domain}: too few patients for grouped folds")
    if method not in {"elastic_net", "lasso"}:
        raise ValueError(f"unsupported screening method: {method}")
    if fixed_alpha is not None and (method != "lasso" or not np.isfinite(fixed_alpha)
                                    or fixed_alpha <= 0):
        raise ValueError("fixed_alpha requires Lasso and a positive finite value")
    ratios = config["l1_ratios"] if method == "elastic_net" else [1.0]
    alphas = [fixed_alpha] if fixed_alpha is not None else config["alphas"]
    params = [(float(a), float(l)) for a in alphas for l in ratios]
    scores = {p: [] for p in params}
    patients = clean_train[ID].astype(str).to_numpy()
    for repeat in range(config["cv_repeats"]):
        splitter = GroupKFold(n_splits=config["folds"], shuffle=True,
                             random_state=config["seed"] + repeat)
        for tr_index, va_index in splitter.split(clean_train, groups=patients):
            tr_base = clean_train.iloc[tr_index]
            va_base = clean_train.iloc[va_index]
            tr = rows.loc[rows[ID].isin(tr_base[ID])]
            va = rows.loc[rows[ID].isin(va_base[ID])]
            if tr.empty or va.empty:
                raise ValueError(f"{domain}: empty patient-level validation fold")
            processor = _processor(pool, config, tr_base)
            for alpha, ratio in params:
                predictions = []
                for draw in range(config["mice_imputations"]):
                    x, y, t, _, w = _design(tr, tr_base, processor, draw)
                    xv, yv, tv, idv, _ = _design(va, va_base, processor, draw)
                    model, yt, xt = _fit_adjusted(x, y, t, w, alpha, ratio, config)
                    predictions.append(_predict(model, yt, xt, xv, tv))
                scores[(alpha, ratio)].append(_patient_rmse(yv, np.mean(predictions, axis=0), idv))
    ranked = sorted(params, key=lambda p: (np.mean(scores[p]), p[0], p[1]))
    best_alpha, best_ratio = ranked[0]
    full_processor = _processor(pool, config, clean_train)
    full_selected, full_strengths = set(), []
    for draw in range(config["mice_imputations"]):
        x, y, t, _, w = _design(rows, clean_train, full_processor, draw)
        full_model, _, _ = _fit_adjusted(x, y, t, w, best_alpha, best_ratio, config)
        full_selected.update(_selected_groups(full_model, full_processor, config))
        full_strengths.append(_group_strengths(full_model, full_processor))
    full_strength = {f: float(np.mean([v[f] for v in full_strengths])) for f in pool.raw_field}
    rng = np.random.default_rng(config["seed"] + list(OUTCOMES).index(domain))
    patient_ids = clean_train[ID].astype(str).to_numpy()
    count = {field: 0 for field in pool.raw_field}
    for _ in range(config["stability_subsamples"]):
        sample = set(rng.choice(patient_ids, size=max(10, int(len(patient_ids) *
                          config["subsample_fraction"])), replace=False))
        sub_base = clean_train.loc[clean_train[ID].isin(sample)]
        sub_rows = rows.loc[rows[ID].isin(sample)]
        proc = _processor(pool, config, sub_base)
        for draw in range(config["mice_imputations"]):
            sx, sy, st, _, sw = _design(sub_rows, sub_base, proc, draw)
            fit, _, _ = _fit_adjusted(sx, sy, st, sw, best_alpha, best_ratio, config)
            for field in _selected_groups(fit, proc, config):
                count[field] += 1
    n_sub = config["stability_subsamples"] * config["mice_imputations"]
    ranked_fields = sorted((f for f in count if count[f] / n_sub >= config["selection_frequency_threshold"]),
                           key=lambda f: (-count[f], -full_strength[f], f))
    (selected, selected_columns, zero_variance_omitted, exclusion_reason,
     used_columns, rank_skipped) = _assemble_fields(
        ranked_fields, full_processor, x, t, config["max_original_variables"],
        config["max_design_columns"])
    (uncapped, _, _, _, uncapped_count, uncapped_rank_skipped) = _assemble_fields(
        ranked_fields, full_processor, x, t, None, None)
    for field in pool.raw_field:
        if field not in selected and field not in exclusion_reason:
            exclusion_reason[field] = "below_stability_threshold"
    output = pd.DataFrame([{
        "domain": domain, "method": method, "alpha": best_alpha, "raw_field": f,
        "selection_frequency": count[f] / n_sub,
        "full_fit_max_abs_coefficient": full_strength[f],
        "selected_full_fit": f in full_selected, "decision": "selected" if f in selected else "not_selected",
        "selection_note": (exclusion_reason[f] if f not in selected else
                           "zero_variance_columns_omitted" if f in zero_variance_omitted else ""),
        "expanded_columns": "|".join(selected_columns[f] if f in selected else full_processor.groups[f]),
        "screening_status": "development_candidate_not_frozen",
    } for f in pool.raw_field])
    diagnostic = {"domain": domain, "method": method,
                  "n_train_patients": int(rows[ID].nunique()),
                  "n_train_outcome_rows": len(rows), "alpha": best_alpha, "l1_ratio": best_ratio,
                  "cv_patient_rmse": float(np.mean(scores[(best_alpha, best_ratio)])),
                  "cv_all_parameters": [{"alpha": a, "l1_ratio": l,
                                         "mean_patient_rmse": float(np.mean(scores[(a, l)]))}
                                        for a, l in params],
                  "selected_raw_fields": selected, "selected_design_columns": used_columns,
                  "rank_deficient_groups_skipped": sorted(rank_skipped),
                  "zero_variance_columns_omitted": zero_variance_omitted,
                  "n_above_stability_threshold": len(ranked_fields),
                  "uncapped_selected_raw_fields": uncapped,
                  "uncapped_selected_design_columns": uncapped_count,
                  "uncapped_rank_deficient_groups_skipped": sorted(uncapped_rank_skipped),
                  "exclusion_counts": {key: int(value) for key, value in
                                       pd.Series(exclusion_reason).value_counts().items()}}
    return output, diagnostic
