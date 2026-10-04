"""All same-type raw-variable pairs and train-only empirical associations."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PAIR_COLUMNS = ["variable_a", "variable_b", "predictor_type", "candidate_pool_version"]
LEVELS = {"L1", "L2", "L3", "L4", "无已知关系", "待核查"}


def all_pairs(pool: pd.DataFrame, version: str) -> pd.DataFrame:
    rows = []
    for kind in ("continuous", "categorical"):
        names = sorted(pool.loc[pool.predictor_type.eq(kind), "raw_field"].tolist())
        rows.extend((a, b, kind, version) for a, b in combinations(names, 2))
    result = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    p_c = int(pool.predictor_type.eq("continuous").sum())
    p_k = int(pool.predictor_type.eq("categorical").sum())
    expected = p_c * (p_c - 1) // 2 + p_k * (p_k - 1) // 2
    if len(result) != expected or result.duplicated(["variable_a", "variable_b"]).any():
        raise ValueError("same-type pair coverage is incomplete")
    return result


def pair_review(pairs: pd.DataFrame, prior: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    review = pairs.copy()
    review["relation_level"] = "待核查"
    review["evidence"] = "requires definition or clinical evidence review"
    if prior is not None and len(prior):
        needed = {"variable_a", "variable_b", "relation_level", "evidence"}
        if not needed.issubset(prior.columns) or prior.duplicated(["variable_a", "variable_b"]).any():
            raise ValueError("invalid prior pair review")
        if not set(prior.relation_level).issubset(LEVELS):
            raise ValueError("invalid pair relation level")
        if prior.evidence.isna().any() or prior.evidence.astype(str).str.strip().eq("").any():
            raise ValueError("pair relation level requires evidence")
        merged = review.merge(prior[list(needed)], on=["variable_a", "variable_b"], how="left",
                              suffixes=("", "_prior"), validate="one_to_one")
        if len(merged) != len(review) or len(prior) > len(review):
            raise ValueError("prior review references an ineligible pair")
        for col in ("relation_level", "evidence"):
            merged[col] = merged[col + "_prior"].fillna(merged[col])
            merged = merged.drop(columns=[col + "_prior"])
        review = merged
    if review.relation_level.eq("L1").any():
        dup = review.loc[review.relation_level.eq("L1"), ["variable_a", "variable_b"]]
        raise ValueError(f"L1 retained duplicates require eligibility revision: {dup.head().to_dict('records')}")
    relation = review.loc[review.relation_level.isin({"L2", "L3", "L4"})].copy()
    if set(map(tuple, relation[["variable_a", "variable_b"]].to_numpy())) != set(map(
            tuple, review.loc[review.relation_level.isin({"L1", "L2", "L3", "L4"}),
                              ["variable_a", "variable_b"]].to_numpy())):
        raise ValueError("relation table does not cover all L1–L4 pairs")
    return review, relation


def _cramers_v(left: pd.Series, right: pd.Series) -> tuple[float, int, str]:
    mask = left.notna() & right.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, n, "fewer than three comparable patients"
    a, ka = pd.factorize(left[mask], sort=True)
    b, kb = pd.factorize(right[mask], sort=True)
    if len(ka) < 2 or len(kb) < 2:
        return np.nan, n, "one variable is constant"
    table = np.bincount(a * len(kb) + b, minlength=len(ka) * len(kb)).reshape(len(ka), len(kb))
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(np.sum((table - expected) ** 2 / expected))
    return float(np.sqrt((chi2 / n) / min(len(ka) - 1, len(kb) - 1))), n, ""


def empirical_associations(pairs: pd.DataFrame, clean_train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in pairs.itertuples():
        a, b = pair.variable_a, pair.variable_b
        x, y = clean_train[a], clean_train[b]
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if pair.predictor_type == "categorical":
            value, _, reason = _cramers_v(x, y)
            rows.append({**pair._asdict(), "method": "Cramers V", "n_pair_observed": n,
                         "association_value": value, "not_computable_reason": reason})
        else:
            if n < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                for method in ("Pearson", "Spearman"):
                    rows.append({**pair._asdict(), "method": method, "n_pair_observed": n,
                                 "association_value": np.nan,
                                 "not_computable_reason": "fewer than three patients or constant variable"})
                continue
            xa, ya = x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
            for method, first, second in (("Pearson", xa, ya),
                                          ("Spearman", rankdata(xa), rankdata(ya))):
                rows.append({**pair._asdict(), "method": method, "n_pair_observed": n,
                             "association_value": float(np.corrcoef(first, second)[0, 1]),
                             "not_computable_reason": ""})
    return pd.DataFrame(rows, columns=[*PAIR_COLUMNS, "method", "n_pair_observed",
                                       "association_value", "not_computable_reason"])
