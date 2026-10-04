"""Fold-fitted, Form-I-only baseline processor with two numeric scales."""
from __future__ import annotations

import json
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .source import ID


@dataclass
class Transformed:
    screening: pd.DataFrame
    clinical: pd.DataFrame
    missing: pd.DataFrame


class BaselineProcessor:
    def __init__(self, pool: pd.DataFrame, rare_min_count: int = 20,
                 include_missing_indicators: bool = False,
                 continuous_imputation: str = "median", mice_seed: int = 20261003,
                 mice_max_iter: int = 3, mice_nearest_features: int = 15):
        self.pool = pool.reset_index(drop=True).copy()
        self.rare_min_count = rare_min_count
        self.include_missing_indicators = include_missing_indicators
        if continuous_imputation not in {"median", "mice"}:
            raise ValueError("unsupported continuous imputation")
        self.continuous_imputation = continuous_imputation
        self.mice_seed = mice_seed
        self.mice_max_iter = mice_max_iter
        self.mice_nearest_features = mice_nearest_features
        self.imputer: IterativeImputer | None = None
        self.parameters: dict = {}
        self.groups: dict[str, list[str]] = {}
        self.column_info: dict[str, dict] = {}
        self.columns: list[str] = []

    def _mice_matrix(self, frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
        columns = [(pd.to_numeric(frame[f], errors="raise").to_numpy(dtype=float)
                    - self.parameters[f]["mean"]) / self.parameters[f]["std"] for f in fields]
        if "ASex" in set(self.pool.raw_field):
            sex = frame["ASex"].astype("string")
            columns.extend([sex.eq(level).fillna(False).to_numpy(dtype=float)
                            for level in ("1", "2")])
        return np.column_stack(columns)

    def fit(self, frame: pd.DataFrame) -> "BaselineProcessor":
        if frame[ID].duplicated().any():
            raise ValueError("BaselineProcessor.fit requires one row per patient")
        self.parameters, self.groups, self.column_info, self.columns = {}, {}, {}, []
        for row in self.pool.itertuples():
            field = row.raw_field
            if row.predictor_type == "continuous":
                observed = pd.to_numeric(frame[field], errors="raise").dropna().astype(float)
                if observed.empty:
                    raise ValueError(f"{field}: all missing in training fold")
                mean = float(observed.mean())
                std = float(observed.std(ddof=0))
                if not np.isfinite(std) or std == 0:
                    raise ValueError(f"{field}: zero variance in training fold")
                # Lower observed median keeps integer clinical fields integer.
                ordered = np.sort(observed.to_numpy())
                median = float(ordered[(len(ordered) - 1) // 2])
                self.parameters[field] = {"type": "continuous", "mean": mean,
                                          "std": std, "impute_z": (median - mean) / std,
                                          "impute_original": median,
                                          "observed_min": float(observed.min()),
                                          "observed_max": float(observed.max()),
                                          "bound_min": None if pd.isna(row.bound_min) else float(row.bound_min),
                                          "bound_max": None if pd.isna(row.bound_max) else float(row.bound_max)}
                col = "x__" + field
                self.groups[field] = [col]
                self.column_info[col] = {"raw_field": field, "kind": "continuous",
                                         "unit": "source clinical unit"}
                self.columns.append(col)
                if self.include_missing_indicators:
                    indicator = "m__" + field
                    self.groups[field].append(indicator)
                    self.columns.append(indicator)
                    self.column_info[indicator] = {"raw_field": field, "kind": "missing_indicator"}
            else:
                values = frame[field].astype("string").fillna("__MISSING__")
                counts = values.value_counts()
                if counts.empty:
                    raise ValueError(f"{field}: empty training fold")
                rare = {str(x) for x, n in counts.items() if n < self.rare_min_count and x != "__MISSING__"}
                mapped = values.map(lambda x: "__OTHER__" if x in rare else str(x))
                present = sorted(mapped.unique().tolist())
                if len(present) < 2:
                    # A single constant category has no estimable main effect.
                    self.parameters[field] = {"type": "categorical", "constant": present[0],
                                              "legal_raw": row.legal_levels_train.split("|"),
                                              "rare_raw": sorted(rare), "reference": present[0],
                                              "levels": present}
                    self.groups[field] = []
                    continue
                nonmissing = mapped[mapped.ne("__MISSING__")]
                reference = (nonmissing.value_counts().sort_index().sort_values(ascending=False,
                             kind="stable").index[0] if len(nonmissing) else present[0])
                self.parameters[field] = {"type": "categorical", "legal_raw": row.legal_levels_train.split("|"),
                                          "rare_raw": sorted(rare), "reference": reference,
                                          "levels": present}
                cols = []
                for index, level in enumerate(present):
                    if level == reference:
                        continue
                    col = f"x__{field}__{index}"
                    cols.append(col)
                    self.columns.append(col)
                    self.column_info[col] = {"raw_field": field, "kind": "one_hot", "level": level,
                                             "reference": reference}
                self.groups[field] = cols
        if self.continuous_imputation == "mice":
            fields = [r.raw_field for r in self.pool.itertuples() if r.predictor_type == "continuous"]
            if fields:
                z = self._mice_matrix(frame, fields)
                self.imputer = IterativeImputer(sample_posterior=True, max_iter=self.mice_max_iter,
                                                n_nearest_features=min(self.mice_nearest_features, z.shape[1]),
                                                skip_complete=True, random_state=self.mice_seed)
                self.imputer.fit(z)
        return self

    def transform(self, frame: pd.DataFrame, draw: int = 0) -> Transformed:
        if not self.parameters:
            raise ValueError("BaselineProcessor is not fitted")
        screen: dict[str, object] = {ID: frame[ID].astype("string")}
        clinical: dict[str, object] = {ID: frame[ID].astype("string")}
        missing: dict[str, object] = {ID: frame[ID].astype("string")}
        imputed = {}
        if self.continuous_imputation == "mice":
            fields = [r.raw_field for r in self.pool.itertuples() if r.predictor_type == "continuous"]
            if fields:
                if self.imputer is None:
                    raise ValueError("fitted MICE imputer is required")
                z = self._mice_matrix(frame, fields)
                model = copy.deepcopy(self.imputer)
                model.random_state_ = np.random.RandomState(self.mice_seed + 1009 * draw + 17)
                completed = model.transform(z)
                if completed.shape != z.shape:
                    raise ValueError("MICE changed the continuous design width")
                imputed = dict(zip(fields, completed[:, :len(fields)].T))
        for row in self.pool.itertuples():
            field = row.raw_field
            p = self.parameters[field]
            is_missing = frame[field].isna()
            missing[field + "__raw_missing"] = is_missing.astype(int)
            if p["type"] == "continuous":
                values = pd.to_numeric(frame[field], errors="raise").astype(float)
                if p["bound_min"] is not None and values.dropna().lt(p["bound_min"]).any():
                    raise ValueError(f"{field}: below configured bound")
                if p["bound_max"] is not None and values.dropna().gt(p["bound_max"]).any():
                    raise ValueError(f"{field}: above configured bound")
                if self.continuous_imputation == "mice":
                    restored = pd.Series(imputed[field] * p["std"] + p["mean"], index=frame.index)
                    restored.loc[is_missing] = restored.loc[is_missing].clip(
                        lower=max(p["observed_min"], p["bound_min"] if p["bound_min"] is not None else -np.inf),
                        upper=min(p["observed_max"], p["bound_max"] if p["bound_max"] is not None else np.inf))
                    z = (restored - p["mean"]) / p["std"]
                else:
                    z = ((values - p["mean"]) / p["std"]).fillna(p["impute_z"])
                    restored = z * p["std"] + p["mean"]
                restored.loc[~is_missing] = values.loc[~is_missing]
                col = self.groups[field][0]
                screen[col] = z
                clinical[col] = restored
                if self.include_missing_indicators:
                    indicator = "m__" + field
                    screen[indicator] = is_missing.astype(int)
                    clinical[indicator] = is_missing.astype(int)
            else:
                values = frame[field].astype("string").fillna("__MISSING__")
                raw_legal = set(p["legal_raw"])
                unknown = values.ne("__MISSING__") & ~values.isin(raw_legal)
                if unknown.any():
                    raise ValueError(f"{field}: undeclared raw code {values[unknown].unique()[:8].tolist()}")
                rare = set(p["rare_raw"])
                mapped = values.map(lambda x: "__OTHER__" if x in rare else str(x))
                unseen = ~mapped.isin(p["levels"])
                if unseen.any():
                    # A dictionary-declared code unseen in this training fold has no
                    # estimable coefficient. Use OTHER when learned, otherwise its
                    # fixed reference category; undeclared raw codes already failed.
                    mapped.loc[unseen] = ("__OTHER__" if "__OTHER__" in p["levels"]
                                          else p["reference"])
                for col in self.groups[field]:
                    level = self.column_info[col]["level"]
                    dummy = mapped.eq(level).astype(int)
                    screen[col] = dummy
                    clinical[col] = dummy
        a = pd.DataFrame(screen, index=frame.index)[[ID, *self.columns]]
        b = pd.DataFrame(clinical, index=frame.index)[[ID, *self.columns]]
        m = pd.DataFrame(missing, index=frame.index)
        for label, matrix in (("screening", a), ("clinical", b)):
            values = matrix[self.columns].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{label} design has non-finite values")
        return Transformed(a, b, m)

    def to_dict(self) -> dict:
        return {"parameters": self.parameters, "groups": self.groups,
                "column_info": self.column_info, "columns": self.columns,
                "rare_min_count": self.rare_min_count,
                "include_missing_indicators": self.include_missing_indicators,
                "continuous_imputation": self.continuous_imputation,
                "mice_seed": self.mice_seed, "mice_max_iter": self.mice_max_iter,
                "mice_nearest_features": self.mice_nearest_features}

    @classmethod
    def from_dict(cls, pool: pd.DataFrame, obj: dict) -> "BaselineProcessor":
        instance = cls(pool, obj["rare_min_count"], obj["include_missing_indicators"],
                       obj.get("continuous_imputation", "median"), obj.get("mice_seed", 20261003),
                       obj.get("mice_max_iter", 3), obj.get("mice_nearest_features", 15))
        instance.parameters = obj["parameters"]
        instance.groups = obj["groups"]
        instance.column_info = obj["column_info"]
        instance.columns = obj["columns"]
        return instance
