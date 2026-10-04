"""Focused boundary checks; never reads the sealed temporal or 20-year outcomes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from m0.mixed_effects_m0_template import load_contract
from variable_screening import CONTRACT_VERSION
from variable_screening.pipeline import sha256
from variable_screening.preprocess import BaselineProcessor
from variable_screening.relations import all_pairs, pair_review
from variable_screening.screening import DEFAULT_CONFIG, screen_domain
from variable_screening.source import ID, OUTCOMES, cohort_and_splits, outcome_long
from variable_screening.eligibility import clean_candidates
from variable_screening.temporal_validation import prepare_processor, score_fixed_effects


class WorkflowChecks(unittest.TestCase):
    def test_temporal_documented_codes_use_frozen_other(self):
        pool = pd.DataFrame({"raw_field": ["AMarStIj", "ANCatDis"],
                             "predictor_type": ["categorical", "categorical"],
                             "special_codes": ["", ""], "legal_levels_train": ["1|2|3", "1|2|3"],
                             "bound_min": [np.nan, np.nan], "bound_max": [np.nan, np.nan]})
        train = pd.DataFrame({ID: ["a", "b", "c", "d"],
                              "AMarStIj": ["1", "1", "2", "3"],
                              "ANCatDis": ["1", "1", "2", "3"]})
        fitted = BaselineProcessor(pool, rare_min_count=2).fit(train)
        original = json.dumps(fitted.to_dict(), sort_keys=True)
        processor, extensions = prepare_processor(pool, fitted.to_dict(), set(pool.raw_field))
        self.assertEqual(json.dumps(fitted.to_dict(), sort_keys=True), original)
        self.assertEqual({(x["field"], x["code"]) for x in extensions},
                         {("AMarStIj", "7"), ("ANCatDis", "8")})
        validation = pd.DataFrame({ID: ["v"], "AMarStIj": ["7"], "ANCatDis": ["8"]})
        design = processor.transform(clean_candidates(validation, processor.pool)).clinical
        for field in pool.raw_field:
            other = [col for col in processor.groups[field]
                     if processor.column_info[col]["level"] == "__OTHER__"]
            self.assertEqual(len(other), 1)
            self.assertEqual(int(design.loc[0, other[0]]), 1)
        validation.loc[0, "AMarStIj"] = "9999"
        with self.assertRaises(ValueError):
            clean_candidates(validation, processor.pool)

    def test_temporal_score_uses_fixed_effects_and_excludes_death(self):
        rows = []
        for domain in OUTCOMES:
            for person in ("p1", "p2"):
                for node in (1, 5, 10, 15):
                    alive = person == "p1" or node == 1
                    rows.append({ID: person, "domain": domain, "BYear": node,
                                 "t": (node - 1) / 5, "split": "temporal_test_primary",
                                 "observed": 65. if person == "p1" and node == 1 else np.nan,
                                 "evaluable": person == "p1" and node == 1,
                                 "death_timing": "not_known_dead_by_node" if alive else "after_death"})
        long = pd.DataFrame(rows)
        clinical = pd.DataFrame({ID: ["p1", "p2"], "x__test": [2., 3.]})
        manifest = {"domain_design_columns": {d: ["x__test"] for d in OUTCOMES}}
        models = [{"domain": d, "converged": True,
                   "fixed_effects": {"const": 10., "t": 2., "x__test": 3.}} for d in OUTCOMES]
        scored = score_fixed_effects(long, clinical, manifest, models)
        self.assertEqual(float(scored.loc[scored[ID].eq("p1") & scored.BYear.eq(5), "prediction"].iloc[0]), 17.6)
        self.assertTrue(scored.loc[scored[ID].eq("p2") & scored.BYear.eq(5), "prediction"].isna().all())
        wrong = long.copy()
        wrong.loc[0, "BYear"] = 20
        with self.assertRaises(ValueError):
            score_fixed_effects(wrong, clinical, manifest, models)

    def test_patient_split_and_partial_outcome(self):
        f1 = pd.DataFrame({ID: [f"p{i:03d}" for i in range(30)],
                           "AInjDt": [1998] * 20 + [2002] * 10,
                           "AInjAge": ["30"] * 30})
        cohort, counts = cohort_and_splits(f1)
        self.assertEqual(counts["split_counts"]["train"], 16)
        self.assertEqual(counts["split_counts"]["internal_validation"], 4)
        self.assertEqual(counts["split_counts"]["temporal_test_primary"], 10)
        f2 = pd.DataFrame({ID: ["p000", "p001"], "BYear": [5, 5],
                           **{col: [999 if domain == "physical_independence" else 60, 70]
                              for domain, col in OUTCOMES.items()},
                           "BFolUpCt": ["1", "1"], "BLostRsn": [None, None],
                           "BIntCond": ["1", "1"]})
        rec = pd.DataFrame({ID: f1[ID], "SDthDt": [2003, None] + [None] * 28})
        grid = outcome_long(cohort, f2, rec)
        person = grid.loc[grid[ID].eq("p000") & grid.BYear.eq(5)]
        self.assertEqual(set(person.death_timing), {"same_year_uncertain"})
        self.assertEqual(int(person.evaluable.sum()), 0)
        other = grid.loc[grid[ID].eq("p001") & grid.BYear.eq(5)]
        self.assertEqual(int(other.evaluable.sum()), 4)
        self.assertEqual(len(grid), 30 * 4 * 4)

    def test_fold_fitted_inverse_scale_and_encoding(self):
        pool = pd.DataFrame([
            {"raw_field": "score", "predictor_type": "continuous", "bound_min": 0,
             "bound_max": 100, "legal_levels_train": ""},
            {"raw_field": "group", "predictor_type": "categorical", "bound_min": np.nan,
             "bound_max": np.nan, "legal_levels_train": "A|B"},
        ])
        train = pd.DataFrame({ID: ["a", "b", "c", "d"],
                              "score": [10.0, 20.0, 30.0, np.nan],
                              "group": ["A", "A", "B", None]})
        processor = BaselineProcessor(pool, rare_min_count=1).fit(train)
        transformed = processor.transform(train)
        self.assertAlmostEqual(transformed.clinical.loc[3, "x__score"], 20.0)
        self.assertAlmostEqual(transformed.screening.loc[3, "x__score"], 0.0)
        self.assertEqual(transformed.missing.loc[3, "score__raw_missing"], 1)
        self.assertEqual(float(transformed.clinical.loc[0, "x__score"]), 10.0)
        learned = json.dumps(processor.to_dict(), sort_keys=True)
        validation = pd.DataFrame({ID: ["v"], "score": [100.0], "group": ["B"]})
        processor.transform(validation)
        self.assertEqual(learned, json.dumps(processor.to_dict(), sort_keys=True))
        validation.loc[0, "group"] = "999"
        with self.assertRaises(ValueError):
            processor.transform(validation)

    def test_mice_uses_training_patients_and_preserves_observed_values(self):
        pool = pd.DataFrame({"raw_field": ["a", "b", "ASex"],
                             "predictor_type": ["continuous", "continuous", "categorical"],
                             "bound_min": [0, 0, np.nan], "bound_max": [100, 100, np.nan],
                             "legal_levels_train": ["", "", "1|2"]})
        train = pd.DataFrame({ID: [f"p{i}" for i in range(20)],
                              "a": np.arange(20, dtype=float) + 10,
                              "b": [float(2 * i + 5) if i % 3 else np.nan for i in range(20)],
                              "ASex": ["1" if i % 2 else "2" for i in range(20)]})
        processor = BaselineProcessor(pool, continuous_imputation="mice",
                                      mice_max_iter=2, mice_nearest_features=2).fit(train)
        training_mean = processor.parameters["a"]["mean"]
        self.assertEqual(processor.imputer.n_features_in_, 4)
        heldout = pd.DataFrame({ID: ["v1", "v2"], "a": [15., 100.],
                                "b": [np.nan, 25.], "ASex": ["2", "1"]})
        first = processor.transform(heldout, draw=0).clinical
        second = processor.transform(heldout, draw=1).clinical
        self.assertEqual(processor.parameters["a"]["mean"], training_mean)
        self.assertEqual(first.loc[1, "x__a"], 100.)
        self.assertEqual(first.loc[1, "x__b"], 25.)
        self.assertTrue(np.isfinite(first.loc[0, "x__b"]))
        self.assertNotEqual(first.loc[0, "x__b"], second.loc[0, "x__b"])
        self.assertEqual(first.loc[0, "x__b"], processor.transform(heldout, draw=0).clinical.loc[0, "x__b"])

    def test_same_type_pair_coverage_and_relation(self):
        pool = pd.DataFrame({"raw_field": ["a", "b", "c", "d"],
                             "predictor_type": ["continuous", "continuous",
                                                "categorical", "categorical"]})
        pairs = all_pairs(pool, "v1")
        self.assertEqual(set(map(tuple, pairs[["variable_a", "variable_b"]].to_numpy())),
                         {("a", "b"), ("c", "d")})
        prior = pd.DataFrame({"variable_a": ["c"], "variable_b": ["d"],
                              "relation_level": ["L4"], "evidence": ["reviewed clinical rationale"]})
        review, relation = pair_review(pairs, prior)
        self.assertEqual(len(review), 2)
        self.assertEqual(len(relation), 1)
        self.assertEqual(relation.relation_level.iloc[0], "L4")

    def test_unobserved_missing_level_does_not_drop_whole_field(self):
        ids = [f"p{i:02d}" for i in range(40)]
        groups = ["A" if i % 2 == 0 else "B" for i in range(30)] + [None] * 10
        base = pd.DataFrame({ID: ids, "group": groups})
        rows = pd.DataFrame([
            {ID: person, "domain": "physical_independence", "split": "train",
             "BYear": node, "t": (node - 1) / 5,
             "observed": (20 if group == "A" else 80) + node,
             "evaluable": group is not None}
            for person, group in zip(ids, groups) for node in (1, 5)
        ])
        pool = pd.DataFrame([{"raw_field": "group", "predictor_type": "categorical",
                              "legal_levels_train": "A|B", "bound_min": np.nan,
                              "bound_max": np.nan}])
        config = dict(DEFAULT_CONFIG, folds=2, stability_subsamples=4,
                      rare_min_count=1, alphas=[0.01], l1_ratios=[0.9],
                      coef_threshold=0.0, relative_group_threshold=0.0)
        selected, diagnostic = screen_domain("physical_independence", base, rows,
                                             pool, config)
        self.assertEqual(selected.decision.iloc[0], "selected")
        self.assertEqual(diagnostic["selected_design_columns"], 1)
        self.assertEqual(len(diagnostic["zero_variance_columns_omitted"]["group"]), 1)
        lasso, lasso_diagnostic = screen_domain("physical_independence", base, rows,
                                                pool, config, method="lasso", fixed_alpha=0.01)
        self.assertEqual(lasso.decision.iloc[0], "selected")
        self.assertEqual(lasso_diagnostic["method"], "lasso")
        self.assertEqual(lasso_diagnostic["alpha"], 0.01)
        self.assertEqual(float(lasso.alpha.iloc[0]), 0.01)
        self.assertIsNone(DEFAULT_CONFIG["max_original_variables"])
        self.assertIsNone(DEFAULT_CONFIG["max_design_columns"])
        self.assertEqual(DEFAULT_CONFIG["primary_alpha"], 0.5)
        self.assertEqual(DEFAULT_CONFIG["alphas"], [0.5])

    def test_m0_contract_rejects_tampering_and_sealed_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = ["a", "b", "c", "d"]
            split = pd.DataFrame({ID: ids, "AInjDt": [1998] * 4,
                                  "split": ["train", "train", "train", "internal_validation"],
                                  "rule_version": ["v1"] * 4})
            baseline = pd.DataFrame({ID: ids, "split": split.split, "x__score": [1., 2., 3., 4.]})
            rows = []
            for person, group in zip(ids, split.split):
                for node in (1, 5, 10, 15):
                    for domain in OUTCOMES:
                        rows.append({ID: person, "split": group, "BYear": node,
                                     "domain": domain, "t": (node - 1) / 5,
                                     "observed": 50., "evaluable": True,
                                     "death_timing": "not_known_dead_by_node",
                                     "has_form2": True, "x__score": float(ids.index(person) + 1)})
            long = pd.DataFrame(rows)
            selected = pd.DataFrame({"domain": list(OUTCOMES), "raw_field": ["score"] * 4,
                                     "method": ["lasso"] * 4, "alpha": [0.5] * 4,
                                     "decision": ["selected"] * 4})
            candidate = pd.DataFrame({"raw_field": ["score"]})
            for name, frame in (("patient_splits.csv", split), ("prepared_baseline.csv", baseline),
                                ("model_ready_long.csv", long),
                                ("selected_variables_by_domain.csv", selected),
                                ("candidate_variable_pool.csv", candidate)):
                frame.to_csv(root / name, index=False)
            manifest = {"contract_version": CONTRACT_VERSION,
                        "analysis_status": "development_not_frozen",
                        "screening_method_by_domain": {d: "lasso" for d in OUTCOMES},
                        "screening_alpha_by_domain": {d: 0.5 for d in OUTCOMES},
                        "domain_design_columns": {d: ["x__score"] for d in OUTCOMES},
                        "domain_selected_raw_fields": {d: ["score"] for d in OUTCOMES},
                        "groups": {"score": ["x__score"]},
                        "column_info": {"x__score": {"kind": "continuous"}},
                        "files_sha256": {name: sha256(root / name) for name in (
                            "patient_splits.csv", "prepared_baseline.csv", "model_ready_long.csv",
                            "selected_variables_by_domain.csv", "candidate_variable_pool.csv")}}
            (root / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow-development"):
                load_contract(root)
            _, loaded = load_contract(root, allow_development=True)
            self.assertEqual(len(loaded), len(long))
            manifest["screening_method_by_domain"]["mobility"] = "elastic_net"
            (root / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires Lasso"):
                load_contract(root, allow_development=True)
            manifest["screening_method_by_domain"]["mobility"] = "lasso"
            manifest["screening_alpha_by_domain"]["mobility"] = 1.0
            (root / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alpha=0.5"):
                load_contract(root, allow_development=True)
            manifest["screening_alpha_by_domain"]["mobility"] = 0.5
            selected.loc[selected.domain.eq("mobility"), "alpha"] = 1.0
            selected.to_csv(root / "selected_variables_by_domain.csv", index=False)
            manifest["files_sha256"]["selected_variables_by_domain.csv"] = sha256(
                root / "selected_variables_by_domain.csv")
            (root / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "screening alpha mismatch"):
                load_contract(root, allow_development=True)
            selected.loc[selected.domain.eq("mobility"), "alpha"] = 0.5
            selected.to_csv(root / "selected_variables_by_domain.csv", index=False)
            manifest["files_sha256"]["selected_variables_by_domain.csv"] = sha256(
                root / "selected_variables_by_domain.csv")
            long.loc[0, "BYear"] = 20
            long.to_csv(root / "model_ready_long.csv", index=False)
            manifest["files_sha256"]["model_ready_long.csv"] = sha256(root / "model_ready_long.csv")
            (root / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed temporal or 20-year"):
                load_contract(root, allow_development=True)


if __name__ == "__main__":
    unittest.main()
