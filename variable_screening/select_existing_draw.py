"""Select one saved MICE draw using internal validation, without refitting MICE."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .source import ID, OUTCOMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(source: Path, old_m0: Path, output: Path) -> None:
    manifest = json.loads((source / "feature_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((old_m0 / "run_metadata.json").read_text(encoding="utf-8"))
    if manifest.get("mice_imputations") != 5 or len(manifest.get("imputation_draw_files", [])) != 5:
        raise ValueError("source must be the existing five-draw MICE run")
    if metadata["upstream_manifest_sha256"] != sha256(source / "feature_manifest.json"):
        raise ValueError("existing M0 does not match the five-draw upstream artifacts")
    for name, expected in manifest["files_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    by_domain = {item["domain"]: item for item in metadata["models"]}
    if set(by_domain) != set(OUTCOMES):
        raise ValueError("the existing M0 does not cover all four domains")
    rows = []
    for draw, item in enumerate(manifest["imputation_draw_files"]):
        long = pd.read_csv(source / item["long"], dtype={ID: "string"})
        evaluable = long.evaluable.astype("string").str.lower().eq("true")
        for domain in OUTCOMES:
            data = long.loc[long.domain.eq(domain) & long.split.eq("internal_validation") & evaluable]
            names = ["const", "t", *manifest["domain_design_columns"][domain]]
            model = by_domain[domain]["imputation_models"][draw]
            if not model["converged"] or set(model["fixed_effects"]) != set(names):
                raise ValueError(f"{domain}, draw {draw}: invalid saved M0 coefficients")
            x = np.column_stack([np.ones(len(data)), data[names[1:]].to_numpy(dtype=float)])
            beta = np.asarray([model["fixed_effects"][name] for name in names])
            error = x @ beta - data.observed.to_numpy(dtype=float)
            rows.append({"source_draw_index": draw, "domain": domain,
                         "n_internal_evaluable": len(data), "rmse": float(np.sqrt(np.mean(error ** 2)))})
    comparison = pd.DataFrame(rows)
    ranking = comparison.groupby("source_draw_index", as_index=False).rmse.mean()
    ranking = ranking.rename(columns={"rmse": "mean_four_domain_internal_rmse"})
    comparison = comparison.merge(ranking, on="source_draw_index", validate="many_to_one")
    best = int(ranking.sort_values(["mean_four_domain_internal_rmse", "source_draw_index"]).iloc[0].source_draw_index)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing versions")
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file() or path.name == "feature_manifest.json":
            continue
        if path.name.startswith(("prepared_baseline_imputation_", "model_ready_long_imputation_")):
            continue
        shutil.copy2(path, output / path.name)
    selected = manifest["imputation_draw_files"][best]
    for key, canonical in (("baseline", "prepared_baseline.csv"), ("long", "model_ready_long.csv")):
        shutil.copy2(source / selected[key], output / canonical)
    comparison.to_csv(output / "draw_comparison_internal.csv", index=False, encoding="utf-8-sig")
    decision = {"selection_data": "1995-2000 internal_validation only",
                "selection_rule": "minimum equal-weight mean RMSE across four CHART domains; lower source index breaks ties",
                "selected_source_draw_index_zero_based": best,
                "selected_source_draw_number_one_based": best + 1,
                "screening_mice_imputations": 5, "final_mice_imputations": 1,
                "source_upstream_manifest_sha256": sha256(source / "feature_manifest.json"),
                "source_m0_metadata_sha256": sha256(old_m0 / "run_metadata.json"),
                "mice_refitted": False}
    (output / "draw_selection_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["analysis_status"] = "development_not_frozen"
    manifest["baseline_designation"] = "official_m0_reference_baseline_v2_selected_existing_draw"
    manifest["screening_mice_imputations"] = 5
    manifest["mice_imputations"] = 1
    manifest["selected_mice_draw_index"] = best
    manifest["imputation_draw_files"] = [{"draw": 0, "source_draw_index": best,
                                          "baseline": "prepared_baseline.csv", "long": "model_ready_long.csv"}]
    manifest["selection_revision"] = "existing_five_draw_screening_then_internal_validation_choice_of_one_saved_draw"
    manifest["source_five_draw_manifest_sha256"] = decision["source_upstream_manifest_sha256"]
    manifest["draw_selection_code_sha256"] = sha256(Path(__file__))
    manifest["files_sha256"] = {name: sha256(output / name) for name in manifest["files_sha256"]
                                if (output / name).exists() and not name.startswith((
                                    "prepared_baseline_imputation_", "model_ready_long_imputation_"))}
    for name in ("prepared_baseline.csv", "model_ready_long.csv", "draw_comparison_internal.csv",
                 "draw_selection_decision.json"):
        manifest["files_sha256"][name] = sha256(output / name)
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Selected existing MICE draw {best + 1} of 5 into {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("old_m0", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.old_m0, args.output)


if __name__ == "__main__":
    main()
