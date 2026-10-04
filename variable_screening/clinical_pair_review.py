"""Reproduce the definition-based review of the selected Form I predictor pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .relations import pair_review


SOURCE = "NSCISC 2021 public Form I data dictionary"
FIM = {
    "AFEatgDs": ("eating", 144, "self_care"),
    "AFGrmgDs": ("grooming", 145, "self_care"),
    "AFBdMgDs": ("bladder management", 152, "sphincter"),
    "AFMBCWDs": ("bed/chair transfer", 156, "transfer"),
    "AFMTShDs": ("tub/shower transfer", 160, "transfer"),
    "AFLWWcDs": ("walk/wheelchair locomotion", 163, "locomotion"),
    "AFLStrDs": ("stairs", 165, "locomotion"),
    "AFScorDs": ("13-item FIM motor total", 166, "total"),
}
ASIA = {field: (field[4:6], 238, "upper" if field[4] == "C" else "lower")
        for field in ("AASAC5DL", "AASAC6DL", "AASAC7DL", "AASAL2DL", "AASAL3DL")}
OTHER = {
    "AI2ADays": ("injury to first system admission, days", 58, "admission_time"),
    "AI2RhADa": ("injury to inpatient rehabilitation admission, days", 59, "admission_time"),
    "AHDaSyAc": ("system acute medical care stay, days", 60, "stay_time"),
    "AHDaSyRb": ("inpatient rehabilitation stay, days", 61, "stay_time"),
    "AInjAge": ("age at injury, years", 63, "age"),
    "ASex": ("sex", 64, "demographic"),
    "ATrmEtio": ("traumatic etiology", 69, "injury_cause"),
    "ABdMMDis": ("bladder management method at discharge", 82, "bladder_method"),
    "ARace": ("racial or ethnic group", 168, "social"),
    "AMarStIj": ("marital status at injury", 170, "social"),
    "AEducLvl": ("education at injury", 172, "social"),
    "APrimPay": ("primary payer", 210, "social"),
    "APrLvlSt": ("primary occupational/educational status", 213, "social"),
    "ANCatDis": ("neurologic impairment category at discharge", 257, "neurologic_category"),
}
DEFINITIONS = {**FIM, **ASIA, **OTHER}
ASIA_SOURCE = "https://asia-spinalinjury.org/isncsci-resources/"
SEX_BLADDER_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7614058/"
EDUCATION_WORK_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10070183/"
RACE_PAYER_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9754933/"


def classify(a: str, b: str) -> tuple[str, str]:
    if a not in DEFINITIONS or b not in DEFINITIONS:
        raise ValueError(f"unreviewed definition: {a}, {b}")
    left, right = DEFINITIONS[a], DEFINITIONS[b]
    basis = (f"{SOURCE}, PDF pp. {left[1]}, {right[1]}: "
             f"{a}={left[0]}; {b}={right[0]}.")
    if a in ASIA and b in ASIA:
        if left[2] == right[2]:
            return "L3", basis + " Same-limb segmental key-muscle strength, distinct myotomes and no single-field conversion. " + ASIA_SOURCE
        return "L4", basis + " Upper- and lower-limb motor strength are related neurological functions with different myotomes. " + ASIA_SOURCE
    if a in FIM and b in FIM:
        if "AFScorDs" in (a, b):
            return "L3", basis + " Motor total is calculated from 13 items; a single component cannot determine the total, so no pairwise conversion (dictionary p. 166)."
        if left[2] == right[2]:
            return "L3", basis + " Two distinct activities within the same FIM motor subdomain, each scored for assistance; no item-to-item conversion."
        return "L4", basis + " Different FIM motor activities, related through functional independence but measuring distinct tasks."
    if (a in ASIA and b in FIM) or (a in FIM and b in ASIA):
        return "L4", basis + " Neurological key-muscle strength and activity-level independence are clinically related but different constructs. " + ASIA_SOURCE
    if {a, b} == {"AI2ADays", "AI2RhADa"}:
        return "L3", basis + " Both measure days since injury but end at different admissions; the interval between admissions is not in either value alone."
    times = {"AI2ADays", "AI2RhADa", "AHDaSyAc", "AHDaSyRb"}
    if a in times and b in times:
        return "L4", basis + " Related care pathway intervals with different start/end definitions; no constant conversion."
    if (a in times and b in FIM) or (a in FIM and b in times):
        return "L4", basis + " Care timing/duration and discharge task performance are clinically related but distinct measures."
    if (a == "AInjAge" and b in FIM) or (b == "AInjAge" and a in FIM):
        return "L4", basis + " Age and functional independence may be associated, but age is not an activity score."
    if {a, b} == {"ASex", "ABdMMDis"}:
        return "L4", basis + " Sex and bladder management are distinct; an SCI cohort reports sex differences in discharge bladder methods. " + SEX_BLADDER_SOURCE
    if {a, b} == {"AEducLvl", "APrLvlSt"}:
        return "L4", basis + " Education and occupational/educational status are distinct social variables with documented labor-market associations after SCI. " + EDUCATION_WORK_SOURCE
    if {a, b} == {"ARace", "APrimPay"}:
        return "L4", basis + " Race/ethnicity and payer type are distinct social/administrative variables; SCI claims research documents related disparities. " + RACE_PAYER_SOURCE
    if {a, b} in ({"ANCatDis", "ATrmEtio"}, {"ANCatDis", "ABdMMDis"}):
        return "L4", basis + " Neurologic impairment category and injury cause or bladder care are clinically related but measure different constructs."
    return "无已知关系", basis + " Definition review found no direct duplicate, conversion, closely shared measurement, or specifically supported clinical link for this pair; this does not assert statistical independence."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(upstream: Path) -> None:
    pairs_path = upstream / "variable_pair_review.csv"
    signed_path = upstream / "variable_pair_review_signedoff_2026_10_04.csv"
    manifest_path = upstream / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mice_imputations") != 1 or manifest.get("baseline_designation") != "official_m0_reference_baseline_v2_selected_existing_draw":
        raise ValueError("clinical review requires the selected-existing-draw baseline v2")
    for name, expected in manifest["files_sha256"].items():
        if _sha256(upstream / name) != expected:
            raise ValueError(f"upstream hash mismatch before clinical review: {name}")
    pairs = pd.read_csv(pairs_path, dtype="string")
    required = ["variable_a", "variable_b", "predictor_type", "candidate_pool_version"]
    if not set(required).issubset(pairs.columns) or pairs.duplicated(["variable_a", "variable_b"]).any():
        raise ValueError("invalid review queue")
    decisions = [classify(str(a), str(b)) for a, b in zip(pairs.variable_a, pairs.variable_b)]
    review = pairs[required].copy()
    review["relation_level"] = [item[0] for item in decisions]
    review["evidence"] = [item[1] for item in decisions]
    review["review_basis"] = "definition_and_primary_clinical_sources_2026-10-04"
    expected_pairs = json.loads((upstream / "selection_counts.json").read_text(encoding="utf-8"))["n_same_type_pairs"]
    if review.relation_level.eq("待核查").any() or len(review) != expected_pairs:
        raise ValueError("the current selected-variable pair queue is not completely reviewed")
    review.to_csv(signed_path, index=False, encoding="utf-8-sig")
    adjudicated, relation = pair_review(pairs[required], review)
    adjudicated.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    relation.to_csv(upstream / "variable_relation_table.csv", index=False, encoding="utf-8-sig")
    counts_path = upstream / "pipeline_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    counts["relations"]["n_pending_clinical_review"] = 0
    counts["relations"]["n_reviewed_by_definition_and_primary_sources"] = len(review)
    counts_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["analysis_status"] = "frozen"
    manifest["clinical_relation_review"] = {
        "status": "definition_based_review_complete",
        "review_date": "2026-10-04",
        "reviewer": "Codex, with official definitions and cited primary clinical research",
        "independent_clinician_signoff": False,
        "n_pairs": len(review),
        "level_counts": {str(k): int(v) for k, v in review.relation_level.value_counts().items()},
        "review_file": signed_path.name,
    }
    for name in (pairs_path.name, signed_path.name, "variable_relation_table.csv", counts_path.name):
        manifest["files_sha256"][name] = _sha256(upstream / name)
    manifest["clinical_review_code_sha256"] = _sha256(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(review.relation_level.value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    args = parser.parse_args()
    run(args.upstream)


if __name__ == "__main__":
    main()
