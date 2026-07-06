from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "balanced_acc": "higher",
    "auroc": "higher",
    "f1": "higher",
    "collapse": "lower",
    "positive_rate_error": "lower",
    "positive_rate": "none",
}


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing method name in {value!r}")
    return name, Path(path)


def resolve_result_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "catsa_loso_formal_results.csv",
        path / "wesad_loso_formal_results.csv",
        path / "galaxy_loso_formal_results.csv",
        path / "galaxy_watch_loso_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a LOSO result CSV in {path}")


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def load_deploy_frame(name: str, path: Path) -> pd.DataFrame:
    csv_path = resolve_result_csv(path)
    raw = pd.read_csv(csv_path)
    if "subject" not in raw.columns:
        raise ValueError(f"{csv_path} does not contain a subject column.")

    out = pd.DataFrame({"subject": raw["subject"].astype(str), "method": name, "source_csv": str(csv_path)})
    for metric in METRIC_DIRECTIONS:
        out[metric] = numeric_column(raw, f"deploy_watch_{metric}")
        if out[metric].isna().all() and f"watch_only_{metric}" in raw.columns:
            out[metric] = numeric_column(raw, f"watch_only_{metric}")
        if out[metric].isna().all() and metric in raw.columns:
            out[metric] = numeric_column(raw, metric)

    for metric in ("balanced_acc", "auroc", "f1", "collapse", "positive_rate_error", "positive_rate"):
        out[f"teacher_{metric}"] = numeric_column(raw, f"teacher_{metric}")
    return out


def try_wilcoxon(diff: np.ndarray, alternative: str) -> float:
    clean = diff[np.isfinite(diff)]
    if clean.size == 0:
        return float("nan")
    if np.allclose(clean, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(clean, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def summarize_pair(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = reference.merge(
        candidate,
        on="subject",
        suffixes=("_reference", "_candidate"),
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        if direction == "none":
            continue
        ref = pd.to_numeric(paired[f"{metric}_reference"], errors="coerce")
        cand = pd.to_numeric(paired[f"{metric}_candidate"], errors="coerce")
        valid = ref.notna() & cand.notna()
        if direction == "lower":
            signed = ref[valid] - cand[valid]
            alternative = "greater"
        else:
            signed = cand[valid] - ref[valid]
            alternative = "greater"
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "n": int(valid.sum()),
                "reference_mean": float(ref[valid].mean()) if valid.any() else float("nan"),
                "candidate_mean": float(cand[valid].mean()) if valid.any() else float("nan"),
                "candidate_minus_reference_mean": float((cand[valid] - ref[valid]).mean()) if valid.any() else float("nan"),
                "signed_candidate_better_mean": float(signed.mean()) if valid.any() else float("nan"),
                "candidate_wins": int((signed > 1e-12).sum()),
                "candidate_losses": int((signed < -1e-12).sum()),
                "ties": int((np.abs(signed) <= 1e-12).sum()),
                "wilcoxon_p_candidate_better": try_wilcoxon(signed.to_numpy(dtype=float), alternative=alternative),
                "wilcoxon_p_two_sided": try_wilcoxon(signed.to_numpy(dtype=float), alternative="two-sided"),
            }
        )
        paired[f"{metric}_delta_candidate_minus_reference"] = cand - ref
        paired[f"{metric}_signed_candidate_better"] = np.nan
        paired.loc[valid, f"{metric}_signed_candidate_better"] = signed
    return paired, pd.DataFrame(rows)


def summarize_teacher_gap(frame: pd.DataFrame, method_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        if direction == "none":
            continue
        deploy = pd.to_numeric(frame[metric], errors="coerce")
        teacher = pd.to_numeric(frame[f"teacher_{metric}"], errors="coerce")
        valid = deploy.notna() & teacher.notna()
        if direction == "lower":
            signed = deploy[valid] - teacher[valid]
        else:
            signed = teacher[valid] - deploy[valid]
        rows.append(
            {
                "method": method_name,
                "metric": metric,
                "direction": direction,
                "n": int(valid.sum()),
                "deploy_mean": float(deploy[valid].mean()) if valid.any() else float("nan"),
                "teacher_mean": float(teacher[valid].mean()) if valid.any() else float("nan"),
                "teacher_minus_deploy_mean": float((teacher[valid] - deploy[valid]).mean()) if valid.any() else float("nan"),
                "signed_teacher_better_mean": float(signed.mean()) if valid.any() else float("nan"),
                "teacher_wins": int((signed > 1e-12).sum()),
                "teacher_losses": int((signed < -1e-12).sum()),
                "ties": int((np.abs(signed) <= 1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_text_report(
    path: Path,
    reference_name: str,
    candidate_name: str,
    pair_summary: pd.DataFrame,
    teacher_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines = [
        f"reference={reference_name}",
        f"candidate={candidate_name}",
        "",
        "[paired deploy comparison]",
        pair_summary.to_string(index=False),
        "",
        "[teacher vs deploy within each run]",
        teacher_summary.to_string(index=False),
        "",
    ]
    if "auroc_delta_candidate_minus_reference" in paired.columns:
        worst = paired.sort_values("auroc_delta_candidate_minus_reference").head(10)
        best = paired.sort_values("auroc_delta_candidate_minus_reference", ascending=False).head(10)
        cols = [
            "subject",
            "auroc_reference",
            "auroc_candidate",
            "auroc_delta_candidate_minus_reference",
            "balanced_acc_delta_candidate_minus_reference",
            "f1_delta_candidate_minus_reference",
            "collapse_delta_candidate_minus_reference",
            "positive_rate_error_delta_candidate_minus_reference",
        ]
        cols = [col for col in cols if col in paired.columns]
        lines.extend(["[worst AUROC deltas]", worst[cols].to_string(index=False), ""])
        lines.extend(["[best AUROC deltas]", best[cols].to_string(index=False), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise diagnostics for CATSA/WESAD/Galaxy LOSO result CSVs.")
    parser.add_argument("--reference", type=str, required=True, help="NAME=DIR_OR_CSV, e.g. PureKD=artifacts/...")
    parser.add_argument("--candidate", type=str, required=True, help="NAME=DIR_OR_CSV, e.g. Ours=artifacts/...")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference_name, reference_path = parse_named_path(args.reference)
    candidate_name, candidate_path = parse_named_path(args.candidate)
    reference = load_deploy_frame(reference_name, reference_path)
    candidate = load_deploy_frame(candidate_name, candidate_path)

    paired, pair_summary = summarize_pair(reference, candidate)
    teacher_summary = pd.concat(
        [
            summarize_teacher_gap(reference, reference_name),
            summarize_teacher_gap(candidate, candidate_name),
        ],
        axis=0,
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "pairwise_subject_diagnostics.csv"
    summary_path = args.output_dir / "pairwise_metric_summary.csv"
    teacher_path = args.output_dir / "teacher_gap_summary.csv"
    report_path = args.output_dir / "pairwise_diagnostics.txt"

    paired.to_csv(paired_path, index=False)
    pair_summary.to_csv(summary_path, index=False)
    teacher_summary.to_csv(teacher_path, index=False)
    write_text_report(report_path, reference_name, candidate_name, pair_summary, teacher_summary, paired)

    print(pair_summary.to_string(index=False))
    print()
    print(f"Saved paired subject diagnostics to {paired_path}")
    print(f"Saved metric summary to {summary_path}")
    print(f"Saved teacher gap summary to {teacher_path}")
    print(f"Saved text report to {report_path}")


if __name__ == "__main__":
    main()
