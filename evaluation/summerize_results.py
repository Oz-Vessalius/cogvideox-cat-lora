from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


NUMERIC_PAIRS = [
    ("subject_consistency_base", "subject_consistency_lora", "subject_consistency"),
    ("style_alignment_base", "style_alignment_lora", "style_alignment"),
    ("clarity_base", "clarity_lora", "clarity"),
    ("motion_naturalness_base", "motion_naturalness_lora", "motion_naturalness"),
    ("structural_plausibility_base", "structural_plausibility_lora", "structural_plausibility"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parse_tags(value: str) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    return [tag.strip() for tag in raw.replace(";", ",").split(",") if tag.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge human scores and auto metrics, then summarize results.")
    parser.add_argument("--evaluation_csv", default="evaluation.csv")
    parser.add_argument("--auto_metrics_csv", default="auto_metrics.csv")
    parser.add_argument("--output_md", default="summary.md")
    parser.add_argument("--output_json", default="summary.json")
    args = parser.parse_args()

    evaluation_rows = read_csv(Path(args.evaluation_csv))
    auto_rows = read_csv(Path(args.auto_metrics_csv))
    auto_by_id = {row["id"]: row for row in auto_rows}

    score_summary: dict[str, dict[str, float]] = {}
    failure_base = Counter()
    failure_lora = Counter()

    for base_col, lora_col, key in NUMERIC_PAIRS:
        base_values: list[float] = []
        lora_values: list[float] = []
        for row in evaluation_rows:
            base_val = parse_float(row.get(base_col, ""))
            lora_val = parse_float(row.get(lora_col, ""))
            if base_val is not None:
                base_values.append(base_val)
            if lora_val is not None:
                lora_values.append(lora_val)
        base_avg = average(base_values)
        lora_avg = average(lora_values)
        score_summary[key] = {
            "base_avg": round(base_avg, 4),
            "lora_avg": round(lora_avg, 4),
            "improvement": round(lora_avg - base_avg, 4),
        }

    for row in evaluation_rows:
        for tag in parse_tags(row.get("failure_tags_base", "")):
            failure_base[tag] += 1
        for tag in parse_tags(row.get("failure_tags_lora", "")):
            failure_lora[tag] += 1

    base_success = sum(int(row.get("base_exists", "0") or 0) for row in auto_rows)
    lora_success = sum(int(row.get("lora_exists", "0") or 0) for row in auto_rows)
    total = len(auto_rows)

    summary = {
        "total_samples": total,
        "base_success_rate": round(base_success / total, 4) if total else 0.0,
        "lora_success_rate": round(lora_success / total, 4) if total else 0.0,
        "scores": score_summary,
        "failure_tags_base": dict(failure_base),
        "failure_tags_lora": dict(failure_lora),
    }

    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Evaluation Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Base success rate: {summary['base_success_rate']:.2%}",
        f"- LoRA success rate: {summary['lora_success_rate']:.2%}",
        "",
        "## Score Comparison",
        "",
        "| Metric | Base Avg | LoRA Avg | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, values in score_summary.items():
        lines.append(
            f"| {key} | {values['base_avg']:.4f} | {values['lora_avg']:.4f} | {values['improvement']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Failure Tags",
            "",
            "### Base",
        ]
    )
    if failure_base:
        for tag, count in failure_base.most_common():
            lines.append(f"- `{tag}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "### LoRA"])
    if failure_lora:
        for tag, count in failure_lora.most_common():
            lines.append(f"- `{tag}`: {count}")
    else:
        lines.append("- None")

    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary markdown to {args.output_md}")
    print(f"Wrote summary json to {args.output_json}")


if __name__ == "__main__":
    main()