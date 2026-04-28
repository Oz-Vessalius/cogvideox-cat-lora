from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evaluation CSV from prompt and output directories.")
    parser.add_argument("--prompts_file", required=True, help="Path to test_prompts.txt")
    parser.add_argument("--base_dir", required=True, help="Directory containing base videos, e.g. compare_outputs/base")
    parser.add_argument("--lora_dir", required=True, help="Directory containing lora videos, e.g. compare_outputs/lora")
    parser.add_argument("--output_csv", default="evaluation.csv", help="Output CSV path")
    args = parser.parse_args()

    prompts_file = Path(args.prompts_file)
    base_dir = Path(args.base_dir)
    lora_dir = Path(args.lora_dir)
    output_csv = Path(args.output_csv)

    prompts = read_prompts(prompts_file)

    rows = []
    for idx, prompt in enumerate(prompts, start=1):
        video_name = f"{idx:03d}.mp4"
        rows.append(
            {
                "id": f"{idx:03d}",
                "prompt": prompt,
                "base_video": str(base_dir / video_name),
                "lora_video": str(lora_dir / video_name),
                "subject_consistency_base": "",
                "subject_consistency_lora": "",
                "style_alignment_base": "",
                "style_alignment_lora": "",
                "clarity_base": "",
                "clarity_lora": "",
                "motion_naturalness_base": "",
                "motion_naturalness_lora": "",
                "structural_plausibility_base": "",
                "structural_plausibility_lora": "",
                "failure_tags_base": "",
                "failure_tags_lora": "",
                "comment": "",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "id",
        "prompt",
        "base_video",
        "lora_video",
        "subject_consistency_base",
        "subject_consistency_lora",
        "style_alignment_base",
        "style_alignment_lora",
        "clarity_base",
        "clarity_lora",
        "motion_naturalness_base",
        "motion_naturalness_lora",
        "structural_plausibility_base",
        "structural_plausibility_lora",
        "failure_tags_base",
        "failure_tags_lora",
        "comment",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()