from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stat_video(path: Path) -> dict[str, object]:
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "exists": int(exists),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3) if exists else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute lightweight automatic metrics from output directories.")
    parser.add_argument("--prompts_file", required=True, help="Path to test_prompts.txt")
    parser.add_argument("--base_dir", required=True, help="Directory containing base videos")
    parser.add_argument("--lora_dir", required=True, help="Directory containing lora videos")
    parser.add_argument("--output_csv", default="auto_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts_file))
    base_dir = Path(args.base_dir)
    lora_dir = Path(args.lora_dir)
    output_csv = Path(args.output_csv)

    rows = []
    for idx, prompt in enumerate(prompts, start=1):
        video_name = f"{idx:03d}.mp4"
        base_video = base_dir / video_name
        lora_video = lora_dir / video_name
        base_stats = stat_video(base_video)
        lora_stats = stat_video(lora_video)
        rows.append(
            {
                "id": f"{idx:03d}",
                "prompt": prompt,
                "base_video": str(base_video),
                "lora_video": str(lora_video),
                "base_exists": base_stats["exists"],
                "lora_exists": lora_stats["exists"],
                "base_size_mb": base_stats["size_mb"],
                "lora_size_mb": lora_stats["size_mb"],
                "base_size_bytes": base_stats["size_bytes"],
                "lora_size_bytes": lora_stats["size_bytes"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "id",
        "prompt",
        "base_video",
        "lora_video",
        "base_exists",
        "lora_exists",
        "base_size_mb",
        "lora_size_mb",
        "base_size_bytes",
        "lora_size_bytes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()