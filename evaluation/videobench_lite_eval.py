from __future__ import annotations

"""Lightweight pairwise evaluation inspired by VBench for CogVideoX base-vs-LoRA comparison."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v3 as iio
import numpy as np
import imageio.v2 as iio_v2
from PIL import Image
import open_clip
import torch

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PromptProfile:
    prompt: str
    colors: list[str]
    multicolor: bool
    low_light: bool
    bright_light: bool
    warm_light: bool
    interaction_expected: bool
    motion_target: float


@dataclass
class VideoMetrics:
    frame_count: int
    fps: float
    duration_sec: float
    sharpness: float
    contrast: float
    brightness: float
    saturation: float
    warmness: float
    temporal_consistency: float
    motion_magnitude: float
    motion_smoothness: float
    skin_ratio: float
    color_ratios: dict[str, float]
    visual_score: float
    temporal_score: float
    prompt_score: float
    clip_score_mean: float
    clip_score_max: float
    total_score: float
    thumbnail: str


COLOR_SPECS = {
    "orange": ((8, 80, 40), (24, 255, 255)),
    "ginger": ((8, 80, 40), (24, 255, 255)),
    "brown": ((5, 60, 20), (22, 255, 180)),
    "grey": ((0, 0, 50), (180, 55, 210)),
    "gray": ((0, 0, 50), (180, 55, 210)),
    "silver": ((0, 0, 90), (180, 45, 230)),
    "white": ((0, 0, 170), (180, 60, 255)),
    "black": ((0, 0, 0), (180, 255, 55)),
    "cream": ((15, 10, 140), (35, 110, 255)),
}


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt_profile(prompt: str) -> PromptProfile:
    text = prompt.lower()
    colors = [color for color in COLOR_SPECS if color in text]
    multicolor = any(word in text for word in ("calico", "tortoiseshell", "bicolor", "black-and-white"))
    low_light = "low light" in text or "dark" in text
    bright_light = any(term in text for term in ("bright", "even lighting", "daylight"))
    warm_light = any(term in text for term in ("warm", "sunlight", "afternoon light", "natural light"))
    interaction_expected = "hand" in text or "person" in text or "stroked" in text or "scratches" in text

    low_motion_terms = ("sleeping", "resting", "lying", "stare", "steady gaze", "still", "reclining")
    medium_motion_terms = ("blink", "head tilt", "turn", "tracks", "grooming", "licking", "pet", "stroked", "scratches")
    high_motion_terms = ("rolling", "yawn", "raises one paw", "reacting", "follows the bubble")
    motion_target = 0.14
    if any(term in text for term in low_motion_terms):
        motion_target = 0.10
    if any(term in text for term in medium_motion_terms):
        motion_target = 0.18
    if any(term in text for term in high_motion_terms):
        motion_target = 0.28

    return PromptProfile(
        prompt=prompt,
        colors=colors,
        multicolor=multicolor,
        low_light=low_light,
        bright_light=bright_light,
        warm_light=warm_light,
        interaction_expected=interaction_expected,
        motion_target=motion_target,
    )


def iter_sample_indices(frame_count: int, samples: int = 8) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= samples:
        return list(range(frame_count))
    return sorted({int(round(i * (frame_count - 1) / (samples - 1))) for i in range(samples)})


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class ClipScorer:
    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()
        self.text_cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def score(self, frames_bgr: list[np.ndarray], prompt: str) -> tuple[float, float]:
        if not frames_bgr or not prompt:
            return 0.0, 0.0

        if prompt not in self.text_cache:
            tokens = self.tokenizer([prompt]).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_cache[prompt] = text_features
        text_features = self.text_cache[prompt]

        pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
        image_tensor = torch.stack([self.preprocess(frame) for frame in pil_frames]).to(self.device)
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(-1)
        sims = sims.detach().float().cpu().numpy()
        mean_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))
        mean_score = clip01((mean_sim + 1.0) / 2.0) * 100.0
        max_score = clip01((max_sim + 1.0) / 2.0) * 100.0
        return mean_score, max_score


def build_color_hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def mask_ratio(hsv_frame: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> float:
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower_np, upper_np)
    return float(mask.mean() / 255.0)


def estimate_color_ratios(frame_bgr: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ratios = {name: mask_ratio(hsv, *spec) for name, spec in COLOR_SPECS.items()}
    return ratios


def estimate_skin_ratio(frame_bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(mask.mean() / 255.0)


def expected_brightness(profile: PromptProfile) -> float:
    if profile.low_light:
        return 0.30
    if profile.bright_light:
        return 0.72
    return 0.58


def frame_scores(
    frames_bgr: list[np.ndarray],
    gray_frames: list[np.ndarray],
    profile: PromptProfile,
) -> tuple[dict[str, float], list[np.ndarray]]:
    if not frames_bgr:
        return {}, []

    resized = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in frames_bgr]
    gray_resized = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in resized]

    sharpness_vals = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in gray_resized]
    contrast_vals = [float(frame.std() / 255.0) for frame in gray_resized]
    brightness_vals = [float(frame.mean() / 255.0) for frame in gray_resized]

    sat_vals: list[float] = []
    warmness_vals: list[float] = []
    skin_vals: list[float] = []
    color_maps: list[dict[str, float]] = []
    hists: list[np.ndarray] = []
    for frame in resized:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_vals.append(float(hsv[:, :, 1].mean() / 255.0))
        b, g, r = cv2.split(frame.astype(np.float32) / 255.0)
        warmness_vals.append(float((r.mean() + 1e-6) / (b.mean() + 1e-6)))
        skin_vals.append(estimate_skin_ratio(frame))
        color_maps.append(estimate_color_ratios(frame))
        hists.append(build_color_hist(frame))

    hist_sims = [
        cosine_similarity(prev_hist, next_hist)
        for prev_hist, next_hist in zip(hists[:-1], hists[1:])
    ]
    temporal_consistency = float(np.mean(hist_sims)) if hist_sims else 0.0

    flow_means: list[float] = []
    for prev_gray, next_gray in zip(gray_resized[:-1], gray_resized[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        flow_means.append(float(np.mean(magnitude) / 20.0))
    motion_magnitude = float(np.mean(flow_means)) if flow_means else 0.0
    motion_std = float(np.std(flow_means)) if flow_means else 1.0
    motion_smoothness = 1.0 / (1.0 + motion_std / (motion_magnitude + 1e-4))

    color_ratios: dict[str, float] = {}
    for name in COLOR_SPECS:
        color_ratios[name] = float(np.mean([frame_map[name] for frame_map in color_maps]))

    brightness = float(np.mean(brightness_vals))
    contrast = float(np.mean(contrast_vals))
    saturation = float(np.mean(sat_vals))
    warmness = float(np.mean(warmness_vals))
    skin_ratio = float(np.mean(skin_vals))
    sharpness = float(np.mean(sharpness_vals))

    sharpness_score = clip01(math.log1p(sharpness) / 8.0)
    contrast_score = clip01(contrast / 0.22)
    saturation_score = clip01(saturation / 0.35)
    exposure_score = 1.0 - min(abs(brightness - expected_brightness(profile)) / 0.45, 1.0)
    visual_score = 100.0 * (
        0.40 * sharpness_score
        + 0.20 * contrast_score
        + 0.15 * saturation_score
        + 0.25 * exposure_score
    )

    temporal_score = 100.0 * (
        0.60 * clip01(temporal_consistency)
        + 0.40 * clip01(motion_smoothness)
    )

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "saturation": saturation,
        "warmness": warmness,
        "temporal_consistency": temporal_consistency,
        "motion_magnitude": motion_magnitude,
        "motion_smoothness": motion_smoothness,
        "skin_ratio": skin_ratio,
        "visual_score": visual_score,
        "temporal_score": temporal_score,
        "color_ratios": color_ratios,
    }, resized


def save_contact_sheet(sample_id: str, variant: str, frames_bgr: list[np.ndarray], thumb_dir: Path) -> str:
    thumb_dir.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        return ""
    tiles = frames_bgr[:4]
    while len(tiles) < 4:
        tiles.append(tiles[-1])
    tiles = [cv2.resize(tile, (240, 240), interpolation=cv2.INTER_AREA) for tile in tiles]
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    sheet = np.vstack([top, bottom])
    out_path = thumb_dir / f"{sample_id}_{variant}.jpg"
    cv2.imwrite(str(out_path), sheet)
    return str(out_path)


def analyze_video(
    path: Path,
    profile: PromptProfile,
    sample_id: str,
    variant: str,
    clip_scorer: ClipScorer,
    thumb_dir: Path,
) -> VideoMetrics:
    try:
        reader = iio_v2.get_reader(str(path), format="ffmpeg")
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 0.0)
        nframes = metadata.get("nframes")
        if isinstance(nframes, (int, float)) and math.isfinite(float(nframes)):
            frame_count = int(nframes)
        else:
            frame_count = int(reader.count_frames())
        sample_indices = iter_sample_indices(frame_count, samples=8)
        frames_rgb = [reader.get_data(index) for index in sample_indices]
        reader.close()
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames_rgb]
    except Exception:
        frames_bgr = []
        fps = 0.0
        frame_count = 0

    if not frames_bgr:
        return VideoMetrics(
            frame_count=0,
            fps=0.0,
            duration_sec=0.0,
            sharpness=0.0,
            contrast=0.0,
            brightness=0.0,
            saturation=0.0,
            warmness=0.0,
            temporal_consistency=0.0,
            motion_magnitude=0.0,
            motion_smoothness=0.0,
            skin_ratio=0.0,
            color_ratios={name: 0.0 for name in COLOR_SPECS},
            visual_score=0.0,
            temporal_score=0.0,
            prompt_score=0.0,
            clip_score_mean=0.0,
            clip_score_max=0.0,
            total_score=0.0,
            thumbnail="",
        )

    stats, sheet_frames = frame_scores(frames_bgr, [], profile)
    clip_score_mean, clip_score_max = clip_scorer.score(frames_bgr, profile.prompt)
    prompt_score = 0.7 * clip_score_mean + 0.3 * clip_score_max
    total_score = 0.35 * stats.get("visual_score", 0.0) + 0.25 * stats.get("temporal_score", 0.0) + 0.40 * prompt_score
    thumbnail = save_contact_sheet(sample_id, variant, sheet_frames, thumb_dir)
    duration = (frame_count / fps) if fps > 0 else 0.0
    return VideoMetrics(
        frame_count=frame_count,
        fps=fps,
        duration_sec=duration,
        sharpness=stats.get("sharpness", 0.0),
        contrast=stats.get("contrast", 0.0),
        brightness=stats.get("brightness", 0.0),
        saturation=stats.get("saturation", 0.0),
        warmness=stats.get("warmness", 0.0),
        temporal_consistency=stats.get("temporal_consistency", 0.0),
        motion_magnitude=stats.get("motion_magnitude", 0.0),
        motion_smoothness=stats.get("motion_smoothness", 0.0),
        skin_ratio=stats.get("skin_ratio", 0.0),
        color_ratios=stats.get("color_ratios", {name: 0.0 for name in COLOR_SPECS}),
        visual_score=stats.get("visual_score", 0.0),
        temporal_score=stats.get("temporal_score", 0.0),
        prompt_score=prompt_score,
        clip_score_mean=clip_score_mean,
        clip_score_max=clip_score_max,
        total_score=total_score,
        thumbnail=thumbnail,
    )


def render_html(rows: list[dict[str, str]], pair_rows: list[dict[str, str]], result_dir: Path) -> None:
    html_path = result_dir / "review.html"
    by_id: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_id.setdefault(row["id"], {})[row["variant"]] = row

    sections = []
    for pair in pair_rows:
        sample_id = pair["id"]
        base = by_id[sample_id]["base"]
        lora = by_id[sample_id]["lora"]
        sections.append(
            f"""
            <section class="pair">
              <div class="head">
                <div>
                  <h2>{sample_id}</h2>
                  <p>{pair["prompt"]}</p>
                </div>
                <div class="winner">winner: <strong>{pair["winner"]}</strong> | base {pair["base_total_score"]} vs lora {pair["lora_total_score"]}</div>
              </div>
              <div class="grid">
                <article>
                  <img src="{Path(base["thumbnail"]).resolve().as_uri()}" alt="{sample_id} base">
                  <div class="meta">base | total {base["total_score"]} | visual {base["visual_score"]} | temporal {base["temporal_score"]} | clip-prompt {base["prompt_score"]}</div>
                </article>
                <article>
                  <img src="{Path(lora["thumbnail"]).resolve().as_uri()}" alt="{sample_id} lora">
                  <div class="meta">lora | total {lora["total_score"]} | visual {lora["visual_score"]} | temporal {lora["temporal_score"]} | clip-prompt {lora["prompt_score"]}</div>
                </article>
              </div>
            </section>
            """
        )

    html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <title>Video-Bench-lite Review</title>
      <style>
        body {{
          margin: 0;
          font-family: "Avenir Next", "PingFang SC", sans-serif;
          background: linear-gradient(180deg, #f5f1ea, #efe7dc);
          color: #182126;
        }}
        main {{
          width: min(1320px, calc(100vw - 40px));
          margin: 28px auto 56px;
        }}
        h1 {{ margin: 0 0 8px; }}
        .intro {{ color: #5e6b73; margin-bottom: 24px; }}
        .pair {{
          background: rgba(255,255,255,0.84);
          border: 1px solid #ddcfbf;
          border-radius: 20px;
          padding: 18px;
          margin-bottom: 18px;
        }}
        .head {{
          display: flex;
          justify-content: space-between;
          gap: 20px;
          align-items: start;
          margin-bottom: 12px;
        }}
        .head h2 {{ margin: 0 0 6px; }}
        .head p {{ margin: 0; color: #5e6b73; max-width: 860px; line-height: 1.5; }}
        .winner {{ color: #5e6b73; white-space: nowrap; }}
        .winner strong {{ color: #b6562a; }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }}
        article {{
          background: #fffaf4;
          border: 1px solid rgba(24,33,38,0.08);
          border-radius: 16px;
          padding: 12px;
        }}
        img {{
          display: block;
          width: 100%;
          border-radius: 12px;
        }}
        .meta {{
          margin-top: 10px;
          color: #5e6b73;
          font-size: 14px;
        }}
      </style>
    </head>
    <body>
      <main>
        <h1>Video-Bench-lite Pair Review</h1>
        <p class="intro">这是一个面向当前目录结构的轻量版 Video-Bench 适配：保留 visual quality、temporal quality、prompt alignment 三类维度，支持 base/lora 成对比较。当前 prompt alignment 已升级为 CLIP 语义相似度。</p>
        {''.join(sections)}
      </main>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")


def render_markdown(report: dict[str, float | int | str], pair_rows: list[dict[str, str]], result_dir: Path) -> None:
    lines = [
        "# Video-Bench-lite Report",
        "",
        "## Overview",
        "",
        f"- Framework: `{report['framework']}`",
        f"- Pairs analyzed: `{report['pairs_analyzed']}`",
        f"- Base average total score: `{report['base_avg_total_score']}`",
        f"- Lora average total score: `{report['lora_avg_total_score']}`",
        f"- Base wins: `{report['base_wins']}`",
        f"- Lora wins: `{report['lora_wins']}`",
        "",
        "## Scoring",
        "",
        "- `visual_score`: sharpness + contrast + saturation + exposure",
        "- `temporal_score`: histogram consistency + optical-flow smoothness",
        "- `prompt_score`: CLIP image-text semantic similarity across sampled frames",
        "- `total_score = 0.35 * visual_score + 0.25 * temporal_score + 0.40 * prompt_score`",
        "",
        "## Pair Summary",
        "",
        "| id | winner | base total | lora total | base CLIP prompt | lora CLIP prompt |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['id']} | {row['winner']} | {row['base_total_score']} | {row['lora_total_score']} | {row['base_prompt_score']} | {row['lora_prompt_score']} |"
        )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight Video-Bench style comparison on base/lora outputs.")
    parser.add_argument("--compare_dir", default=str(ROOT / "compare_outputs"), help="Directory containing base/ and lora/.")
    parser.add_argument("--prompts_file", default=str(ROOT / "dataset" / "test_prompts.txt"), help="Prompt file matching generation order.")
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Directory for Video-Bench-lite outputs. Defaults to <compare_dir>/videobench_lite_results.",
    )
    args = parser.parse_args()

    compare_dir = Path(args.compare_dir)
    base_dir = compare_dir / "base"
    lora_dir = compare_dir / "lora"
    prompts_path = Path(args.prompts_file)
    result_dir = Path(args.result_dir) if args.result_dir else compare_dir / "videobench_lite_results"
    thumb_dir = result_dir / "contact_sheets"

    result_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(prompts_path)
    sample_ids = sorted(path.stem for path in base_dir.glob("*.mp4"))
    clip_scorer = ClipScorer()

    rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, str]] = []
    for index, sample_id in enumerate(sample_ids):
        prompt = prompts[index] if index < len(prompts) else ""
        profile = build_prompt_profile(prompt)
        base_path = base_dir / f"{sample_id}.mp4"
        lora_path = lora_dir / f"{sample_id}.mp4"
        if not lora_path.exists():
            continue

        base = analyze_video(base_path, profile, sample_id, "base", clip_scorer, thumb_dir)
        lora = analyze_video(lora_path, profile, sample_id, "lora", clip_scorer, thumb_dir)
        winner = "base" if base.total_score >= lora.total_score else "lora"

        for variant, metrics in (("base", base), ("lora", lora)):
            rows.append(
                {
                    "id": sample_id,
                    "variant": variant,
                    "total_score": f"{metrics.total_score:.2f}",
                    "visual_score": f"{metrics.visual_score:.2f}",
                    "temporal_score": f"{metrics.temporal_score:.2f}",
                    "prompt_score": f"{metrics.prompt_score:.2f}",
                    "clip_score_mean": f"{metrics.clip_score_mean:.2f}",
                    "clip_score_max": f"{metrics.clip_score_max:.2f}",
                    "sharpness": f"{metrics.sharpness:.2f}",
                    "contrast": f"{metrics.contrast:.4f}",
                    "brightness": f"{metrics.brightness:.4f}",
                    "saturation": f"{metrics.saturation:.4f}",
                    "warmness": f"{metrics.warmness:.4f}",
                    "temporal_consistency": f"{metrics.temporal_consistency:.4f}",
                    "motion_magnitude": f"{metrics.motion_magnitude:.4f}",
                    "motion_smoothness": f"{metrics.motion_smoothness:.4f}",
                    "skin_ratio": f"{metrics.skin_ratio:.4f}",
                    "fps": f"{metrics.fps:.3f}",
                    "duration_sec": f"{metrics.duration_sec:.3f}",
                    "frame_count": str(metrics.frame_count),
                    "thumbnail": metrics.thumbnail,
                    "prompt": prompt,
                }
            )

        pair_rows.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "base_total_score": f"{base.total_score:.2f}",
                "lora_total_score": f"{lora.total_score:.2f}",
                "base_prompt_score": f"{base.prompt_score:.2f}",
                "lora_prompt_score": f"{lora.prompt_score:.2f}",
                "winner": winner,
            }
        )

    if not rows:
        raise SystemExit(f"No valid base/lora pairs found in {compare_dir}")

    with (result_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (result_dir / "pair_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    base_scores = [float(row["base_total_score"]) for row in pair_rows]
    lora_scores = [float(row["lora_total_score"]) for row in pair_rows]
    report = {
        "framework": "Video-Bench-lite + CLIP",
        "pairs_analyzed": len(pair_rows),
        "base_avg_total_score": round(sum(base_scores) / len(base_scores), 2) if base_scores else 0.0,
        "lora_avg_total_score": round(sum(lora_scores) / len(lora_scores), 2) if lora_scores else 0.0,
        "base_wins": sum(1 for row in pair_rows if row["winner"] == "base"),
        "lora_wins": sum(1 for row in pair_rows if row["winner"] == "lora"),
    }
    (result_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(rows, pair_rows, result_dir)
    render_markdown(report, pair_rows, result_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
