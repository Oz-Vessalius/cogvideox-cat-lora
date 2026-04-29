# python compare_base_lora.py \
#   --base_model /root/autodl-tmp/CogVideoX-5b \
#   --lora_path /root/diffusers/examples/cogvideo/cogvideox-lora-output \
#   --prompts_file /root/diffusers/examples/cogvideo/test_prompts.txt \
#   --output_root /root/diffusers/examples/cogvideo/compare_outputs \
#   --height 960 \
#   --width 544 \
#   --num_frames 25 \
#   --lora_alpha 32 \
#   --lora_rank 32

import argparse
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def read_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def make_pipe(model_path: str, dtype: torch.dtype) -> CogVideoXPipeline:
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def generate_videos(
    pipe: CogVideoXPipeline,
    prompts: list[str],
    out_dir: Path,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    fps: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts, start=1):
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        frames = result.frames[0]
        video_path = out_dir / f"{idx:03d}.mp4"
        export_to_video(frames, str(video_path), fps=fps)
        print(f"saved: {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side base vs LoRA outputs for the same prompt set.")
    parser.add_argument("--base_model", required=True, help="Base CogVideoX model path")
    parser.add_argument("--lora_path", required=True, help="LoRA output dir or weights dir")
    parser.add_argument("--prompts_file", default="dataset/test_prompts.txt")
    parser.add_argument("--output_root", default="compare_outputs")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=544)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_rank", type=float, default=32.0)
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts_file))
    output_root = Path(args.output_root)
    base_dir = output_root / "base"
    lora_dir = output_root / "lora"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("loading base pipeline...")
    base_pipe = make_pipe(args.base_model, dtype)

    print("generating base videos...")
    generate_videos(
        pipe=base_pipe,
        prompts=prompts,
        out_dir=base_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
    )

    print("loading lora weights...")
    lora_pipe = make_pipe(args.base_model, dtype)
    lora_pipe.load_lora_weights(args.lora_path, adapter_name="cogvideox-lora")
    lora_scale = args.lora_alpha / args.lora_rank
    lora_pipe.set_adapters(["cogvideox-lora"], [lora_scale])

    print("generating lora videos...")
    generate_videos(
        pipe=lora_pipe,
        prompts=prompts,
        out_dir=lora_dir,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
    )

    print("done.")


if __name__ == "__main__":
    main()
