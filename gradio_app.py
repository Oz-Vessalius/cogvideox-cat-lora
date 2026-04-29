from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo to compare base and LoRA CogVideoX generation.")
    parser.add_argument("--base_model", required=True, help="Path or HF repo id for the base CogVideoX model.")
    parser.add_argument("--lora_path", required=True, help="Path to the LoRA output directory or weight folder.")
    parser.add_argument("--output_dir", default="gradio_outputs", help="Directory for generated mp4 files.")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA adapter scale used for inference.")
    parser.add_argument("--server_name", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port.")
    return parser.parse_args()


def load_pipe(model_path: str, dtype: torch.dtype) -> CogVideoXPipeline:
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("Loading base pipeline...")
    base_pipe = load_pipe(args.base_model, dtype)

    print("Loading LoRA pipeline...")
    lora_pipe = load_pipe(args.base_model, dtype)
    lora_pipe.load_lora_weights(args.lora_path, adapter_name="cogvideox-lora")
    lora_pipe.set_adapters(["cogvideox-lora"], [args.lora_scale])

    def generate_video(
        prompt: str,
        model_type: str,
        height: float,
        width: float,
        num_frames: float,
        steps: float,
        guidance_scale: float,
        fps: float,
        seed: float,
    ) -> str:
        pipe = lora_pipe if model_type == "LoRA" else base_pipe
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        frames = result.frames[0]
        filename = output_dir / f"{model_type.lower()}_{abs(hash((prompt, int(seed)))) % 10**8}_{int(seed)}.mp4"
        export_to_video(frames, str(filename), fps=int(fps))
        return str(filename)

    demo = gr.Interface(
        fn=generate_video,
        inputs=[
            gr.Textbox(
                label="Prompt",
                lines=4,
                value="A vertical close-up shot of an orange tabby cat resting on a soft indoor cushion in warm natural sunlight.",
            ),
            gr.Radio(["Base", "LoRA"], value="LoRA", label="Model"),
            gr.Number(value=960, label="Height"),
            gr.Number(value=544, label="Width"),
            gr.Number(value=25, label="Num Frames"),
            gr.Number(value=50, label="Inference Steps"),
            gr.Number(value=6.0, label="Guidance Scale"),
            gr.Number(value=8, label="FPS"),
            gr.Number(value=42, label="Seed"),
        ],
        outputs=gr.Video(label="Generated Video"),
        title="CogVideoX Base vs LoRA Demo",
        description="Generate cat videos with either the base CogVideoX model or your LoRA-finetuned model.",
    )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
