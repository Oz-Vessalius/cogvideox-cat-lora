import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

BASE_MODEL = "/root/autodl-tmp/CogVideoX-5b"
LORA_PATH = "/root/diffusers/examples/cogvideo/cogvideox-lora-output"
OUTPUT_DIR = Path("/root/diffusers/examples/cogvideo/gradio_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("Loading base pipeline...")
base_pipe = CogVideoXPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype)
base_pipe.to("cuda")
base_pipe.vae.enable_slicing()
base_pipe.vae.enable_tiling()

print("Loading lora pipeline...")
lora_pipe = CogVideoXPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype)
lora_pipe.to("cuda")
lora_pipe.vae.enable_slicing()
lora_pipe.vae.enable_tiling()
lora_pipe.load_lora_weights(LORA_PATH, adapter_name="cogvideox-lora")
lora_pipe.set_adapters(["cogvideox-lora"], [1.0])


def generate_video(prompt, model_type, height, width, num_frames, steps, guidance_scale, fps, seed):
    pipe = lora_pipe if model_type == "LoRA" else base_pipe
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

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
    filename = OUTPUT_DIR / f"{model_type.lower()}_{abs(hash(prompt)) % 10**8}_{seed}.mp4"
    export_to_video(frames, str(filename), fps=int(fps))
    return str(filename)


demo = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Prompt", lines=4, value="A vertical close-up shot of an orange tabby cat resting on a soft indoor cushion in warm natural sunlight."),
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

demo.launch(server_name="0.0.0.0", server_port=7860)
