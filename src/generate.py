# src/generate.py
import os, time, json
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from .utils import load_styles, ensure_dir, now_tag

styles = load_styles()

def _build_pipe(model_id: str, dtype: torch.dtype, device: str, lora_path: str | None = None):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # LoRA optional laden
    if lora_path and os.path.exists(lora_path):
        try:
            pipe.load_lora_weights(lora_path)
            # Optional (schneller zur Laufzeit, kein Grad benötigt):
            if hasattr(pipe, "fuse_lora"):
                pipe.fuse_lora()
            print(f"✅ LoRA geladen: {lora_path}")
        except Exception as e:
            print(f"⚠️ Konnte LoRA nicht laden ({lora_path}): {e}")

    return pipe

def generate(prompt: str,
             style: str = "comic",
             steps: int | None = None,
             guidance_scale: float | None = None,
             seed: int = 0,
             n: int = 1,
             negative: str | None = None,
             lora_path: str | None = None):
    """Rendert n Bilder im angegebenen Stil. Optional mit LoRA."""
    cfg = styles[style]
    size = tuple(cfg["size"])
    steps = steps or cfg.get("steps", 30)
    guidance_scale = guidance_scale or cfg.get("guidance_scale", 7.5)
    negative = (negative if negative is not None else cfg.get("negative", "")) or ""
    model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = _build_pipe(model_id, dtype, device, lora_path=lora_path)

    if seed and seed > 0:
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = torch.Generator(device=device)

    outdir = "out/pages"
    ensure_dir(outdir)
    tag = now_tag()

    rendered = []
    meta = {
        "prompt": prompt,
        "negative": negative,
        "style": style,
        "size": size,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "n": n,
        "model_id": model_id,
        "lora_path": lora_path or ""
    }

    for i in range(n):
        img = pipe(
            prompt + " " + cfg.get("suffix", ""),
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=size[1], width=size[0],
            generator=g
        ).images[0]

        p = os.path.join(outdir, f"{tag}_{i:02d}.png")
        img.save(p)
        rendered.append(p)
        print(f"💾 Gespeichert: {p}")

    with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return rendered, meta
