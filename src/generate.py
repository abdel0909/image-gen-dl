# src/generate.py
# Bild-Erzeugung mit Diffusers, Style-Config + optionaler Negativ-Prompt

import os
from typing import List, Tuple, Dict, Any, Optional

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# flexible Imports (lokal / modulstart)
try:
    from .utils import load_styles, ensure_outdir
except Exception:
    from src.utils import load_styles, ensure_outdir

# Ein globaler Pipeline-Cache, damit das Modell nicht jedes Mal neu geladen wird
_PIPELINE = None
_PIPELINE_MODEL_ID = None


def _load_pipeline(model_id: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    """Lädt (oder cached) die SD-Pipeline."""
    global _PIPELINE, _PIPELINE_MODEL_ID
    if _PIPELINE is not None and _PIPELINE_MODEL_ID == model_id:
        return _PIPELINE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    _PIPELINE = pipe
    _PIPELINE_MODEL_ID = model_id
    return pipe


def _size_from_style(style_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Liest Breite/Höhe aus style_cfg['size'] (z. B. [704, 512])."""
    size = style_cfg.get("size", [704, 512])
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return 704, 512


def generate(
    prompt: str,
    style: str,
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 0,
    n: int = 1,
    negative: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Erzeugt n Bilder und speichert sie als PNG unter out/pages/.
    Gibt (pfade, meta) zurück.
    """
    styles = load_styles()
    if style not in styles:
        raise ValueError(f"Unbekannter Stil '{style}'. Verfügbar: {list(styles.keys())}")

    style_cfg = styles[style]
    width, height = _size_from_style(style_cfg)

    # Style-Defaults fallbacken, falls UI sie nicht übergeben hat
    steps = int(steps or style_cfg.get("steps", 30))
    guidance_scale = float(guidance_scale or style_cfg.get("guidance_scale", 7.5))
    if negative is None:
        negative = style_cfg.get("negative", "")

    # Modell-ID wählen (kannst du in styles.json auch pro Stil hinterlegen, sonst global)
    model_id = style_cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = _load_pipeline(model_id, dtype=dtype)

    # RNG/Seed
    if seed and seed > 0:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    else:
        generator = torch.Generator(device=pipe.device)

    ensure_outdir("out/pages")
    paths: List[str] = []

    for i in range(n):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative or "",
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        img: Image.Image = out.images[0]
        path = os.path.join("out", "pages", f"page_{style}_{i:02d}.png")
        img.save(path)
        paths.append(path)

    meta = dict(
        prompt=prompt,
        negative=negative,
        style=style,
        steps=steps,
        guidance_scale=guidance_scale,
        size=[width, height],
        n=n,
        seed=seed,
        model_id=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return paths, meta
