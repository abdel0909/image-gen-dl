import os
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from tqdm import trange
from .utils import load_styles, ensure_dirs, now_tag, seed_everything, compose_prompt

MODEL_ID = "runwayml/stable-diffusion-v1-5"  # offen & stabil

def load_pipe(dtype=None, device=None):
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe

def generate(prompt: str, style: str="comic", steps:int|None=None,
             guidance_scale:float|None=None, seed:int|None=None,
             n:int=1, negative_prompt:str|None=None, out_dir="out"):
    ensure_dirs()
    styles = load_styles()
    cfg = styles[style]

    W, H = cfg["size"]
    if steps is None: steps = cfg["steps"]
    if guidance_scale is None: guidance_scale = cfg["guidance_scale"]
    if negative_prompt is None: negative_prompt = cfg.get("negative","")

    tag = now_tag()
    seed, g = seed_everything(seed)
    pipe = load_pipe()

    final_prompt = compose_prompt(prompt, cfg.get("suffix",""))
    paths = []
    for i in trange(n, desc="render"):
        img = pipe(
            final_prompt,
            negative_prompt=negative_prompt,
            width=W, height=H,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=g
        ).images[0]
        path = os.path.join(out_dir, f"{style}_{tag}_{i+1:02d}.png")
        img.save(path)
        paths.append(path)

    return paths, {"seed": seed, "prompt": final_prompt, "style": style, "steps": steps, "guidance": guidance_scale}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--style", default="comic", choices=["comic","instagram","kinderbuch"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    paths, meta = generate(
        prompt=args.prompt, style=args.style, steps=args.steps,
        guidance_scale=args.guidance, seed=args.seed, n=args.n, out_dir=args.out_dir
    )
    print(json.dumps({"paths": paths, "meta": meta}, ensure_ascii=False, indent=2))
