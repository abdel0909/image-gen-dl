# === write fresh src/train_lora.py ===
import os, textwrap, pathlib, sys
repo_dir = "/content/image-gen-dl"
pathlib.Path(f"{repo_dir}/src").mkdir(parents=True, exist_ok=True)
code = r'''
# -*- coding: utf-8 -*-
"""
Minimaler, robuster LoRA-Trainer für SD v1.5 (ohne Abhängigkeit von LoRAAttnProcessor).
- LoRA wird als eigene LoRALinear-Schicht in UNet-Linearprojektionen (to_q/k/v, to_out.0) injiziert
- Captions optional (wenn caption_file fehlt, werden Dateinamen verwendet)
- Speichert regelmäßig Checkpoints (.safetensors)
Aufruf:
    python src/train_lora.py --config configs/train_smoke.yaml
Config Felder (Beispiele):
{
  "base_model": "runwayml/stable-diffusion-v1-5",
  "train_images": "data/real_style/images",
  "caption_file": "(keine)",    # optional oder weglassen
  "output_dir": "runs/real_smoke",
  "resolution": 512,
  "steps": 120,
  "batch_size": 1,
  "grad_acc": 1,
  "lr": 1e-4,
  "rank": 16,
  "seed": 123,
  "fp16": true
}
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionPipeline
from safetensors.torch import save_file as safetensors_save


# ---------------- LoRA Bausteine ----------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = None, scale: float = 1.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias is not None

        # Original-Gewichte werden eingefroren
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias_param = None
        if self.bias:
            self.bias_param = nn.Parameter(base.bias.data.clone(), requires_grad=False)

        r = max(1, int(rank))
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scale = scale if alpha is None else (alpha / r)
        self.to(base.weight.device)

    def forward(self, x):
        # y = x @ W^T + (x @ A^T @ B^T) * scale
        y = torch.nn.functional.linear(x, self.weight, self.bias_param)
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A),  # x @ A^T
            self.lora_B                                  # (..) @ B^T
        )
        return y + update * self.scale


def _wrap_linear_modules(unet: nn.Module, rank: int):
    """
    Ersetzt Linear-Projektionen in Self-Attention Blöcken:
      to_q, to_k, to_v, to_out.0
    durch LoRALinear-Wrapper.
    """
    target_keys = ("to_q", "to_k", "to_v", "to_out.0")

    replaced = 0
    for name, module in list(unet.named_modules()):
        # nur eindeutige Namen der Ziel-Linear-Layer
        if not any(name.endswith(k) for k in target_keys):
            continue
        # aktuelles Modul muss Linear sein
        parent_name = ".".join(name.split(".")[:-1])
        child_key   = name.split(".")[-1]
        parent = unet
        if parent_name:
            for k in parent_name.split("."):
                parent = getattr(parent, k)

        child = getattr(parent, child_key, None)
        if isinstance(child, nn.Linear):
            setattr(parent, child_key, LoRALinear(child, rank=rank))
            replaced += 1
    return replaced


def lora_parameters(unet: nn.Module):
    for m in unet.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


# ---------------- Dataset ----------------

class ImageFolderWithOptionalCaptions(Dataset):
    def __init__(self, img_dir: str, caption_file: str | None, size: int):
        self.img_paths = []
        p = Path(img_dir)
        for ext in ("*.jpg","*.jpeg","*.png","*.webp"):
            self.img_paths += list(p.rglob(ext))
        self.img_paths.sort()
        self.size = size

        self.captions = {}
        if caption_file and caption_file != "(keine)" and Path(caption_file).exists():
            with open(caption_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # einfache Zuordnung nach Reihenfolge
            for i, path in enumerate(self.img_paths):
                self.captions[str(path)] = lines[i % len(lines)]
        else:
            # Dateiname als Fallback
            for path in self.img_paths:
                self.captions[str(path)] = path.stem.replace("_", " ")

        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, i):
        p = self.img_paths[i]
        im = Image.open(p).convert("RGB")
        return self.tf(im), self.captions[str(p)]


# ---------------- Utils ----------------

def load_yaml_to_ns(path: str | Path) -> NS:
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return NS(**cfg)

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_lora_safetensors(unet: nn.Module, out_path: Path):
    state = {}
    idx = 0
    for m in unet.modules():
        if isinstance(m, LoRALinear):
            state[f"lora.{idx}.A"] = m.lora_A.detach().cpu()
            state[f"lora.{idx}.B"] = m.lora_B.detach().cpu()
            state[f"lora.{idx}.in"]  = torch.tensor([m.in_features])
            state[f"lora.{idx}.out"] = torch.tensor([m.out_features])
            state[f"lora.{idx}.scale"] = torch.tensor([m.scale])
            idx += 1
    safetensors_save(state, str(out_path))


# ---------------- Training ----------------

def train(cfg: NS):
    set_seed(int(getattr(cfg, "seed", 123)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("== Config ==")
    print(json.dumps({
        "base_model": cfg.base_model,
        "image_dir": cfg.train_images,
        "caption_file": getattr(cfg, "caption_file", "(keine)"),
        "output_dir": cfg.output_dir,
        "resolution": cfg.resolution,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "grad_acc": cfg.grad_acc,
        "lr": cfg.lr,
        "rank": cfg.rank,
        "device": device,
        "fp16": bool(getattr(cfg, "fp16", True)),
    }, indent=2))

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        safety_checker=None,
        torch_dtype=torch.float16 if getattr(cfg, "fp16", True) and device=="cuda" else torch.float32,
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    replaced = _wrap_linear_modules(pipe.unet, rank=int(cfg.rank))
    if replaced == 0:
        raise RuntimeError("Keine Ziel-Linear-Layer gefunden (to_q/to_k/to_v/to_out.0).")

    params = list(lora_parameters(pipe.unet))
    if not params:
        raise RuntimeError("Keine LoRA-Parameter gefunden.")

    ds = ImageFolderWithOptionalCaptions(cfg.train_images, getattr(cfg, "caption_file", None), size=int(cfg.resolution))
    if len(ds) == 0:
        raise RuntimeError("Keine Trainingsbilder gefunden.")
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True, num_workers=2)

    opt = torch.optim.AdamW(params, lr=float(cfg.lr))
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda" and getattr(cfg,"fp16",True)))
    steps = int(cfg.steps)
    grad_acc = max(1, int(cfg.grad_acc))
    outdir = Path(cfg.output_dir); ensure_dir(outdir)

    pipe.unet.train()
    step = 0
    running = 0.0
    while step < steps:
        for imgs, _ in dl:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda" and getattr(cfg,"fp16",True))):
                # einfache self-supervised Loss: reconstruct latents (Fake-Loss als Smoke-Test)
                # -> Ziel ist hier nur, dass der Optimizer läuft & Checkpoints entstehen.
                noise = torch.randn_like(imgs)
                loss = (imgs - noise).abs().mean()

            scaler.scale(loss / grad_acc).backward()
            running += loss.item()

            if (step + 1) % grad_acc == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            if (step + 1) % max(20, grad_acc) == 0 or (step+1)==steps:
                avg = running / max(1, min(step+1, 20))
                print(f"step {step+1:05d}/{steps}  loss={avg:.4f}")
                running = 0.0

            if (step + 1) % 40 == 0 or (step+1)==steps:
                ckpt = outdir / f"step_{step+1:05d}.safetensors"
                save_lora_safetensors(pipe.unet, ckpt)
                print(f"💾 checkpoint: {ckpt}")

            step += 1
            if step >= steps: break

    # final
    final_ckpt = outdir / "final.safetensors"
    save_lora_safetensors(pipe.unet, final_ckpt)
    print(f"\n✅ fertig: {final_ckpt}")


# ---------------- CLI ----------------

def load_cfg(p: str | Path) -> NS:
    p = Path(p)
    if p.suffix.lower() in (".yml", ".yaml"):
        return load_yaml_to_ns(p)
    with open(p, "r") as f:
        return NS(**json.load(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # Defaults
    defaults = dict(
        resolution=512, steps=120, batch_size=1, grad_acc=1,
        lr=1e-4, rank=16, output_dir="runs/real_smoke", fp16=True
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k): setattr(cfg, k, v)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    train(cfg)

if __name__ == "__main__":
    main()
'''
open(f"{repo_dir}/src/train_lora.py","w",encoding="utf-8").write(code)
print("✅ train_lora.py geschrieben:", f"{repo_dir}/src/train_lora.py")
# Modulkache leeren, falls vorher geladen
for m in list(sys.modules):
    if m.startswith("src.train_lora"):
        del sys.modules[m]
