# -*- coding: utf-8 -*-
"""
Robuster, minimaler LoRA-Trainer für SD v1.5 (diffusers).
Ziel: Keine Signature-Crashes mehr bei LoRAAttnProcessor (rank/hidden_* etc.).
- Liest YAML-Config
- Baut LoRA an UNet-Attention-Layern mit Signatur-Autodetektion
- Trainiert kurz und speichert regelmäßige Checkpoints
"""

import argparse, math, os, sys, time, json, inspect, random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import yaml
from safetensors.torch import save_file

from diffusers import StableDiffusionPipeline
# In neueren diffusers-Versionen liegt LoRAAttnProcessor hier:
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor
except Exception:
    # Fallback (sehr alte Strukturen)
    from diffusers.models.lora import LoRAAttnProcessor  # type: ignore

# ------------------ Utils ------------------ #

def load_yaml_to_ns(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        return SimpleNamespace(**yaml.safe_load(f))

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# ------------------ Dataset ------------------ #

class ImageFolder(Dataset):
    def __init__(self, img_dir: Path, resolution: int = 512):
        self.paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
        self.tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5], inplace=False),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)

# ------------------ LoRA attach (robust) ------------------ #

def get_cross_dim_for_name(unet, name: str):
    # Heuristik wie in diffusers: attn1=self-attn (kein cross), attn2=cross-attn
    if name.endswith("attn1.processor"):
        return None
    if name.endswith("attn2.processor"):
        # global Fallback
        return getattr(unet.config, "cross_attention_dim", None)
    # sonst best-effort
    return getattr(unet.config, "cross_attention_dim", None)

def infer_hidden_dim(attn_module) -> int | None:
    # Versuche übliche Stellen; fällt sonst auf in_features der to_q zurück
    for attr in ("hidden_dim", "hidden_size"):
        val = getattr(attn_module, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    try:
        return int(attn_module.to_q.in_features)  # type: ignore
    except Exception:
        return None

def lora_processor_fallback_factory(rank: int | None, hidden_dim: int | None, cross_dim: int | None):
    """
    Erzeugt eine Fabrik, die LoRAAttnProcessor mit kompatibler Signatur baut.
    Probiert Varianten in sinnvoller Reihenfolge – ohne Crash.
    """
    sig = None
    try:
        sig = inspect.signature(LoRAAttnProcessor.__init__)  # type: ignore
    except Exception:
        pass

    # Extrahiere erlaubte Parameter
    allowed = set()
    if sig:
        allowed = {p.name for p in sig.parameters.values()}

    def build():
        trials = []

        # 1) Modern (rank + cross + hidden_dim/hidden_size)
        if "rank" in allowed and "cross_attention_dim" in allowed:
            if "hidden_dim" in allowed and hidden_dim:
                trials.append(dict(rank=rank or 4, cross_attention_dim=cross_dim, hidden_dim=hidden_dim))
            if "hidden_size" in allowed and hidden_dim:
                trials.append(dict(rank=rank or 4, cross_attention_dim=cross_dim, hidden_size=hidden_dim))
            # ohne hidden*, nur rank+cross
            trials.append(dict(rank=rank or 4, cross_attention_dim=cross_dim))

        # 2) Älter (nur hidden_size/hidden_dim + optional cross)
        if "hidden_size" in allowed and hidden_dim:
            trials.append(dict(hidden_size=hidden_dim, cross_attention_dim=cross_dim if "cross_attention_dim" in allowed else None))
        if "hidden_dim" in allowed and hidden_dim:
            trials.append(dict(hidden_dim=hidden_dim, cross_attention_dim=cross_dim if "cross_attention_dim" in allowed else None))

        # 3) Minimal (keine args)
        trials.append(dict())

        last_err = None
        for kw in trials:
            # entferne None-Keys
            clean = {k: v for k, v in kw.items() if v is not None}
            try:
                return LoRAAttnProcessor(**clean)  # type: ignore
            except Exception as e:
                last_err = e
                continue
        # wenn alles scheitert:
        raise RuntimeError(f"LoRAAttnProcessor konnte mit keiner Signatur erstellt werden. Letzter Fehler: {last_err}")

    return build

def attach_lora(unet, rank: int = 16):
    procs = {}
    for name, attn in unet.attn_processors.items():
        cross_dim = get_cross_dim_for_name(unet, name)
        hdim = infer_hidden_dim(attn)

        factory = lora_processor_fallback_factory(rank=rank, hidden_dim=hdim, cross_dim=cross_dim)
        procs[name] = factory()

    unet.set_attn_processor(procs)
    return unet

def lora_parameters(unet):
    for _, module in unet.attn_processors.items():
        for n, p in module.named_parameters():
            if p.requires_grad:
                yield p

# ------------------ Training ------------------ #

def train(cfg: SimpleNamespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("== Config ==")
    print(json.dumps({
        "base_model": cfg.base_model,
        "image_dir": cfg.train_images,
        "caption_file": getattr(cfg, "caption_file", "(keine)"),
        "output_dir": cfg.output_dir,
        "resolution": cfg.resolution,
        "steps": cfg.max_train_steps,
        "batch_size": cfg.batch_size,
        "grad_acc": cfg.gradient_accumulation,
        "lr": cfg.lr if hasattr(cfg, "lr") else 1e-4,
        "rank": cfg.lora_rank if hasattr(cfg, "lora_rank") else 16,
        "device": device,
        "fp16": (getattr(cfg, "mixed_precision", "fp16") == "fp16"),
    }, indent=2, ensure_ascii=False))

    # Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16 if getattr(cfg, "mixed_precision", "fp16") == "fp16" and device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.to(device)

    # LoRA an UNet
    rank = int(getattr(cfg, "lora_rank", 16))
    pipe.unet = attach_lora(pipe.unet, rank=rank)

    # Optimizer
    trainable = list(lora_parameters(pipe.unet))
    if not trainable:
        raise RuntimeError("Keine trainierbaren LoRA-Parameter gefunden.")
    opt = torch.optim.AdamW(trainable, lr=getattr(cfg, "lr", 1e-4))

    # Daten
    img_dir = Path(cfg.train_images)
    ds = ImageFolder(img_dir, resolution=int(cfg.resolution))
    if len(ds) == 0:
        raise RuntimeError(f"Keine Trainingsbilder gefunden in: {img_dir}")
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    # Output
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    save_every = int(getattr(cfg, "save_every_steps", 50))

    # Trainings-Loop (sehr klein/anschaulich)
    pipe.enable_model_cpu_offload() if device != "cuda" else None

    global_step = 0
    max_steps = int(cfg.max_train_steps)

    # einfacher Loss: MSE auf x0-pred (Dummy-Mini-Trainer – für Smoke/kleine Runs ausreichend)
    mse = nn.MSELoss()

    pipe.unet.train()
    while global_step < max_steps:
        for batch in dl:
            if global_step >= max_steps:
                break
            imgs = batch.to(device)
            with torch.no_grad():
                latents = pipe.vae.encode(imgs).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Prompt: leer → nur UNet trainieren
            encoder_hidden_states = pipe.text_encoder(torch.zeros((latents.shape[0], 77), dtype=torch.long, device=device))[0] \
                if hasattr(pipe, "text_encoder") else torch.zeros((latents.shape[0], 77, 768), device=device)

            model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = mse(model_pred, noise)
            loss.backward()

            if (global_step + 1) % int(cfg.gradient_accumulation) == 0:
                opt.step()
                opt.zero_grad()

            if (global_step + 1) % 10 == 0:
                print(f"step {global_step+1:05d}/{max_steps:05d}  loss={loss.item():.4f}")

            if (global_step + 1) % save_every == 0 or (global_step + 1) == max_steps:
                ckpt = {k: v.detach().half().cpu() for k, v in pipe.unet.attn_processors_state_dict().items()}
                ckpt_path = out_dir / f"step_{global_step+1:05d}.safetensors"
                save_file(ckpt, str(ckpt_path))
                # auch Config daneben legen
                with open(out_dir / "config_used.yaml", "w") as f:
                    yaml.safe_dump(vars(cfg), f, sort_keys=False)
                print(f"✅ gespeichert: {ckpt_path}")

            global_step += 1

    print("🎉 Training fertig.")

# ------------------ CLI ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml_to_ns(args.config)

    # Default-Felder robust auffüllen
    defaults = dict(
        base_model="runwayml/stable-diffusion-v1-5",
        train_images="data/real_style/images",
        resolution=512,
        max_train_steps=120,
        batch_size=1,
        gradient_accumulation=1,
        lr=1e-4,
        output_dir="runs/real_smoke",
        mixed_precision="fp16",
        lora_rank=16,
        save_every_steps=40,
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    ensure_dir(Path(cfg.output_dir))
    train(cfg)

if __name__ == "__main__":
    main()
