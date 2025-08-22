# -*- coding: utf-8 -*-
"""
Minimaler, robuster LoRA-Trainer für Stable Diffusion v1.5 (Diffusers).
- Bildordner + optionale captions.txt (eine Zeile pro Bild: "filename.jpg\tdein prompt")
- SD v1.5 als Basismodell (oder kompatible v1.x Modelle)
- LoRA nur auf UNet-Attention, text encoder bleibt frozen
- Mixed Precision (fp16) optional, automatisches Speichern von Checkpoints

Config (YAML) Felder (Beispiel):
base_model: runwayml/stable-diffusion-v1-5
train_images: data/real_style/images
caption_file: null                # oder "data/real_style/captions.txt"
output_dir: runs/real_smoke
resolution: 512
max_train_steps: 120
batch_size: 1
gradient_accumulation: 1
lr: 1.0e-4
rank: 16
seed: 123
save_every_steps: 40
mixed_precision: fp16             # "fp16" oder "no"
prompt_prefix: ""                 # optionaler Prompt-Präfix, wenn keine Captions vorhanden
"""

import os
import math
import json
import time
import random
import argparse
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import yaml

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_cosine_schedule_with_warmup


# ========= Hilfsfunktionen =========

def load_yaml_to_ns(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ========= LoRA an UNet hängen (ohne hidden_size/hidden_dim) =========
def attach_lora(unet: nn.Module, rank: int = 16):
    """
    Hängt LoRA-Schichten an alle Attention-Module des UNet.
    Übergibt nur 'rank' und bei cross-attn 'cross_attention_dim' (wenn bekannt).
    Dadurch keine Konflikte mit hidden_size/hidden_dim.
    """
    procs = {}
    # neuere diffusers: unet.attn_processors dict -> Namen enthalten ...attn1/attn2.processor
    for name, attn in unet.attn_processors.items():
        is_self = name.endswith("attn1.processor")
        cross_dim = None
        if not is_self:
            # versuche cross-dim aus UNet-Config oder Attn-Modul
            cross_dim = getattr(unet.config, "cross_attention_dim", None)
            if cross_dim is None:
                cross_dim = getattr(attn, "cross_attention_dim", None)

        kwargs = {"rank": int(rank)}
        if cross_dim is not None:
            kwargs["cross_attention_dim"] = int(cross_dim)

        procs[name] = LoRAAttnProcessor(**kwargs)

    unet.set_attn_processor(procs)
    return unet


# ========= Dataset =========

class ImageCaptionFolder(Dataset):
    """
    Lädt Bilder aus einem Ordner. Optional captions.txt mit Zeilen:
      <filename>\t<prompt text>
    Wenn keine Caption gefunden -> fallback: prefix oder leeres Prompt.
    """
    def __init__(self, img_dir: Path, caption_file: Path | None, size: int = 512, prefix: str = ""):
        self.img_dir = Path(img_dir)
        self.size = int(size)
        self.prefix = prefix or ""

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        self.images = [p for p in self.img_dir.iterdir() if p.suffix.lower() in exts]
        self.images.sort()

        self.captions = {}
        if caption_file and Path(caption_file).exists():
            with open(caption_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        fname, cap = line.split("\t", 1)
                    elif "," in line:
                        # toleranter: CSV-artig
                        fname, cap = line.split(",", 1)
                    else:
                        # nur caption? dann gilt für alle -> selten sinnvoll
                        fname, cap = "", line
                    if fname:
                        self.captions[fname] = cap
                    else:
                        # globaler fallback
                        self.prefix = cap

        self.tf = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.tf(img)

        # Caption bestimmen
        cap = self.captions.get(img_path.name, "").strip()
        if not cap:
            cap = self.prefix
        # letzter Fallback: neutrales Prompt (damit Textencoder nicht leer ist)
        if not cap:
            cap = "a photo"

        return {
            "pixel_values": pixel_values,
            "caption": cap,
        }


# ========= Train =========

def train(cfg: SimpleNamespace):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = str(getattr(cfg, "mixed_precision", "no")).lower()
    use_fp16 = (mp == "fp16" and device == "cuda")

    set_seed(getattr(cfg, "seed", None))

    print("\n== Konfiguration ==")
    print(json.dumps({
        "base_model": cfg.base_model,
        "image_dir": cfg.train_images,
        "caption_file": getattr(cfg, "caption_file", None) or "(keine)",
        "output_dir": cfg.output_dir,
        "resolution": cfg.resolution,
        "steps": cfg.max_train_steps,
        "batch_size": cfg.batch_size,
        "grad_acc": cfg.gradient_accumulation,
        "lr": cfg.lr,
        "rank": cfg.rank,
        "device": device,
        "fp16": use_fp16,
    }, indent=2, ensure_ascii=False))

    img_dir = Path(cfg.train_images)
    ensure_dir(img_dir)
    ensure_dir(cfg.output_dir)

    # ===== Pipeline laden =====
    dtype = torch.float16 if use_fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.to(device)

    # Text-Encoder & VAE bleiben frozen
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # ===== LoRA an UNet hängen =====
    pipe.unet = attach_lora(pipe.unet, rank=int(cfg.rank))

    # nur LoRA-Parameter trainieren
    def lora_params(module):
        for n, p in module.named_parameters():
            if "lora" in n and p.requires_grad:
                yield p

    opt = torch.optim.AdamW(lora_params(pipe.unet), lr=float(cfg.lr))

    # ===== Dataset & Loader =====
    ds = ImageCaptionFolder(
        img_dir=img_dir,
        caption_file=getattr(cfg, "caption_file", None),
        size=int(cfg.resolution),
        prefix=getattr(cfg, "prompt_prefix", "")
    )
    if len(ds) == 0:
        raise RuntimeError(f"Keine Trainingsbilder in {img_dir} gefunden.")

    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=2, pin_memory=(device == "cuda"))

    # ===== Scheduler =====
    num_update_steps_per_epoch = math.ceil(len(dl) / int(cfg.gradient_accumulation))
    max_train_steps = int(cfg.max_train_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=max(1, max_train_steps // 20),
        num_training_steps=max_train_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ===== Trainings-Loop =====
    pipe.unet.train()

    global_step = 0
    save_every = int(getattr(cfg, "save_every_steps", 100))

    print("\n== Training startet ==")
    last_log = time.time()

    while global_step < max_train_steps:
        for batch in dl:
            global_step += 1
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            prompts = batch["caption"]

            # Text-Encoder
            text_inputs = pipe.tokenizer(
                list(prompts),
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = text_inputs.input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = pipe.text_encoder(input_ids)[0]

            # VAE -> Latents
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215

            # Noise + timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Vorwärts (UNet vorhersagt Rauschen)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Rückwärts
            scaler.scale(loss / int(cfg.gradient_accumulation)).backward()

            if global_step % int(cfg.gradient_accumulation) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                lr_scheduler.step()

            # Logging
            if time.time() - last_log > 5:
                print(f"step {global_step:05d}/{max_train_steps}  loss={loss.item():.4f}")
                last_log = time.time()

            # Checkpoint speichern
            if global_step % save_every == 0 or global_step == max_train_steps:
                ckpt_dir = Path(cfg.output_dir) / f"step_{global_step:05d}"
                ensure_dir(ckpt_dir)
                # nur LoRA (Attention Processors) sichern:
                pipe.unet.save_attn_procs(ckpt_dir)
                # Config mitschreiben
                with open(ckpt_dir / "train_config.json", "w", encoding="utf-8") as f:
                    json.dump(vars(cfg), f, indent=2, ensure_ascii=False)
                print(f"💾 Checkpoint gespeichert: {ckpt_dir}")

            if global_step >= max_train_steps:
                break

    print(f"\n✅ Training fertig in {human_time(time.time()-t0)}. Letzte Stufe: {global_step}.")


# ========= CLI =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml_to_ns(args.config)

    # Defaults setzen / absichern
    defaults = dict(
        caption_file=None,
        prompt_prefix="",
        resolution=512,
        max_train_steps=200,
        batch_size=1,
        gradient_accumulation=1,
        lr=1e-4,
        rank=16,
        save_every_steps=100,
        mixed_precision="fp16",
        seed=123
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k) or getattr(cfg, k) is None:
            setattr(cfg, k, v)

    ensure_dir(cfg.output_dir)
    train(cfg)

if __name__ == "__main__":
    main()
