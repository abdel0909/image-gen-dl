# src/train_lora.py — minimaler, robuster LoRA-Trainer für SD v1.5 (Diffusers)
# Läuft mit: configs/train_smoke.yaml oder configs/train_real.yaml
import os, sys, math, json, time, argparse, socket, random, shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

import yaml

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor

# ----------------------------- Utils -----------------------------

def load_yaml_namespace(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return SimpleNamespace(**data)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def print_kv(title: str, d: dict):
    print(f"\n== {title} ==")
    for k, v in d.items():
        print(f"{k:>18}: {v}")

# ----------------------------- Dataset -----------------------------

class ImageCaptionFolder(Dataset):
    def __init__(self, img_dir: str, caption_file: str | None = None, size: int = 512):
        self.img_dir = Path(img_dir)
        self.size = size

        exts = (".png", ".jpg", ".jpeg", ".webp")
        self.paths = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in exts])

        self.captions = {}
        if caption_file and Path(caption_file).exists():
            with open(caption_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines()]
            # map sequentially by filename order
            for i, p in enumerate(self.paths):
                if i < len(lines) and lines[i]:
                    self.captions[p.name] = lines[i]

        self.tfm = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        im = self.tfm(im)
        cap = self.captions.get(p.name, "")
        return {"pixel_values": im, "text": cap}

def maybe_make_dummy_images(img_dir: Path, n=60, size=512):
    img_dir = ensure_dir(img_dir)
    exts = (".png", ".jpg", ".jpeg", ".webp")
    existing = [p for p in img_dir.glob("*") if p.suffix.lower() in exts]
    if len(existing) > 0:
        return len(existing)

    print("🧪 Kein Datensatz gefunden → erzeuge Dummy-Bilder …")
    rng = random.Random(123)
    for i in range(n):
        im = Image.new("RGB", (size, size), (rng.randint(0,255), rng.randint(0,255), rng.randint(0,255)))
        dr = ImageDraw.Draw(im)
        # einfache Formen, damit es "echter" wirkt
        for _ in range(6):
            x0, y0 = rng.randint(10, size-150), rng.randint(10, size-150)
            x1, y1 = x0 + rng.randint(60, 200), y0 + rng.randint(60, 200)
            dr.rectangle([x0, y0, x1, y1], outline=(0,0,0), width=4)
        im.save(img_dir / f"{i:04d}.png")
    return n

# ----------------------------- LoRA Attach -----------------------------

def attach_lora(unet, rank=16):
    """
    Hängt LoRA-Schichten an alle Attention-Module des UNet.
    Aktuelle diffusers erwartet 'hidden_dim' (nicht mehr 'hidden_size').
    """
    procs = {}
    for name, attn in unet.attn_processors.items():
        # attn1 = self-attn, kein cross_dim
        cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # neues Attribut in aktuellen Versionen:
        hidden_dim = getattr(attn, "hidden_size", None)
        if hidden_dim is None:
            # Fallback: viele Module haben 'to_q' mit in_features
            try:
                hidden_dim = attn.to_q.in_features  # best effort fallback
            except Exception:
                hidden_dim = unet.config.block_out_channels[-1]
        procs[name] = LoRAAttnProcessor(
            hidden_dim=hidden_dim,
            cross_attention_dim=cross_dim,
            rank=rank
        )
    unet.set_attn_processor(procs)
    return unet

def lora_parameters(unet):
    for _, module in unet.attn_processors.items():
        for name, p in module.named_parameters():
            if "lora_" in name and p.requires_grad:
                yield p

# ----------------------------- Training -----------------------------

def train(cfg_path: str):
    cfg = load_yaml_namespace(cfg_path)

    # Defaults
    base_model = getattr(cfg, "base_model", "runwayml/stable-diffusion-v1-5")
    image_dir  = getattr(cfg, "train_images", "data/real_style/images")
    caption    = getattr(cfg, "caption_file", None)  # darf fehlen
    out_dir    = getattr(cfg, "output_dir", "runs/real_lora")
    resolution = int(getattr(cfg, "resolution", 512))
    steps      = int(getattr(cfg, "max_train_steps", 800))
    bs         = int(getattr(cfg, "batch_size", 1))
    grad_acc   = int(getattr(cfg, "gradient_accumulation", 1))
    lr         = float(getattr(cfg, "lr", 1e-4))
    rank       = int(getattr(cfg, "lora_rank", 16))
    save_every = int(getattr(cfg, "save_every_steps", max(steps//5, 50)))
    seed       = int(getattr(cfg, "seed", 123))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    ensure_dir(out_dir)

    print_kv("Config", {
        "base_model": base_model, "image_dir": image_dir, "caption_file": caption or "(keine)",
        "output_dir": out_dir, "resolution": resolution, "steps": steps,
        "batch_size": bs, "grad_acc": grad_acc, "lr": lr, "rank": rank,
        "device": device
    })

    # Daten sicherstellen (Dummy, falls leer)
    img_dir = Path(image_dir)
    n_imgs = maybe_make_dummy_images(img_dir, n=60, size=resolution)
    print(f"✅ Trainingsbilder: {n_imgs}")

    # Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # LoRA anhängen
    attach_lora(pipe.unet, rank=rank)
    # Nur LoRA-Parameter trainieren
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(pipe.unet):
        p.requires_grad_(True)

    # Dataset & Loader
    ds = ImageCaptionFolder(image_dir, caption_file=caption, size=resolution)
    if len(ds) == 0:
        raise RuntimeError("Kein einziges Trainingsbild gefunden.")
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    # Optimizer
    opt = torch.optim.AdamW(lora_parameters(pipe.unet), lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    autocast = torch.cuda.amp.autocast if device=="cuda" else torch.autocast  # type: ignore

    # Text-Encoder & VAE im eval, nur UNet LoRA trainieren
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.unet.train()

    global_step = 0
    best_loss = float("inf")
    last_save = 0

    # Trainings-Schleife
    while global_step < steps:
        for batch in dl:
            if global_step >= steps:
                break

            images = batch["pixel_values"].to(device)
            prompts = batch["text"]

            # Encode latents
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                if prompts and any(prompts):
                    enc = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                else:
                    enc = pipe.tokenizer([""] * images.size(0), padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                cond = pipe.text_encoder(**enc).last_hidden_state

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device=device, dtype=torch.long)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample
                loss = nn.functional.mse_loss(model_pred, noise, reduction="mean") / grad_acc

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_acc == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                print(f"step {global_step:05d}/{steps}  loss={loss.item()*grad_acc:.4f}")

            # Speichern
            if global_step - last_save >= save_every or global_step == steps:
                ckpt_dir = ensure_dir(Path(out_dir) / f"step_{global_step:05d}")
                # nur LoRA-Gewichte speichern
                pipe.save_lora_weights(ckpt_dir)
                # Config-Kopie mitschreiben
                shutil.copy2(cfg_path, ckpt_dir / Path(cfg_path).name)
                print(f"💾 Checkpoint gespeichert: {ckpt_dir}")
                last_save = global_step

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    # best separat
                    best_dir = ensure_dir(Path(out_dir) / "best")
                    pipe.save_lora_weights(best_dir)
                    shutil.copy2(cfg_path, best_dir / Path(cfg_path).name)
                    print(f"⭐ Best aktualisiert (loss {best_loss*grad_acc:.4f})")

    # Abschluss
    final_dir = ensure_dir(Path(out_dir) / "final")
    pipe.save_lora_weights(final_dir)
    shutil.copy2(cfg_path, final_dir / Path(cfg_path).name)
    print(f"\n✅ Fertig. Finale LoRA-Gewichte: {final_dir}")

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Pfad zur YAML-Config")
    args = ap.parse_args()
    train(args.config)

if __name__ == "__main__":
    main()
