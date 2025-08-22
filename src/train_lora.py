# -*- coding: utf-8 -*-
"""
Minimaler, robuster LoRA-Trainer für SD v1.5 (Diffusers).
- caption_file optional
- robustes attach_lora (kompatibel zu verschiedenen diffusers-Versionen)
- Checkpoint-Speicher alle N Schritte

Aufruf:
  python src/train_lora.py --config configs/train_smoke.yaml
"""

import os, sys, math, json, time, argparse, random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import yaml

# ====================== Utilities ======================

def load_yaml_to_ns(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Defaults (falls in YAML nicht vorhanden)
    defaults = dict(
        base_model="runwayml/stable-diffusion-v1-5",
        train_images="data/real_style/images",
        caption_file=None,
        resolution=512,
        max_train_steps=120,
        batch_size=1,
        gradient_accumulation=1,
        lr=1e-4,
        lora_rank=16,
        output_dir="runs/real_lora",
        prompt_prefix="",
        negative_prompt="",
        save_every_steps=40,
        seed=123,
    )
    for k, v in defaults.items():
        data.setdefault(k, v)
    return SimpleNamespace(**data)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def print_kv(title, d: dict):
    print(f"\n== {title} ==")
    for k, v in d.items():
        print(f"{k:>14}: {v}")

# ====================== Dataset ======================

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir: str | Path, caption_file: str | None, size: int, prompt_prefix: str = ""):
        self.image_dir = Path(image_dir)
        self.prompt_prefix = (prompt_prefix or "").strip()
        self.images = []

        # Collect images
        if self.image_dir.is_dir():
            for p in sorted(self.image_dir.glob("*")):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.images.append(p)

        # Optional captions.txt (eine Zeile pro Bildname oder "prompt: ...")
        self.captions = {}
        if caption_file:
            cap_path = Path(caption_file)
            if cap_path.exists():
                with open(cap_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Formate unterstützen:
                        # "IMG_0001.jpg: prompt text"
                        # oder nur "prompt text" (dann global)
                        if ":" in line:
                            name, cap = line.split(":", 1)
                            self.captions[name.strip()] = cap.strip()
                        else:
                            # globale Fallback-Caption, wenn kein Bild-spezifischer Eintrag
                            self.captions["*"] = line.strip()

        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),                    # [0,1]
            transforms.Normalize([0.5], [0.5])        # [-1,1]
        ])

    def __len__(self):
        return len(self.images)

    def _caption_for(self, name: str):
        if name in self.captions:
            c = self.captions[name]
        else:
            c = self.captions.get("*", "")
        c = c or ""  # None -> ""
        if self.prompt_prefix:
            # z. B. "natural light, candid, instagram photo"
            if c:
                return f"{self.prompt_prefix}, {c}"
            return self.prompt_prefix
        return c

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        pixel = self.tf(img)
        caption = self._caption_for(img_path.name)
        return {
            "pixel_values": pixel,
            "prompt": caption,
        }

# ====================== Diffusers / SD v1.5 ======================

def import_lora_class():
    """
    Liefert (LoRAAttnProcessor, save_attn_procs_fct) robust für verschiedene diffusers-Versionen.
    """
    try:
        # Neuere diffusers
        from diffusers.models.attention_processor import LoRAAttnProcessor
        def saver(unet, out):
            # existiert in neueren diffusers
            try:
                unet.save_attn_procs(out, safe_serialization=True)
            except Exception:
                # Fallback: rohes state_dict
                sd = {}
                for n, m in unet.attn_processors.items():
                    if isinstance(m, LoRAAttnProcessor):
                        for k, v in m.state_dict().items():
                            sd[f"{n}.{k}"] = v
                try:
                    from safetensors.torch import save_file
                    save_file(sd, str(Path(out) / "lora.safetensors"))
                except Exception:
                    torch.save(sd, str(Path(out) / "lora.bin"))
        return LoRAAttnProcessor, saver
    except Exception:
        # Ältere diffusers
        from diffusers.models.lora import LoRAAttnProcessor
        def saver(unet, out):
            # kein save_attn_procs in alten Versionen
            sd = {}
            for n, m in unet.attn_processors.items():
                if isinstance(m, LoRAAttnProcessor):
                    for k, v in m.state_dict().items():
                        sd[f"{n}.{k}"] = v
            try:
                from safetensors.torch import save_file
                save_file(sd, str(Path(out) / "lora.safetensors"))
            except Exception:
                torch.save(sd, str(Path(out) / "lora.bin"))
        return LoRAAttnProcessor, saver

def load_pipeline(base_model: str, device: str = "cuda"):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    # Safety Checker deaktivieren (Training)
    pipe.safety_checker = None
    return pipe

# ====================== LoRA robust anhängen ======================

def _hidden_from_attn(attn) -> int | None:
    # Versuche verschiedene Felder/Module
    for attr in ("to_q", "to_k", "to_v", "to_out"):
        mod = getattr(attn, attr, None)
        if mod is not None and hasattr(mod, "in_features"):
            return int(mod.in_features)
    # Alternativ: head_dim * heads
    if hasattr(attn, "head_dim") and hasattr(attn, "heads"):
        try:
            return int(attn.head_dim) * int(attn.heads)
        except Exception:
            pass
    # Letzer Fallback: None
    return None

def attach_lora(unet, rank: int):
    """
    Hängt LoRA an alle Attention-Module.
    Kompatibel zu diffusers-Versionen mit hidden_dim ODER hidden_size ODER nur rank.
    """
    LoRAAttnProcessor, _ = import_lora_class()

    procs = {}
    for name, attn in unet.attn_processors.items():
        # self-attention hat keine Cross-Dim
        cross_dim = None if name.endswith("attn1.processor") else getattr(unet.config, "cross_attention_dim", None)
        hidden = _hidden_from_attn(attn)

        # Konstruiere kwargs robust
        tried = []
        for kwargs in (
            {"hidden_dim": hidden, "cross_attention_dim": cross_dim, "rank": rank},
            {"hidden_size": hidden, "cross_attention_dim": cross_dim, "rank": rank},
            {"rank": rank},  # minimal
        ):
            try:
                procs[name] = LoRAAttnProcessor(**{k: v for k, v in kwargs.items() if v is not None})
                break
            except TypeError as e:
                tried.append(str(e))
        else:
            # Wenn alle Versuche scheitern, klare Meldung
            msg = "\n".join(tried)
            raise RuntimeError(f"LoRAAttnProcessor konnte nicht erstellt werden für '{name}'. Versuche:\n{msg}")

    unet.set_attn_processor(procs)

def lora_parameters(unet):
    for _, module in unet.attn_processors.items():
        # Nur LoRA-Parameter trainieren
        for n, p in module.named_parameters():
            if "lora_" in n and p.requires_grad:
                yield p

# ====================== Text + Latents ======================

def encode_text(pipe, prompts: list[str]):
    tok = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(pipe.device)
    with torch.no_grad():
        enc = pipe.text_encoder(input_ids)[0]
    return enc

@torch.no_grad()
def encode_images_to_latents(pipe, pixels: torch.Tensor):
    # pixels: (B,3,H,W) in [-1,1]
    pixels = pixels.to(pipe.device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(pixels).latent_dist.sample()
    latents = latents * 0.18215  # SD v1.5 scaling
    return latents

# ====================== Training ======================

def train(cfg: SimpleNamespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg.seed))

    ensure_dir(cfg.output_dir)
    # Config-Kopie ablegen (repro)
    with open(Path(cfg.output_dir) / "config_used.yaml", "w") as f:
        yaml.safe_dump(vars(cfg), f, sort_keys=False)

    # Daten
    ds = ImageCaptionDataset(cfg.train_images, cfg.caption_file, cfg.resolution, cfg.prompt_prefix)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset leer: {cfg.train_images}")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Pipeline
    pipe = load_pipeline(cfg.base_model, device=device)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # LoRA anhängen
    attach_lora(pipe.unet, rank=int(cfg.lora_rank))
    for p in lora_parameters(pipe.unet):
        p.requires_grad_(True)

    # Optimizer
    trainable = list(lora_parameters(pipe.unet))
    if len(trainable) == 0:
        raise RuntimeError("Keine trainierbaren LoRA-Parameter gefunden.")
    opt = torch.optim.AdamW(trainable, lr=float(cfg.lr))

    noise_scheduler = pipe.scheduler

    print_kv("Config", dict(
        base_model=cfg.base_model,
        image_dir=cfg.train_images,
        caption_file=("keine" if not cfg.caption_file else cfg.caption_file),
        output_dir=cfg.output_dir,
        resolution=cfg.resolution,
        steps=cfg.max_train_steps,
        batch_size=cfg.batch_size,
        grad_acc=cfg.gradient_accumulation,
        lr=cfg.lr,
        rank=cfg.lora_rank,
        device=device,
        Trainingsbilder=len(ds),
    ))

    global_step = 0
    pipe.unet.train()

    def save_ckpt(tag: str):
        # versucht safetensors, sonst .bin
        stamp = f"step_{global_step:06d}_{tag}"
        out = Path(cfg.output_dir) / stamp
        ensure_dir(out)
        _, saver = import_lora_class()
        saver(pipe.unet, out)
        print(f"[💾] Checkpoint gespeichert → {out}")

    running_loss = 0.0
    for step in range(int(cfg.max_train_steps)):
        for _ in range(int(cfg.gradient_accumulation)):
            batch = next(iter(dl))  # einfacher: frisches Minibatch pro Accu-Step
            pixels = batch["pixel_values"]  # (B,3,H,W)
            prompts = batch["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts] * pixels.size(0)
            # Text
            enc_hid = encode_text(pipe, prompts)
            # Latents
            latents = encode_images_to_latents(pipe, pixels)

            # Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet vorwärts
            model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=enc_hid).sample
            loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean") / float(cfg.gradient_accumulation)
            loss.backward()
            running_loss += loss.item()

        opt.step(); opt.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % 10 == 0:
            avg = running_loss / 10.0
            print(f"step {global_step:05d} | loss={avg:.6f}")
            running_loss = 0.0

        if int(cfg.save_every_steps) > 0 and (global_step % int(cfg.save_every_steps) == 0):
            save_ckpt("auto")

    # Final speichern
    save_ckpt("final")
    print("\n✅ Training fertig.")

# ====================== CLI ======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml_to_ns(args.config)
    ensure_dir(cfg.output_dir)
    train(cfg)

if __name__ == "__main__":
    main()
