# src/train_lora.py  —  vereinfachtes, robustes LoRA-Training (Diffusers)
# Ersetzt komplett deine bestehende train_lora.py

import os, math, time, json, argparse, random
from pathlib import Path
from types import SimpleNamespace as SN

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml

from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel

# ----------------------------
# Utils
# ----------------------------

def load_yaml(path: str) -> SN:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SN(**data)

def set_seed(seed: int):
    if seed is None or int(seed) == 0:  # 0 oder None = random
        seed = int(time.time()) % 2_000_000_000
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_config_copy(cfg: SN, out_dir: Path):
    ensure_dir(out_dir)
    with open(out_dir / "config_used.json", "w") as f:
        json.dump(vars(cfg), f, indent=2, ensure_ascii=False)

# ----------------------------
# Dataset (einfach: Ordner mit JPG/PNG)
# ----------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, root, size=512):
        self.root = Path(root)
        self.size = size
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if len(self.paths) == 0:
            raise RuntimeError(f"Keine Trainingsbilder gefunden in: {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        im = im.resize((self.size, self.size), Image.BICUBIC)
        im = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
                               .view(self.size, self.size, 3)
                               .float() / 255.0).numpy())  # (H,W,3) float
        im = im.permute(2, 0, 1) * 2 - 1  # [-1,1] und nach (C,H,W)
        return {"pixel_values": im}

# ----------------------------
# LoRA an UNet hängen
# ----------------------------

def add_lora_to_unet(unet, r=16, alpha=16):
    lora_attn_procs = {}
    for name, module in unet.named_modules():
        if hasattr(module, "set_processor"):
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=module.to_q.in_features, rank=r)
    unet.set_attn_processor(lora_attn_procs)
    # Parameter-Container, damit wir nur LoRA-Gewichte optimieren
    trainable = AttnProcsLayers(unet.attn_processors)
    for p in trainable.parameters():
        p.requires_grad_(True)
    return trainable

# ----------------------------
# Training
# ----------------------------

def train(cfg: SN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if str(getattr(cfg, "mixed_precision", "fp16")).lower() == "fp16" and torch.cuda.is_available() else torch.float32

    print(">> Lade Pipeline …")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype
    )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    vae: AutoencoderKL = pipe.vae
    text_encoder: CLIPTextModel = pipe.text_encoder
    tokenizer: CLIPTokenizer = pipe.tokenizer
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # LoRA nur auf die UNet-Attention trainieren
    trainable = add_lora_to_unet(unet, r=int(getattr(cfg, "lora_rank", 16)), alpha=int(getattr(cfg, "lora_alpha", 16)))
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=float(getattr(cfg, "lr", 1e-4)))

    # Dataset & Loader
    print(">> Baue Dataset …")
    ds = ImageFolderDataset(cfg.train_images, size=int(getattr(cfg, "resolution", 512)))
    dl = DataLoader(ds, batch_size=int(getattr(cfg, "batch_size", 1)), shuffle=True, num_workers=0, drop_last=True)

    # Prompt-Handling (ein globaler Stil-Prefix funktioniert für LoRA erstaunlich gut)
    prompt_prefix = getattr(cfg, "prompt_prefix", "")
    tokens = tokenizer([prompt_prefix] * int(getattr(cfg, "batch_size", 1)), padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)

    out_dir = Path(getattr(cfg, "output_dir", "runs/lora_out"))
    ensure_dir(out_dir)
    save_every = int(getattr(cfg, "save_every_steps", 200))
    max_steps = int(getattr(cfg, "max_train_steps", 800))

    print(f">> Starte Training: {len(ds)} Bilder, steps={max_steps}, save_every={save_every}")
    global_step = 0
    unet.train()
    text_encoder.eval()
    vae.eval()

    for epoch in range(10_000):  # wird über steps begrenzt
        for batch in dl:
            global_step += 1
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)

            # 1) zu latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            # 2) Noise addieren
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3) Text-Embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(tokens)[0]

            # 4) UNet vorwärts
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 5) Loss = MSE zwischen predicted noise und echter noise
            loss = nn.functional.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                print(f"[{global_step:05d}/{max_steps}] loss={loss.item():.4f}")

            # Checkpoint speichern?
            if global_step % save_every == 0 or global_step == max_steps:
                ckpt_dir = out_dir / f"step_{global_step:06d}"
                ensure_dir(ckpt_dir)
                # Nur LoRA-Gewichte sichern
                trainable.save_pretrained(ckpt_dir)
                with open(ckpt_dir / "meta.json", "w") as f:
                    json.dump({"step": global_step, "loss": float(loss.item())}, f, indent=2)
                print(f"💾 Checkpoint gespeichert: {ckpt_dir}")

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    print("✅ Training fertig.")
    print(f"➡️  Alle Checkpoints liegen in: {out_dir}")
    return str(out_dir)

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Pfad zu YAML (z. B. configs/train_smoke.yaml)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg.seed = set_seed(getattr(cfg, "seed", 0))
    ensure_dir(getattr(cfg, "output_dir", "runs/lora_out"))
    save_config_copy(cfg, Path(cfg.output_dir))
    train(cfg)

if __name__ == "__main__":
    main()
