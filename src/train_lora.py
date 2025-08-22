# src/train_lora.py — Neufassung (minimal & robust)
# -------------------------------------------------
import os, sys, math, json, time, inspect, argparse, random
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import yaml

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
# Fallback-Import für sehr alte diffusers:
# try:    from diffusers.models.lora import LoRAAttnProcessor
# except: pass

# =========================
#  Utils
# =========================
def load_yaml_to_ns(path: str) -> NS:
    """Lädt YAML tolerant mit Defaults und gibt ein Namespace zurück."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Defaults
    d = {
        "base_model":          "runwayml/stable-diffusion-v1-5",
        "train_images":        "data/real_style/images",
        "caption_file":        None,
        "prompt_prefix":       "photo, candid, natural light",
        "negative_prompt":     "watermark, text, logo",
        "resolution":          512,
        "max_train_steps":     200,
        "batch_size":          1,
        "gradient_accumulation": 1,
        "lr":                  1e-4,
        "lora_rank":           16,
        "mixed_precision":     "fp16",
        "save_every_steps":    50,
        "output_dir":          "runs/real_lora",
        "seed":                123,
    }
    d.update(data or {})
    # Normalisierungen
    d["train_images"]     = str(d["train_images"])
    d["output_dir"]       = str(d["output_dir"])
    d["mixed_precision"]  = str(d["mixed_precision"]).lower()
    return NS(**d)

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

# =========================
#  Dataset
# =========================
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir: str, size: int, prompt_prefix: str = "photo",
                 captions_txt: str | None = None):
        self.root = Path(root_dir)
        self.size = size
        self.prompt_prefix = prompt_prefix

        # Bilder sammeln
        exts = {".jpg",".jpeg",".png",".webp"}
        self.files = [p for p in self.root.glob("*") if p.suffix.lower() in exts]
        self.files.sort()

        # Optional Captions laden
        self.caps = {}
        if captions_txt and Path(captions_txt).exists():
            for line in Path(captions_txt).read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    name, cap = line.split(":", 1)
                    self.caps[name.strip()] = cap.strip()

        if len(self.files) == 0:
            # Dummy-Satz erzeugen, damit es sofort läuft
            ensure_dir(self.root)
            for i in range(60):
                im = Image.new("RGB", (self.size, self.size),
                               (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
                im.save(self.root / f"{i:04d}.png")
            self.files = [p for p in self.root.glob("*.png")]
            self.files.sort()

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        im = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        im = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
                               .view(self.size, self.size, 3)
                              ).numpy()).permute(2,0,1).float() / 255.0  # 0..1
        im = im * 2 - 1  # -> [-1, 1] wie VAE erwartet

        # Prompt bestimmen
        caption = self.caps.get(path.name, self.prompt_prefix)
        return {"pixel_values": im, "caption": caption}

# =========================
#  LoRA attach (robust)
# =========================
def attach_lora(unet, rank: int = 16):
    """
    Setzt LoRA-Prozessoren auf alle Attention-Module.
    Kompatibel mit diffusers-Versionen, die 'hidden_dim' ODER 'hidden_size' erwarten.
    """
    procs = {}
    for name, attn in unet.attn_processors.items():
        cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        hidden_dim = getattr(attn, "hidden_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(attn, "hidden_size", None)
        if hidden_dim is None:
            try:
                hidden_dim = attn.to_q.in_features
            except Exception:
                hidden_dim = unet.config.block_out_channels[-1]

        kw = {"cross_attention_dim": cross_dim, "rank": rank}
        sig = inspect.signature(LoRAAttnProcessor)
        if "hidden_dim" in sig.parameters:
            kw["hidden_dim"] = hidden_dim
        elif "hidden_size" in sig.parameters:
            kw["hidden_size"] = hidden_dim

        procs[name] = LoRAAttnProcessor(**kw)

    unet.set_attn_processor(procs)

    # Nur LoRA-Gewichte trainierbar machen
    for n, p in unet.named_parameters():
        p.requires_grad = "lora" in n

    trainable = [p for p in unet.parameters() if p.requires_grad]
    return trainable

# =========================
#  Training
# =========================
def train(cfg: NS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # Modell laden (lokaler Ordner oder HF-Repo-ID)
    base = cfg.base_model
    if Path(base).exists():
        pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16 if cfg.mixed_precision=="fp16" else torch.float32)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16 if cfg.mixed_precision=="fp16" else torch.float32)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_attention_slicing("max")

    # LoRA anhängen
    trainable_params = attach_lora(pipe.unet, rank=int(cfg.lora_rank))
    opt = torch.optim.AdamW(trainable_params, lr=float(cfg.lr))

    # Dataset / Loader
    ds = ImageCaptionDataset(cfg.train_images, int(cfg.resolution), cfg.prompt_prefix, cfg.caption_file)
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0, drop_last=True)

    # Text-Tokenizer/Encoder
    tok = pipe.tokenizer
    txtenc = pipe.text_encoder

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.mixed_precision=="fp16"))

    global_step = 0
    total = int(cfg.max_train_steps)
    save_every = int(cfg.save_every_steps)

    pipe.unet.train(); txtenc.eval(); pipe.vae.eval()

    print(f"\n== Training startet ==")
    print(f"Bilder: {len(ds)} | steps: {total} | bs: {cfg.batch_size} | rank: {cfg.lora_rank} | device: {device}")
    print(f"Output: {out_dir}")

    while global_step < total:
        for batch in dl:
            if global_step >= total:
                break

            with torch.no_grad():
                # Text-Embeddings
                enc = tok(batch["caption"], padding="max_length", truncation=True,
                          max_length=tok.model_max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                text_emb = txtenc(**enc).last_hidden_state

                # Bilder -> latents
                imgs = batch["pixel_values"].to(device)
                latents = pipe.vae.encode(imgs).latent_dist.sample() * pipe.vae.config.scaling_factor

                # Rauschen + Timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=device, dtype=torch.long)
                noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=(cfg.mixed_precision=="fp16")):
                model_pred = pipe.unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
                loss = F.mse_loss(model_pred, noise, reduction="mean")

            scaler.scale(loss).backward()

            if (global_step + 1) % int(cfg.gradient_accumulation) == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            if (global_step % 10) == 0:
                print(f"step {global_step:05d} | loss={loss.item():.4f}", flush=True)

            # Speichern
            if save_every > 0 and (global_step > 0) and (global_step % save_every == 0):
                ckpt = out_dir / f"step_{global_step:05d}.pt"
                save_lora(pipe.unet, ckpt)
                (out_dir / "config_used.yaml").write_text(yaml.safe_dump(vars(cfg), sort_keys=False), encoding="utf-8")
                print(f"  ✔ checkpoint -> {ckpt}")

            global_step += 1

    # final
    best = out_dir / "best.pt"
    save_lora(pipe.unet, best)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(vars(cfg), sort_keys=False), encoding="utf-8")
    print(f"\n✅ fertig: {best}")
    return str(best)

def save_lora(unet, path: Path):
    """Speichert nur die LoRA-Gewichte (klein)."""
    sd = {k: v.detach().cpu() for k, v in unet.state_dict().items() if "lora" in k}
    torch.save(sd, path)

# =========================
#  CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Pfad zur YAML-Config")
    args = ap.parse_args()

    cfg = load_yaml_to_ns(args.config)
    ensure_dir(Path(cfg.output_dir))
    train(cfg)

if __name__ == "__main__":
    main()
