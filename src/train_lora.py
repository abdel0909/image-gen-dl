# src/train_lora.py  —  komplett neu (Drop-in)
# Minimaler LoRA-Trainer für SD v1.5 (robustes YAML, Captions optional, Checkpoints)
import os, math, argparse, json, time, random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from torchvision import transforms as T


# ---------- Utils ----------

def load_cfg(path: str) -> SimpleNamespace:
    """YAML -> Namespace, mit sinnvollen Defaults. 'caption_file' ist optional."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Defaults
    data.setdefault("base_model", "runwayml/stable-diffusion-v1-5")
    data.setdefault("train_images", "data/real_style/images")
    data.setdefault("caption_file", None)          # <- wichtig: optional!
    data.setdefault("resolution", 512)
    data.setdefault("max_train_steps", 200)
    data.setdefault("batch_size", 1)
    data.setdefault("gradient_accumulation", 1)
    data.setdefault("mixed_precision", "fp16")
    data.setdefault("save_every_steps", 100)
    data.setdefault("output_dir", "runs/real_lora")
    data.setdefault("seed", 123)
    data.setdefault("lora_rank", 16)
    data.setdefault("lr", 1e-4)
    data.setdefault("prompt_prefix", "")

    # Pfade normieren
    data["train_images"]  = str(Path(data["train_images"]))
    if data["caption_file"]:
        data["caption_file"] = str(Path(data["caption_file"]))
    data["output_dir"]   = str(Path(data["output_dir"]))

    return SimpleNamespace(**data)


class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir: str, caption_file: str | None, prompt_prefix: str = "", size: int = 512):
        self.paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            self.paths.extend(sorted(Path(img_dir).glob(ext)))
        if not self.paths:
            raise FileNotFoundError(f"Keine Trainingsbilder in {img_dir} gefunden.")

        self.size = size
        self.prompt_prefix = prompt_prefix.strip() if prompt_prefix else ""

        self.captions = {}
        if caption_file and Path(caption_file).exists():
            # eine Zeile pro Bildname:  '0001.png: eine beschreibung ...'
            with open(caption_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    name, cap = line.split(":", 1)
                    self.captions[name.strip()] = cap.strip()

        self.tf = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        pixel = self.tf(img)

        # Caption: aus Datei, sonst Dateiname, optional Prefix voranstellen
        base = p.name
        cap = self.captions.get(base, Path(base).stem.replace("_", " ").replace("-", " "))
        if self.prompt_prefix:
            cap = f"{self.prompt_prefix} {cap}".strip()

        return {"pixel_values": pixel, "caption": cap}


# ---------- LoRA Setup ----------
def add_lora_to_unet(unet, rank=16):
    """
    Hängt LoRA-Prozessoren an alle Attention-Module im UNet.
    Diffusers >=0.20: LoRAAttnProcessor(hidden_size, cross_attention_dim, rank).
    Wir ermitteln hidden_size heuristisch über die Namenskonventionen.
    """
    unet.requires_grad_(False)
    lora_attn_procs = {}
    # Mapping der Block-Kanäle (hidden sizes) gemäß UNet-Konfig
    block_out = list(unet.config.block_out_channels)
    cross_dim = unet.config.cross_attention_dim

    def hidden_from_name(name: str) -> int:
        # name z.B. "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"
        if name.startswith("down_blocks"):
            idx = int(name.split(".")[1])
            return block_out[idx]
        if name.startswith("up_blocks"):
            idx = int(name.split(".")[1])
            # up-blocks laufen in umgekehrter Reihenfolge
            return block_out[::-1][idx]
        # mid_block
        return block_out[-1]

    for name, module in unet.attn_processors.items():
        hidden_size = hidden_from_name(name)
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            rank=rank
        )

    unet.set_attn_processor(lora_attn_procs)
    # trainierbare LoRA-Parameter freischalten
    trainable = []
    for _, p in unet.attn_processors.items():
        if hasattr(p, "to_q_lora"):
            trainable += list(p.parameters())
    for p in trainable:
        p.requires_grad_(True)
    return trainable


# ---------- Training ----------
def train(cfg: SimpleNamespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config-Kopie mitschreiben
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg.__dict__)), f, sort_keys=False, allow_unicode=True)

    # Pipeline + Scheduler
    dtype = torch.float16 if cfg.mixed_precision.lower() in ("fp16", "float16") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    # Datensatz
    ds = ImageCaptionDataset(cfg.train_images, cfg.caption_file, cfg.prompt_prefix, cfg.resolution)
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    # Noise-Scheduler (DDPM -> wie bei SD)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

    # LoRA an UNet hängen
    trainable_params = add_lora_to_unet(pipe.unet, rank=int(cfg.lora_rank))
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.lr))

    global_step = 0
    pipe.unet.train()

    print(f"🚀 Start — steps={cfg.max_train_steps}, save_every={cfg.save_every_steps}, "
          f"batch={cfg.batch_size}, imgs={len(ds)}")
    last_save = 0
    accum = int(cfg.gradient_accumulation)

    while global_step < int(cfg.max_train_steps):
        for batch in dl:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = batch["caption"]

            # 1) Text-Encoder
            with torch.no_grad():
                text_inp = pipe.tokenizer(
                    list(prompts), padding="max_length",
                    truncation=True, max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt"
                )
                text_inp = {k: v.to(device) for k, v in text_inp.items()}
                enc_out = pipe.text_encoder(**text_inp)
                text_embeds = enc_out[0]

            # 2) VAE-Encode -> latents
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # 3) Noise hinzufügen
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4) Vorhersage
            model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample

            # 5) Ziel (eps)
            target = noise

            loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean") / accum
            loss.backward()

            if (global_step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (global_step + 1) % int(cfg.save_every_steps) == 0:
                ckpt = out_dir / f"step_{global_step+1}.safetensors"
                save_lora(pipe.unet, ckpt)
                print(f"💾 checkpoint: {ckpt}")
                last_save = global_step + 1

            global_step += 1
            if global_step >= int(cfg.max_train_steps):
                break

    # Ende: letztes/bestes speichern (falls nicht gerade gespeichert)
    if last_save != global_step:
        ckpt = out_dir / f"step_{global_step}.safetensors"
        save_lora(pipe.unet, ckpt)
        print(f"✅ fertig — letzter checkpoint: {ckpt}")

    return str(out_dir)


def save_lora(unet, path: Path):
    """
    Speichert nur die LoRA-Gewichte aus den Attention-Prozessoren
    als .safetensors-Datei (klein). Läuft ohne zusätzliche Abhängigkeiten.
    """
    try:
        from safetensors.torch import save_file
    except Exception:
        # Fallback auf torch.save
        def save_file(state, p): torch.save(state, p)

    state = {}
    for name, proc in unet.attn_processors.items():
        # LoRAAttnProcessor hat A/B-Matrizen: to_q_lora, to_k_lora, ...
        for attr in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
            if hasattr(proc, attr):
                lora = getattr(proc, attr)
                state[f"{name}.{attr}.up.weight"] = lora.up.weight.detach().cpu()
                state[f"{name}.{attr}.down.weight"] = lora.down.weight.detach().cpu()
                if hasattr(lora, "alpha"):
                    state[f"{name}.{attr}.alpha"] = torch.tensor(float(lora.alpha))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # wenn safetensors verfügbar:
    try:
        from safetensors.torch import save_file as _save
        _save(state, str(path))
    except Exception:
        torch.save(state, str(path))


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Pfad zur YAML-Config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    print("Config:", cfg.__dict__)

    out = train(cfg)
    print("Output:", out)


if __name__ == "__main__":
    main()
