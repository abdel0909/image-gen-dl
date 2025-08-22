# -*- coding: utf-8 -*-
"""
Minimaler, robuster LoRA-Trainer für SD v1.5
- ohne diffusers.LoRAAttnProcessor (eigene LoRA-Linear)
- captions optional (fixed prompt fallback)
- speichert alle N Schritte Checkpoints (.pt) in cfg["output_dir"]
"""

import os, math, json, time, argparse, random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer

# -------------------------- Utils --------------------------

def load_yaml_to_ns(path: str | Path) -> SimpleNamespace:
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------- LoRA Linear (eigen) ------------------

class LoRALinear(nn.Module):
    """Leichtgewichtige LoRA-Schicht um eine bestehende Linear-Projektion."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: int | None = None):
        super().__init__()
        self.base = base_linear
        self.base.requires_grad_(False)

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        r = max(1, int(rank))
        self.rank = r
        self.alpha = int(alpha) if alpha is not None else r
        self.scaling = self.alpha / self.rank

        # A: in -> r, B: r -> out
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        # Kaiming init für A, Null für B (üblich bei LoRA)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Baseline (gefreezt)
        y = F.linear(x, self.base.weight, self.base.bias)
        # LoRA-Anteil
        lora = x @ self.lora_A.t()
        lora = lora @ self.lora_B.t()
        return y + self.scaling * lora

def patch_unet_lora(unet: nn.Module, rank: int = 16, targets=("to_q","to_k","to_v","to_out.0")):
    """
    Ersetzt alle Linear-Layer, deren Name auf eins der targets endet, durch LoRALinear.
    Gibt Liste trainierbarer Parameter sowie ein Mapping (Name -> Modul) zurück.
    """
    name_to_module = dict(unet.named_modules())
    trainable_params = []
    patched = 0

    def get_parent_and_attr(root, dotted):
        parts = dotted.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    for full_name, m in list(unet.named_modules()):
        if isinstance(m, nn.Linear) and any(full_name.endswith(t) for t in targets):
            parent, attr = get_parent_and_attr(unet, full_name)
            new = LoRALinear(m, rank=rank)
            # auf richtiges device/dtype bringen
            new.to(next(unet.parameters()).device)
            setattr(parent, attr, new)
            # nur LoRA-Parameter trainieren
            trainable_params += [new.lora_A, new.lora_B]
            patched += 1

    return trainable_params, patched

# ------------------------- Dataset -------------------------

class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir: Path, tokenizer: AutoTokenizer, resolution=512,
                 captions_file: Path | None = None, default_prompt="a photo"):
        self.imgs = []
        img_dir = Path(img_dir)
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:
                self.imgs.append(p)
        self.tokenizer = tokenizer
        self.res = int(resolution)
        self.default_prompt = default_prompt
        self.prompts = None

        if captions_file and Path(captions_file).is_file():
            # Einfaches Format: eine Zeile pro Bild; Reihenfolge = Sortierreihenfolge der Bilder
            with open(captions_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines()]
            # ggf. kürzen/auffüllen
            if len(lines) < len(self.imgs):
                lines += [self.default_prompt] * (len(self.imgs) - len(lines))
            self.prompts = lines[:len(self.imgs)]

        self.tf = transforms.Compose([
            transforms.Resize(self.res, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.res),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5],[0.5]),  # -> [-1,1]
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        pixel_values = self.tf(img)
        prompt = self.prompts[idx] if self.prompts else self.default_prompt
        ids = self.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": ids}

# ------------------------- Training ------------------------

def train(cfg: SimpleNamespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(getattr(cfg, "seed", 123)))

    # Pipeline laden
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16 if getattr(cfg,"fp16",True) and device.type=="cuda" else torch.float32
    )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None
    pipe.safety_checker = None  # für Training nicht nötig

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    # Daten
    img_dir = Path(cfg.train_images)
    captions_file = getattr(cfg, "caption_file", None)
    dataset = ImageCaptionDataset(
        img_dir=img_dir, tokenizer=tokenizer,
        resolution=int(getattr(cfg,"resolution",512)),
        captions_file=Path(captions_file) if captions_file and captions_file != "(keine)" else None,
        default_prompt=getattr(cfg, "prompt_prefix", "a photo")
    )
    assert len(dataset) > 0, "Dataset leer – keine Trainingsbilder gefunden."
    dl = DataLoader(dataset, batch_size=int(getattr(cfg,"batch_size",1)), shuffle=True, drop_last=True)

    # UNet patchen (eigene LoRA)
    for p in unet.parameters(): p.requires_grad_(False)
    rank = int(getattr(cfg, "rank", getattr(cfg, "lora_rank", 16)))
    lora_params, patched = patch_unet_lora(unet, rank=rank)
    assert patched > 0, "Keine Attention-Projektionen gepatcht – LoRA konnte nicht eingesetzt werden."

    # Nur LoRA trainieren
    params = [p for p in lora_params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(getattr(cfg,"lr",1e-4)))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and getattr(cfg,"fp16",True)))

    # Ausgabe/Checkpoints
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    # Config-Kopie sichern
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2)

    max_steps = int(getattr(cfg,"max_train_steps",120))
    save_every = int(getattr(cfg,"save_every_steps",40))

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    step = 0
    t0 = time.time()
    while step < max_steps:
        for batch in dl:
            step += 1
            if step > max_steps: break

            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                enc = text_encoder(input_ids)[0]

            # Vorwärts + Loss
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and getattr(cfg,"fp16",True))):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=enc).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if step % 20 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"step {step:05d}/{max_steps:05d}  loss={loss.item():.4f}  ({elapsed:.1f}s)")

            if step % save_every == 0 or step == max_steps:
                ck = {
                    "rank": rank,
                    "state_dict": {  # nur LoRA-Gewichte
                        k: v.detach().cpu()
                        for k, v in unet.state_dict().items()
                        if "lora_A" in k or "lora_B" in k
                    }
                }
                ck_path = out_dir / f"step_{step:05d}.pt"
                torch.save(ck, ck_path)
                print(f"💾 Checkpoint gespeichert → {ck_path}")

    print("✅ Training fertig.")

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml_to_ns(args.config)

    # Backwards-Compat: Keys angleichen
    defaults = dict(
        base_model = "runwayml/stable-diffusion-v1-5",
        train_images = "data/real_style/images",
        caption_file = None,
        output_dir = "runs/real_smoke",
        resolution = 512,
        max_train_steps = 120,
        batch_size = 1,
        gradient_accumulation = 1,
        lr = 1e-4,
        rank = getattr(cfg, "lora_rank", 16),
        fp16 = True,
        seed = 123,
        save_every_steps = 40,
        prompt_prefix = "natural light, candid, shallow depth of field"
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k): setattr(cfg, k, v)

    ensure_dir(Path(cfg.output_dir))
    train(cfg)

if __name__ == "__main__":
    main()
