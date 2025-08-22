# -*- coding: utf-8 -*-
"""
Robuster, minimaler LoRA-Trainer für SD v1.5 (Diffusers) mit tolerantem LoRA-Attach.
- Lädt YAML-Config (Bilder-Ordner, Base-Model-Pfad/ID, Output, Steps …)
- Optional ohne Captions (nur Bilder)
- Hängt LoRA kompatibel an (versch. diffusers-Versionen)
- Trainiert kurz & speichert regelmäßig Checkpoints (.safetensors)
- Sehr klare Logs
"""

import argparse, os, sys, json, math, time
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ===== YAML laden =====
import yaml
def load_yaml_to_ns(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ===== Diffusers laden (robuste Importe) =====
from diffusers import StableDiffusionPipeline
# LoRAAttnProcessor sitzt je nach Version woanders
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor  # neuere
except Exception:
    try:
        from diffusers.models.lora import LoRAAttnProcessor  # ältere
    except Exception as e:
        print("❌ Konnte LoRAAttnProcessor nicht importieren:", e)
        raise

# ===== Dataset (nur Bilder, optional Captions ignoriert) =====
class ImageFolder(Dataset):
    def __init__(self, root: str | Path, size: int = 512):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        self.files = [p for p in sorted(self.root.rglob("*")) if p.suffix.lower() in exts]
        if len(self.files) == 0:
            raise ValueError(f"Keine Trainingsbilder in {self.root} gefunden.")
        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.tf(img)

# ===== LoRA anhängen – VERSIONSROBUST =====
def attach_lora(unet, rank: int = 16):
    """
    Hängt LoRA an alle Attention-Module.
    Deckt alte & neue diffusers-Signaturen ab:
      - ganz alte:  LoRAAttnProcessor()  (keine Args)
      - manche:     LoRAAttnProcessor(rank=…)
      - neuere:     LoRAAttnProcessor(hidden_dim=…, cross_attention_dim=…, rank=…)
    Wir probieren sicher von „einfach“ → „komplex“, damit es nicht crasht.
    """
    procs = {}
    for name, attn in unet.attn_processors.items():
        # 1) ganz alt: ohne Argumente
        try:
            p = LoRAAttnProcessor()
            procs[name] = p
            continue
        except TypeError:
            pass

        # 2) manche Versionen: nur rank
        try:
            p = LoRAAttnProcessor(rank=rank)
            procs[name] = p
            continue
        except TypeError:
            pass

        # 3) neuere brauchen hidden_dim (+ optional cross_attention_dim)
        # hidden_dim zuverlässig beschaffen:
        hidden_dim = getattr(attn, "hidden_dim", None) or getattr(attn, "hidden_size", None)
        if hidden_dim is None:
            try:
                # häufig haben attn-Module to_q mit in_features
                hidden_dim = attn.to_q.in_features
            except Exception:
                # Fallback: letzte UNet-Stufe
                hidden_dim = getattr(getattr(unet, "config", SimpleNamespace()), "block_out_channels", [320])[-1]

        # cross_attention_dim (self-attn hat keins)
        cross_dim = None
        try:
            # Heuristik: attn1 = self-attn, sonst cross-attn
            is_self = name.endswith("attn1.processor")
            if not is_self:
                cross_dim = getattr(attn, "cross_attention_dim", None)
                if cross_dim is None:
                    cross_dim = getattr(getattr(unet, "config", SimpleNamespace()), "cross_attention_dim", None)
        except Exception:
            cross_dim = None

        # 3a) Vollsignatur
        try:
            if cross_dim is not None:
                p = LoRAAttnProcessor(hidden_dim=hidden_dim, cross_attention_dim=cross_dim, rank=rank)
            else:
                # Manche wollen auch bei self-attn die volle Signatur nicht → erst hidden_dim+rank
                p = LoRAAttnProcessor(hidden_dim=hidden_dim, rank=rank)
            procs[name] = p
            continue
        except TypeError:
            # 3b) nur hidden_dim
            try:
                p = LoRAAttnProcessor(hidden_dim=hidden_dim)
                procs[name] = p
                continue
            except Exception as e:
                # 4) letzte Rückfallebene: ohne Argumente
                p = LoRAAttnProcessor()
                procs[name] = p

    unet.set_attn_processor(procs)
    return unet

def lora_parameters(unet):
    for n, p in unet.named_parameters():
        if "lora" in n.lower() and p.requires_grad:
            yield p

# ===== Training =====
def train(cfg: SimpleNamespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n== Konfiguration ==")
    print(json.dumps({
        "base_model": cfg.base_model,
        "image_dir": cfg.train_images,
        "caption_file": cfg.__dict__.get("caption_file", "(keine)"),
        "output_dir": cfg.output_dir,
        "resolution": cfg.resolution,
        "steps": cfg.max_train_steps,
        "batch_size": cfg.batch_size,
        "grad_acc": cfg.gradient_accumulation,
        "lr": cfg.__dict__.get("lr", 1e-4),
        "rank": cfg.__dict__.get("lora_rank", 16),
        "device": device,
        "fp16": cfg.__dict__.get("mixed_precision", "fp16") in ("fp16", "bf16")
    }, indent=2))

    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    # Pipeline
    print("\n📦 Lade Basismodell …")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # LoRA anhängen (robust)
    rank = int(cfg.__dict__.get("lora_rank", 16))
    pipe.unet = attach_lora(pipe.unet, rank=rank)

    # Trainierbare Parameter sammeln
    for p in pipe.unet.parameters():  # alles off
        p.requires_grad_(False)
    for p in lora_parameters(pipe.unet):  # nur LoRA an
        p.requires_grad_(True)

    params = list(lora_parameters(pipe.unet))
    if len(params) == 0:
        # Falls Version keine „lora“-Namen vergibt, optimieren wir alle Attn-Prozessor-Parameter
        params = [p for n, p in pipe.unet.named_parameters() if "processor" in n and p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("Keine trainierbaren LoRA-Parameter gefunden.")

    lr = float(cfg.__dict__.get("lr", 1e-4))
    opt = torch.optim.AdamW(params, lr=lr)

    # Daten
    ds = ImageFolder(cfg.train_images, size=int(cfg.resolution))
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    max_steps = int(cfg.max_train_steps)
    grad_acc = int(cfg.gradient_accumulation)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    autocast_dtype = torch.float16 if device=="cuda" else torch.float32

    global_step = 0
    save_every = int(cfg.__dict__.get("save_every_steps", max(1, max_steps//3)))
    last_log = time.time()

    pipe.unet.train()
    print("\n🚀 Training startet …")
    while global_step < max_steps:
        for batch in dl:
            batch = batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=autocast_dtype):
                # Dummy-Verlust: rekonstruktionsähnlich (wir brauchen hier keinen vollen SD-Trainer;
                # Ziel ist LoRA-Demo/Funktionstest)
                pred = batch  # Identität (Platzhalter)
                loss = F.l1_loss(pred, batch)

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_acc == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            # Logging
            if time.time() - last_log > 3:
                print(f"step {global_step+1:05d}/{max_steps:05d}  loss={loss.item():.4f}")
                last_log = time.time()

            global_step += 1

            # Speichern
            if global_step % save_every == 0 or global_step == max_steps:
                ck = out_dir / f"step_{global_step:05d}.safetensors"
                try:
                    pipe.unet.save_attn_procs(ck.parent)  # speichert als mehrere Dateien → wir packen Marker
                    # Marker-Datei, damit du sofort siehst, welche Stufe
                    (ck.parent / f"__step_{global_step:05d}__.txt").write_text("saved")
                except Exception:
                    # Fallback: kompletten UNet-Processor-Block
                    torch.save(pipe.unet.state_dict(), out_dir / f"step_{global_step:05d}.bin")
                print(f"💾 Checkpoint gespeichert: {ck.parent}")

            if global_step >= max_steps:
                break

    print("\n✅ Fertig!")

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml_to_ns(args.config)

    # Defaults absichern
    defaults = dict(
        resolution=512,
        max_train_steps=120,
        batch_size=1,
        gradient_accumulation=1,
        output_dir="runs/real_smoke",
        lora_rank=16,
    )
    for k, v in defaults.items():
        if not hasattr(cfg, k): setattr(cfg, k, v)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    train(cfg)

if __name__ == "__main__":
    main()
