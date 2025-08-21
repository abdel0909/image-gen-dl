import os, math, json, random, argparse, yaml
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

def load_cfg(path):
    with open(path, "r") as f: return yaml.safe_load(f)

class CaptionDataset(Dataset):
    def __init__(self, img_dir, caption_file=None, size=768, prefix="", fallback_name_as_caption=True):
        self.paths = sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg",".webp"]])
        self.size = size
        self.prefix = prefix
        self.caps = {}
        if caption_file and Path(caption_file).exists():
            for line in Path(caption_file).read_text().splitlines():
                if not line.strip(): continue
                name, cap = line.split(":",1)
                self.caps[name.strip()] = cap.strip()
        self.fallback = fallback_name_as_caption

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        if p.name in self.caps:
            cap = self.caps[p.name]
        elif self.fallback:
            cap = p.stem.replace("_"," ")
        else:
            cap = ""
        return img, (self.prefix + " " + cap).strip()

def save_lora(pipe, out):
    os.makedirs(out, exist_ok=True)
    pipe.save_pretrained(out)  # PEFT LoRA Gewichte landen im Ordner
    print("✅ LoRA gespeichert:", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    base = cfg["base_model"]
    is_sdxl = "xl" in base.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_sdxl:
        pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet

    # LoRA an UNet (und optional Textencoder) hängen
    lconf = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["lora_target_modules"])
    unet = get_peft_model(unet, lconf)
    pipe.unet = unet

    ds = CaptionDataset(
        cfg["train_images"],
        caption_file=cfg.get("caption_file"),
        size=cfg["resolution"],
        prefix=cfg.get("prompt_prefix",""),
        fallback_name_as_caption=True
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg["lr"])
    total_steps = cfg["max_train_steps"]
    lr_sched = get_cosine_with_hard_restarts_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=total_steps, num_cycles=1)

    unet.train()
    step = 0
    while step < total_steps:
        for imgs, caps in dl:
            step += 1
            imgs = imgs.permute(0,3,1,2).to(device, dtype=torch.float16)/255.0
            latents = vae.encode(imgs).latent_dist.sample()*0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (imgs.shape[0],), device=device, dtype=torch.long)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            enc = tokenizer(list(caps), padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            if is_sdxl:
                cond = text_encoder(enc.input_ids)[0]
                cond_v = pipe.text_encoder_2(enc.input_ids)[0] if hasattr(pipe, "text_encoder_2") else cond
                encoder_hidden_states = cond
            else:
                encoder_hidden_states = text_encoder(enc.input_ids)[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step(); lr_sched.step(); opt.zero_grad()

            if step % 50 == 0:
                print(f"step {step}/{total_steps} | loss {loss.item():.4f}")

            if step >= total_steps: break

    save_lora(pipe, cfg["output_dir"])

if __name__ == "__main__":
    main()
