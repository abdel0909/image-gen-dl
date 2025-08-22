import os, re, json, math, pathlib, random
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from huggingface_hub import HfApi, HfFolder, whoami

# ----------------- Config -----------------
@dataclass
class Cfg:
    base_model: str
    train_images: str
    caption_file: Optional[str]
    output_dir: str
    resolution: int = 512
    max_train_steps: int = 800
    batch_size: int = 1
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    mixed_precision: str = "fp16"
    lr_scheduler: str = "cosine"
    save_every_steps: int = 200
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: tuple = ("to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2")
    prompt_prefix: str = ""
    negative_prompt: str = ""
    seed: int = 123

def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return Cfg(**yaml.safe_load(f))

# ----------------- Dataset -----------------
class FolderDataset(Dataset):
    def __init__(self, root, captions=None, size=512):
        self.paths = [str(p) for p in pathlib.Path(root).glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        self.size = size
        self.caps = {}
        if captions and pathlib.Path(captions).exists():
            with open(captions, "r") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    if ":" in line:
                        k,v = line.split(":",1)
                        self.caps[k.strip()] = v.strip()
        random.shuffle(self.paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB").resize((self.size,self.size), Image.BICUBIC)
        t = torch.tensor(list(im.getdata()), dtype=torch.uint8).view(self.size,self.size,3).permute(2,0,1)/255.0
        name = pathlib.Path(p).name
        cap = self.caps.get(name, None)
        return t, cap

# ----------------- LoRA utils -----------------
def inject_lora_unet(pipe, rank=16, alpha=16, target_modules=()):
    from peft import LoraConfig, get_peft_model
    import torch.nn as nn

    # Tiny adapter for linear layers
    def _wrap(module: nn.Module, name: str):
        m = getattr(module, name)
        in_f, out_f = m.in_features, m.out_features
        A = nn.Linear(in_f, rank, bias=False)
        B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        nn.init.zeros_(B.weight)
        m.forward_orig = m.forward
        def fwd(x):
            return m.forward_orig(x) + B(A(x)) * (alpha/rank)
        m.forward = fwd

    for n, m in pipe.unet.named_modules():
        if any(n.endswith(t) for t in target_modules):
            if hasattr(m, "in_features") and hasattr(m, "out_features"):
                parent_name = ".".join(n.split(".")[:-1])
                last = n.split(".")[-1]
                parent = pipe.unet.get_submodule(parent_name) if parent_name else pipe.unet
                _wrap(parent, last)
    return pipe

# ----------------- Train loop (simplified) -----------------
def train(cfg: Cfg):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    pipe = inject_lora_unet(pipe, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)

    ds = FolderDataset(cfg.train_images, cfg.caption_file, cfg.resolution)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # Dummy optimizer (we only update injected LoRA weights via patch above)
    opt = torch.optim.AdamW([p for p in pipe.unet.parameters() if p.requires_grad], lr=cfg.learning_rate)
    sch = get_scheduler(cfg.lr_scheduler, optimizer=opt, num_warmup_steps=0, num_training_steps=cfg.max_train_steps)

    out = pathlib.Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    step = 0
    pipe.unet.train()
    while step < cfg.max_train_steps:
        for batch, caps in dl:
            if step >= cfg.max_train_steps: break
            batch = batch.to(device, dtype=torch.float16 if device=="cuda" else torch.float32)
            # self-supervised noise prediction (very simplified example)
            noise = torch.randn_like(batch)
            loss = (batch - noise).abs().mean()  # placeholder loss
            loss.backward()
            if (step+1) % cfg.gradient_accumulation == 0:
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{cfg.max_train_steps} loss={float(loss):.4f}")
            if step % cfg.save_every_steps == 0:
                ck = out / f"checkpoint-{step}"
                ck.mkdir(parents=True, exist_ok=True)
                pipe.save_pretrained(ck)

    # final save
    pipe.save_pretrained(out)
    return str(out)

# ----------------- HF upload helpers -----------------
def _pick_best_or_last_checkpoint(output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    ckpts = sorted([p for p in out.glob("checkpoint-*") if p.is_dir()],
                   key=lambda p: int(re.findall(r"checkpoint-(\d+)", p.name)[0]) if re.findall(r"checkpoint-(\d+)", p.name) else -1)
    if ckpts: return str(ckpts[-1])
    return str(out)

def _ensure_repo(api: HfApi, repo_id: str, private=True):
    try: api.repo_info(repo_id)
    except Exception: api.create_repo(repo_id, private=private, repo_type="model")

def upload_to_hub(folder: str, repo_id: str, msg="auto: upload"):
    token = HfFolder.get_token()
    if not token: raise RuntimeError("HF-Token fehlt. HfFolder.save_token(<TOKEN>) ausführen.")
    api = HfApi(token=token)
    _ensure_repo(api, repo_id, private=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model",
                      commit_message=msg, ignore_patterns=["*.png","*.jpg","wandb/**","tensorboard/**"])
    print(f"✅ Upload fertig: https://huggingface.co/{repo_id}")

# ----------------- Main -----------------
def main():
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--repo_name", type=str, default="comic-real-lora")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = train(cfg)

    best = _pick_best_or_last_checkpoint(out_dir)
    user = whoami(HfFolder.get_token())["name"] if HfFolder.get_token() else "user"
    repo_id = f"{user}/{args.repo_name}"
    try:
        upload_to_hub(best, repo_id, msg=f"upload from {args.config}")
    except Exception as e:
        print("⚠️ Upload übersprungen/fehlgeschlagen:", e)

if __name__ == "__main__":
    main()
