import os, json, zipfile, time, random
from typing import Tuple

def load_styles(path="configs/styles.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dirs():
    os.makedirs("out", exist_ok=True)

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def seed_everything(seed: int | None):
    import torch
    if seed is None or seed == 0:
        seed = random.randint(1, 2_147_483_647)
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    return seed, g

def make_zip(paths, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path

def compose_prompt(user_prompt: str, style_suffix: str):
    if style_suffix:
        return f"{user_prompt}, {style_suffix}"
    return user_prompt
