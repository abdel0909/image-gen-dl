# src/utils.py
import os, json, zipfile, datetime
from typing import Dict, Any

def load_styles(path: str = "configs/styles.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        # Fallback: relativ zu src/
        alt = os.path.join(os.path.dirname(__file__), "..", path)
        alt = os.path.normpath(alt)
        if os.path.exists(alt):
            path = alt
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)

def make_zip(paths, zip_path: str):
    ensure_outdir(os.path.dirname(zip_path))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            arcname = os.path.basename(p)
            z.write(p, arcname=arcname)

def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # src/utils.py (Ergänzung)
import os, glob, json, datetime

def discover_loras(root="models/loras"):
    """
    Sucht rekursiv nach *.safetensors in models/loras/** und
    gibt eine Liste von (name, path) zurück. Der neueste zuerst.
    """
    paths = glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True)
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    items = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
        items.append((f"LoRA: {base} ({ts})", p))
    return items
