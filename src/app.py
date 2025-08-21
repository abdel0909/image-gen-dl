# src/app.py
import os, json
from IPython.display import display, clear_output
import ipywidgets as widgets
from google.colab import files

from .generate import generate
from .utils import load_styles, make_zip, now_tag, discover_loras

styles = load_styles()

# LoRA-Auswahl vorbereiten
loras = [("Keine (nur Basis-Modell)", "")]
for label, path in discover_loras("models/loras"):
    loras.append((label, path))

# Fallback für LoRA-Template (nimmt sonst 'instagram')
lora_base_key = "lora_template" if "lora_template" in styles else "instagram"
lora_base_cfg = styles.get(lora_base_key, styles["instagram"])

title = widgets.HTML("<h3>🎨 Interaktiver Bild-Generator (mit LoRA-Stil)</h3>")

prompt = widgets.Textarea(
    value="Comic-Illustration: ein stolzer Adler 🦅 und ein kräftiger Bär 🐻, klare Outlines, bunte Farben",
    description="Prompt",
    layout=widgets.Layout(width="100%", height="100px")
)

style_dd = widgets.Dropdown(
    options=[("1 - Comic", "comic"), ("2 - Instagram", "instagram"), ("3 - Kinderbuch", "kinderbuch")],
    value="comic", description="Stil"
)

lora_dd = widgets.Dropdown(
    options=loras,
    value=loras[0][1],
    description="LoRA"
)

seed = widgets.IntText(value=0, description="Seed(0=rand)")
steps = widgets.IntSlider(value=30, min=10, max=60, step=2, description="Steps")
guidance = widgets.FloatSlider(value=7.5, min=3.0, max=12.0, step=0.5, description="Guidance")
count = widgets.IntSlider(value=1, min=1, max=6, step=1, description="Anzahl")
neg = widgets.Text(
    value=styles["comic"]["negative"],
    description="Negativ",
    layout=widgets.Layout(width="100%")
)

btn_render = widgets.Button(description="🖌️ Rendern", button_style="primary")
btn_save = widgets.Button(description="💾 Speichern / Download (ZIP)")
out = widgets.Output()

def _apply_style_defaults(key: str):
    cfg = styles[key]
    steps.value = int(cfg.get("steps", steps.value))
    guidance.value = float(cfg.get("guidance_scale", guidance.value))
    neg.value = cfg.get("negative", neg.value)

def on_style_change(change):
    if change["name"] == "value":
        _apply_style_defaults(change["new"])

style_dd.observe(on_style_change, names="value")
_apply_style_defaults(style_dd.value)

def on_render(_):
    with out:
        clear_output()
        use_style = style_dd.value
        use_lora = lora_dd.value or ""

        # Wenn eine LoRA gewählt ist, Basiswerte aus lora_template/instagram ziehen (nur Defaults):
        if use_lora:
            base = lora_base_cfg
            if steps.value == 30:  # nur überschreiben, wenn noch auf Default
                steps.value = int(base.get("steps", steps.value))
            if guidance.value == 7.5:
                guidance.value = float(base.get("guidance_scale", guidance.value))
            if not neg.value:
                neg.value = base.get("negative", "")

        print("Lade… (Stil:", use_style, "| LoRA:", ("keine" if not use_lora else os.path.basename(use_lora)), ")")
        paths, meta = generate(
            prompt=prompt.value,
            style=use_style,
            steps=int(steps.value),
            guidance_scale=float(guidance.value),
            seed=int(seed.value),
            n=int(count.value),
            negative=neg.value,
            lora_path=use_lora if use_lora else None
        )
        print("Fertig ✅")
        from PIL import Image
        for p in paths:
            display(Image.open(p))
        out.rendered_paths = paths

def on_save(_):
    with out:
        if not hasattr(out, "rendered_paths") or not out.rendered_paths:
            print("Bitte zuerst rendern.")
            return
        tag = now_tag()
        zip_path = f"out/images_{tag}.zip"
        make_zip(out.rendered_paths, zip_path)
        print("ZIP erstellt:", zip_path)
        files.download(zip_path)

btn_render.on_click(on_render)
btn_save.on_click(on_save)

ui = widgets.VBox([
    title,
    prompt,
    widgets.HBox([style_dd, lora_dd]),
    widgets.HBox([seed, count]),
    widgets.HBox([steps, guidance]),
    neg,
    widgets.HBox([btn_render, btn_save]),
    out
])
display(ui)
