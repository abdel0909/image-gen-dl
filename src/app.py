# src/app.py
# Interaktive UI: Prompt ➜ Stil (auto aus configs/styles.json) ➜ Seed/Steps/N ➜ Render ➜ ZIP-Download

import os, json, traceback
from IPython.display import display, clear_output
import ipywidgets as widgets
from PIL import Image

# --- flexible Imports: funktioniert in Colab (flat) und als Modulstart (python -m src.app)
try:
    from .generate import generate
    from .utils import load_styles, make_zip, now_tag
except Exception:
    from src.generate import generate
    from src.utils import load_styles, make_zip, now_tag

# Google Colab Download nur laden, wenn verfügbar
try:
    from google.colab import files as colab_files  # type: ignore
except Exception:
    colab_files = None

# === Styles laden (robust) ===
styles = load_styles()  # liest configs/styles.json
if not isinstance(styles, dict) or not styles:
    raise RuntimeError("Keine Styles gefunden. Prüfe configs/styles.json")

# Anzeige-Namen: "1 - Comic", "2 - Instagram", ...
style_names = list(styles.keys())
style_options = [(f"{i+1} - {name.title()}", name) for i, name in enumerate(style_names)]
default_style_key = style_names[0]

# === Widgets ===
title = widgets.HTML("<h3>🎨 Interaktiver Bild-Generator (Diffusers)</h3>")

prompt = widgets.Textarea(
    value="Comic-Illustration: ein stolzer Adler 🦅 und ein kräftiger Bär 🐻 "
          "stehen sich gegenüber, links US-Flagge, rechts russische Flagge, "
          "klare Outlines, bunte Farben",
    description="Prompt",
    layout=widgets.Layout(width="100%", height="110px")
)

style_dd = widgets.Dropdown(options=style_options, value=default_style_key, description="Stil")

seed = widgets.IntText(value=0, description="Seed (0=rand)")
steps = widgets.IntSlider(value=int(styles[default_style_key].get("steps", 30)),
                          min=8, max=80, step=1, description="Steps")
guidance = widgets.FloatSlider(value=float(styles[default_style_key].get("guidance_scale", 7.5)),
                               min=2.0, max=15.0, step=0.5, description="Guidance")
count = widgets.IntSlider(value=1, min=1, max=8, step=1, description="Anzahl")

neg = widgets.Text(
    value=str(styles[default_style_key].get("negative", "")),
    description="Negativ",
    layout=widgets.Layout(width="100%")
)

btn_render = widgets.Button(description="🖌️ Rendern", button_style="primary")
btn_save   = widgets.Button(description="💾 Speichern / Download (ZIP)")
out        = widgets.Output()

# Speichert zuletzt erzeugte Bildpfade
render_state = {"paths": []}

def _apply_style_defaults(style_key: str):
    """Werte (steps/guidance/negativ) aus dem Style in die UI übernehmen."""
    cfg = styles.get(style_key, {})
    if "steps" in cfg:
        steps.value = int(cfg["steps"])
    if "guidance_scale" in cfg:
        guidance.value = float(cfg["guidance_scale"])
    if "negative" in cfg:
        neg.value = str(cfg["negative"])

def on_style_change(change):
    if change["name"] == "value":
        _apply_style_defaults(change["new"])

style_dd.observe(on_style_change, names="value")

def on_render(_):
    with out:
        clear_output()
        try:
            print("⏳ Render läuft …")
            paths, meta = generate(
                prompt=prompt.value,
                style=style_dd.value,
                steps=int(steps.value),
                guidance_scale=float(guidance.value),
                seed=int(seed.value),
                n=int(count.value),
                negative=neg.value,   # wird in generate (optional) genutzt
            )
            render_state["paths"] = list(paths or [])
            if not render_state["paths"]:
                print("⚠️ Keine Bilder erzeugt.")
                return

            # Anzeige (klein, schnell)
            for p in render_state["paths"]:
                try:
                    img = Image.open(p)
                    w, h = img.size
                    scale = 512 / max(w, h)
                    img = img.resize((int(w*scale), int(h*scale)))
                    display(img)
                except Exception:
                    display(Image.open(p))
            print(f"✅ Fertig: {len(render_state['paths'])} Bild(er)")
        except Exception as e:
            print("❌ Fehler beim Rendern:")
            print(e)
            traceback.print_exc()

def on_save(_):
    with out:
        if not render_state["paths"]:
            print("Bitte zuerst rendern.")
            return
        tag = now_tag()
        os.makedirs("out", exist_ok=True)
        zip_path = f"out/images_{tag}.zip"
        make_zip(render_state["paths"], zip_path)
        print("ZIP erstellt:", zip_path)
        if colab_files is not None:
            try:
                colab_files.download(zip_path)  # Download-Dialog in Colab
            except Exception:
                pass

btn_render.on_click(on_render)
btn_save.on_click(on_save)

ui = widgets.VBox([
    title,
    prompt,
    widgets.HBox([style_dd, seed, count]),
    widgets.HBox([steps, guidance]),
    neg,
    widgets.HBox([btn_render, btn_save]),
    out
])

# Style-Defaults einmal initial setzen
_apply_style_defaults(style_dd.value)

display(ui)
