# Interaktive UI: Prompt ➜ Stil (1/2/3) ➜ Seed/Steps/N ➜ Render ➜ ZIP-Download
import os, json
from IPython.display import display, clear_output
import ipywidgets as widgets
from google.colab import files  # in Jupyter/Colab verfügbar
from .generate import generate
from .utils import load_styles, make_zip, now_tag

styles = load_styles()

# Widgets
title = widgets.HTML("<h3>🎨 Interaktiver Bild-Generator (Diffusers)</h3>")
prompt = widgets.Textarea(
    value="Comic-Illustration: ein stolzer Adler 🦅 und ein kräftiger Bär 🐻 stehen sich gegenüber, "
          "links US-Flagge, rechts russische Flagge, klare Outlines, bunte Farben",
    description="Prompt",
    layout=widgets.Layout(width="100%", height="100px")
)
style_dd = widgets.Dropdown(
    options=[("1 - Comic", "comic"), ("2 - Instagram", "instagram"), ("3 - Kinderbuch", "kinderbuch")],
    value="comic", description="Stil"
)
seed = widgets.IntText(value=0, description="Seed (0=rand)")
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

def on_render(_):
    with out:
        clear_output()
        print("Lade…")
        paths, meta = generate(
            prompt=prompt.value, style=style_dd.value,
            steps=int(steps.value), guidance_scale=float(guidance.value),
            seed=int(seed.value), n=int(count.value)
        )
        print("Fertig ✅")
        from PIL import Image
        for p in paths:
            display(Image.open(p))
        # Update neg default passend zum Stil
        neg.value = styles[style_dd.value]["negative"]
        # Merken
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
        files.download(zip_path)  # öffnet Download-Dialog

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
display(ui)
