# src/train_lora.py
import argparse, yaml, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Pfad zu einer YAML-Config, z.B. configs/train_comic.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ Config nicht gefunden: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("✅ Config geladen:")
    for k, v in cfg.items():
        print(f"  - {k}: {v}")

    print("\nℹ️ Platzhalter-Runner: Das Modul ist da und liest die Config.")
    print("   Im nächsten Schritt füge ich dir die echte LoRA-Trainingslogik ein.")

if __name__ == "__main__":
    main()
