"""Download and verify the AI model ahead of time (so the demo works offline).

Usage:  python download_model.py
"""
import sys

from PIL import Image

from ai_detector import AIDetector
from config import load_config


def main():
    cfg = load_config()
    print(f"Model:    {cfg['AI_MODEL_ID']}")
    print(f"Revision: {cfg['AI_MODEL_REVISION']}")
    detector = AIDetector(cfg["AI_MODEL_ID"], cfg["AI_MODEL_REVISION"], enabled=True,
                          local_files_only=cfg["AI_LOCAL_FILES_ONLY"])
    print("Downloading / loading (about 330 MB on first run)...")
    if not detector.load():
        print("FAILED:", detector.error)
        return 1
    print("Loaded:", detector.status())
    result = detector.predict(Image.new("RGB", (256, 256), (128, 128, 128)))
    print("Smoke-test inference on a flat grey image:", result["status"],
          [(s["label"], s["score"]) for s in result.get("scores", [])])
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
