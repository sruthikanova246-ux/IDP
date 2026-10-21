"""Application configuration.

Every setting can be overridden with an environment variable (prefix ``DEI_``)
or by passing a dict of overrides to ``create_app`` (used by the tests).
"""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Pretrained detector selected on the validation split of tests/eval_data (see README "Evaluation"
# and evaluation/results.json). Pinned commit so results are reproducible.
DEFAULT_MODEL_ID = "Smogy/SMOGY-Ai-images-detector"
DEFAULT_MODEL_REVISION = "fb9aadb85642d707f05b40191689d60f75b9a3e6"


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _load_secret_key(instance_dir):
    """Use DEI_SECRET_KEY if set, otherwise a random key persisted in instance/."""
    env_key = os.environ.get("DEI_SECRET_KEY")
    if env_key:
        return env_key
    key_file = instance_dir / "secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    key_file.write_text(key, encoding="utf-8")
    return key


def load_config(overrides=None):
    overrides = overrides or {}
    instance_dir = Path(overrides.get("INSTANCE_DIR") or os.environ.get("DEI_INSTANCE_DIR") or BASE_DIR / "instance")
    instance_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "INSTANCE_DIR": instance_dir,
        "UPLOAD_DIR": instance_dir / "uploads",
        "DERIVED_DIR": instance_dir / "derived",
        "DATABASE_PATH": instance_dir / "evidence.db",
        "LOG_PATH": instance_dir / "app.log",
        "MAX_UPLOAD_MB": int(os.environ.get("DEI_MAX_UPLOAD_MB", "20")),
        # Decompression-bomb guard: refuse images above this many pixels.
        "MAX_IMAGE_PIXELS": int(os.environ.get("DEI_MAX_IMAGE_PIXELS", str(80_000_000))),
        "AI_ENABLED": _env_bool("DEI_AI_ENABLED", True),
        "AI_MODEL_ID": os.environ.get("DEI_MODEL_ID", DEFAULT_MODEL_ID),
        "AI_MODEL_REVISION": os.environ.get("DEI_MODEL_REVISION", DEFAULT_MODEL_REVISION),
        # When true, never touch the network: use only the local Hugging Face cache.
        "AI_LOCAL_FILES_ONLY": _env_bool("DEI_MODEL_LOCAL_ONLY", False),
        # Start loading the model in a background thread when the server starts.
        "AI_PRELOAD": _env_bool("DEI_MODEL_PRELOAD", True),
        "HOST": os.environ.get("DEI_HOST", "127.0.0.1"),
        "PORT": int(os.environ.get("DEI_PORT", "5000")),
        "TESTING": False,
    }
    cfg.update(overrides)
    for key in ("UPLOAD_DIR", "DERIVED_DIR", "DATABASE_PATH", "LOG_PATH"):
        cfg[key] = Path(cfg[key])
    cfg["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    cfg["DERIVED_DIR"].mkdir(parents=True, exist_ok=True)
    if "SECRET_KEY" not in cfg:
        cfg["SECRET_KEY"] = _load_secret_key(instance_dir)
    cfg["MAX_CONTENT_LENGTH"] = cfg["MAX_UPLOAD_MB"] * 1024 * 1024
    return cfg
