import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_detector import AIDetector  # noqa: E402
from app import create_app  # noqa: E402
from config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION  # noqa: E402


def make_app(tmp_path, detector=None, **overrides):
    cfg = {"INSTANCE_DIR": tmp_path, "TESTING": True, "AI_PRELOAD": False,
           "AI_ENABLED": False, "SECRET_KEY": "test-secret", **overrides}
    return create_app(cfg, detector=detector)


def csrf(client):
    client.get("/upload")
    with client.session_transaction() as sess:
        return sess["_csrf"]


def upload(client, name, data, token=None):
    token = token or csrf(client)
    return client.post("/upload", data={"csrf_token": token, "evidence": (io.BytesIO(data), name)},
                       headers={"X-Requested-With": "fetch"}, content_type="multipart/form-data")


def gradient(size=(320, 240)):
    img = Image.new("RGB", size)
    img.putdata([(x % 256, y % 256, (x * y) % 256) for y in range(size[1]) for x in range(size[0])])
    return img


@pytest.fixture
def jpeg_with_exif():
    img = gradient()
    exif = Image.Exif()
    exif[0x010F] = "Canon"                      # Make
    exif[0x0110] = "Canon EOS 5D Mark IV"       # Model
    exif[0x0131] = "Adobe Photoshop 25.0"       # Software
    exif[0x0132] = "2024:05:02 10:00:00"        # DateTime (modified)
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = "2024:05:01 09:00:00"         # DateTimeOriginal
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92, exif=exif)
    return buf.getvalue()


@pytest.fixture
def png_no_exif():
    buf = io.BytesIO()
    gradient().save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def png_generator():
    info = PngInfo()
    info.add_text("parameters", "a photo of a cat, Steps: 30, Sampler: Euler a, Model: sd_xl_base_1.0")
    buf = io.BytesIO()
    gradient((1024, 1024)).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


@pytest.fixture(scope="session")
def real_detector():
    """The real pretrained model from the local Hugging Face cache (run download_model.py first)."""
    det = AIDetector(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, enabled=True, local_files_only=True)
    if not det.load():
        pytest.skip(f"Real AI model not available locally: {det.error}")
    return det
