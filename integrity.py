"""File fingerprinting and upload validation.

Validation is done on the bytes received by the server. Nothing uploaded is
ever executed; files are only decoded by Pillow and stored under random names.
"""
import hashlib
import io
import warnings
from pathlib import Path

from PIL import Image

# Extension -> Pillow format names that are acceptable for that extension.
ALLOWED_EXTENSIONS = {
    "jpg": {"JPEG", "MPO"},
    "jpeg": {"JPEG", "MPO"},
    "jpe": {"JPEG", "MPO"},
    "png": {"PNG"},
    "webp": {"WEBP"},
    "bmp": {"BMP"},
    "gif": {"GIF"},
    "tif": {"TIFF"},
    "tiff": {"TIFF"},
}
ALLOWED_FORMATS = set().union(*ALLOWED_EXTENSIONS.values())

# Canonical file extension used when storing a file of a given decoded format.
STORAGE_EXTENSION = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp",
                     "BMP": "bmp", "GIF": "gif", "TIFF": "tiff"}


class ValidationError(Exception):
    """Raised when an upload is rejected. The message is safe to show users."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_bytes(data):
    """Secondary fingerprint for cross-referencing older tools. Not collision resistant."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def file_extension(filename):
    name = Path(filename or "").name
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def validate_upload(filename, data, max_bytes, max_pixels):
    """Check extension, size and that the bytes decode as a supported image.

    Returns a dict describing the decoded file. Raises ValidationError.
    """
    if not filename:
        raise ValidationError("No file was selected.")
    ext = file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(f"Unsupported file extension '.{ext}'. Allowed: {allowed}.")
    if len(data) == 0:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(f"File is too large ({len(data) / 1048576:.1f} MB). "
                              f"Maximum is {max_bytes / 1048576:.0f} MB.")

    # Decode in two passes: verify() checks structure, load() decodes pixels.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                fmt = img.format
                width, height = img.size
                if width * height > max_pixels:
                    raise ValidationError(f"Image dimensions {width}x{height} exceed the "
                                          f"{max_pixels:,} pixel safety limit.")
                img.verify()
            with Image.open(io.BytesIO(data)) as img:
                img.load()
    except ValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValidationError("Image is too large to decode safely.")
    except Exception:
        raise ValidationError("The file could not be decoded as an image. It may be corrupt "
                              "or not an image at all.")

    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(f"Decoded image format '{fmt}' is not supported.")

    return {
        "extension": ext,
        "detected_format": fmt,
        "extension_matches_content": fmt in ALLOWED_EXTENSIONS[ext],
        "width": width,
        "height": height,
        "storage_extension": STORAGE_EXTENSION[fmt],
    }
