"""Image property, EXIF and embedded-metadata extraction (read-only)."""
import io
import mimetypes

from PIL import ExifTags, Image

EXIF_IFD = 0x8769
GPS_IFD = 0x8825

# Embedded text keys some generators/editors write into PNG/WEBP info chunks.
_TEXT_KEYS_OF_INTEREST = ("parameters", "prompt", "workflow", "Software", "Comment",
                          "Description", "Author", "Creation Time", "Source", "XML:com.adobe.xmp")

_MAX_VALUE_LEN = 500


def _clean(value):
    """Convert EXIF values into JSON-safe, length-limited representations."""
    if isinstance(value, bytes):
        text = value.rstrip(b"\x00")
        try:
            decoded = text.decode("utf-8")
            if decoded.isprintable():
                return decoded[:_MAX_VALUE_LEN]
        except UnicodeDecodeError:
            pass
        return f"<{len(value)} bytes binary>"
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value][:64]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in list(value.items())[:64]}
    if isinstance(value, str):
        return value.replace("\x00", "").strip()[:_MAX_VALUE_LEN]
    if isinstance(value, (int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 6)
    # IFDRational and similar numeric types
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        pass
    text = str(value).replace("\x00", "")
    return text[:_MAX_VALUE_LEN]


def _ifd_to_dict(ifd, names):
    out = {}
    for tag_id, value in ifd.items():
        name = names.get(tag_id, f"Tag_0x{tag_id:04X}")
        if tag_id in (EXIF_IFD, GPS_IFD):
            continue
        out[name] = _clean(value)
    return out


def _gps_decimal(gps):
    """Convert GPS IFD to decimal degrees when complete; otherwise None."""
    try:
        def to_deg(vals, ref):
            d, m, s = (float(v) for v in vals)
            deg = d + m / 60 + s / 3600
            return -deg if ref in ("S", "W") else deg
        lat = to_deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef"))
        lon = to_deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef"))
        return {"latitude": round(lat, 6), "longitude": round(lon, 6)}
    except Exception:
        return None


def device_name(key_fields):
    """'Canon' + 'Canon EOS 5D' -> 'Canon EOS 5D' (many cameras repeat the make in the model)."""
    make = (key_fields.get("camera_make") or "").strip()
    model = (key_fields.get("camera_model") or "").strip()
    if make and model.lower().startswith(make.lower()):
        return model
    return " ".join(p for p in (make, model) if p) or "unknown device"


def _detect_markers(data):
    """Byte-level presence checks for metadata containers Pillow does not parse."""
    head = data[: 4 * 1024 * 1024]
    return {
        "xmp_present": b"http://ns.adobe.com/xap/1.0/" in head or b"<x:xmpmeta" in head,
        # C2PA manifests live in JUMBF boxes labelled "c2pa". Presence only - not validated.
        "c2pa_marker_present": b"c2pa" in head and b"jumb" in head,
        "iptc_present": b"Photoshop 3.0" in head,
    }


def extract_metadata(data, original_filename):
    """Extract metadata from raw file bytes. Never raises for metadata problems."""
    result = {
        "file": {
            "size_bytes": len(data),
            "mime_type_from_extension": mimetypes.guess_type(original_filename)[0],
        },
        "image": {},
        "exif": {},
        "exif_sub_ifd": {},
        "gps": {},
        "key_fields": {},
        "embedded_text": {},
        "containers": {},
        "errors": [],
    }

    with Image.open(io.BytesIO(data)) as img:
        result["image"] = {
            "format": img.format,
            "format_description": img.format_description,
            "mime_type": Image.MIME.get(img.format),
            "width": img.width,
            "height": img.height,
            "megapixels": round(img.width * img.height / 1_000_000, 3),
            "color_mode": img.mode,
            "has_alpha": "A" in img.getbands() or "transparency" in img.info,
            "frames": getattr(img, "n_frames", 1),
            "icc_profile_present": bool(img.info.get("icc_profile")),
            "dpi": _clean(img.info.get("dpi")) if img.info.get("dpi") else None,
            "progressive_jpeg": bool(img.info.get("progressive") or img.info.get("progression")),
        }

        try:
            exif = img.getexif()
            result["exif"] = _ifd_to_dict(exif, ExifTags.TAGS)
            try:
                result["exif_sub_ifd"] = _ifd_to_dict(exif.get_ifd(EXIF_IFD), ExifTags.TAGS)
            except Exception as exc:  # malformed sub-IFD
                result["errors"].append(f"EXIF sub-IFD unreadable: {type(exc).__name__}")
            try:
                gps_raw = exif.get_ifd(GPS_IFD)
                result["gps"] = _ifd_to_dict(gps_raw, ExifTags.GPSTAGS)
            except Exception as exc:
                result["errors"].append(f"GPS IFD unreadable: {type(exc).__name__}")
        except Exception as exc:
            result["errors"].append(f"EXIF unreadable: {type(exc).__name__}")

        for key in _TEXT_KEYS_OF_INTEREST:
            if key in img.info and isinstance(img.info[key], (str, bytes)):
                result["embedded_text"][key] = _clean(img.info[key])
        # PNG tEXt/iTXt chunks are exposed by Pillow as img.text
        for key, value in getattr(img, "text", {}).items():
            result["embedded_text"].setdefault(key, _clean(value))

    result["containers"] = _detect_markers(data)

    exif, sub = result["exif"], result["exif_sub_ifd"]
    software = exif.get("Software") or result["embedded_text"].get("Software")
    result["key_fields"] = {
        "camera_make": exif.get("Make"),
        "camera_model": exif.get("Model"),
        "lens_model": sub.get("LensModel"),
        "capture_datetime": sub.get("DateTimeOriginal"),
        "digitized_datetime": sub.get("DateTimeDigitized"),
        "modified_datetime": exif.get("DateTime"),
        "software": software,
        "artist": exif.get("Artist"),
        "orientation": exif.get("Orientation"),
        "gps_present": bool(result["gps"]),
        "gps_coordinates": _gps_decimal(result["gps"]) if result["gps"] else None,
    }
    result["exif_present"] = bool(exif or sub)
    result["exif_tag_count"] = len(exif) + len(sub) + len(result["gps"])
    return result
