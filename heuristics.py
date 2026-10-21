"""Explainable rule-based image findings and the heuristic caution score.

These checks look for *clues* in file structure and metadata. None of them can
prove manipulation or authenticity. The caution score is a transparent sum of
the weights listed below; it is NOT a probability and it never includes the
AI model's output.
"""
import io
from datetime import datetime

from PIL import Image, ImageChops, ImageOps, ImageStat

from metadata import device_name

GENERATOR_PATTERNS = [
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dall·e", "dalle",
    "firefly", "comfyui", "automatic1111", "novelai", "invokeai", "imagen", "flux",
    "leonardo.ai", "nightcafe", "dreamstudio", "gpt-image", "openai", "ideogram",
]
EDITOR_PATTERNS = [
    "photoshop", "lightroom", "gimp", "snapseed", "canva", "picsart", "pixelmator",
    "affinity", "paint.net", "facetune", "photopea", "fotor", "luminar", "capture one",
    "darktable", "rawtherapee", "lunapic", "pixlr", "meitu", "vsco",
]
# PNG text keys written by popular diffusion front-ends with the generation settings.
GENERATOR_TEXT_KEYS = {"parameters", "prompt", "workflow", "sd-metadata", "dream", "invokeai_metadata"}
GENERATOR_SIDES = {512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1216, 1280, 1344, 1536, 1792, 2048}

WEIGHTS = {
    "generator_signature": 40,
    "editing_software": 15,
    "modified_after_capture": 10,
    "extension_mismatch": 10,
    "no_camera_metadata": 5,
    "generator_typical_dimensions": 5,
    "low_jpeg_quality": 5,
}

# IJG standard luminance quantization table (quality 50), zig-zag independent.
_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
             69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
             81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def _match(text, patterns):
    low = (text or "").lower()
    return next((p for p in patterns if p in low), None)


def _parse_exif_dt(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip()[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def estimate_jpeg_quality(img):
    """Approximate IJG quality from the luminance quantization table (None if not JPEG)."""
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    table = list(tables[0])
    if len(table) != 64:
        return None
    # Pillow returns tables in zig-zag order; compare sorted values to be order independent.
    scale = sum(sorted(table)) / sum(sorted(_STD_LUMA)) * 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(max(1, min(100, round(quality))))


def error_level_analysis(img, out_path, quality=90):
    """Recompress at a fixed JPEG quality and visualise the per-pixel difference."""
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as resaved:
        diff = ImageChops.difference(rgb, resaved.convert("RGB"))
    stat = ImageStat.Stat(diff.convert("L"))
    max_diff = max(hi for _, hi in diff.getextrema()) or 1
    vis = diff.point(lambda v: min(255, int(v * 255.0 / max_diff)))
    vis.thumbnail((1600, 1600))
    vis.save(out_path, "PNG")
    return {
        "resave_quality": quality,
        "mean_difference": round(stat.mean[0], 3),
        "stddev_difference": round(stat.stddev[0], 3),
        "max_difference": int(max_diff),
        "visualisation_scale": round(255.0 / max_diff, 2),
    }


def analyse(img, metadata, file_info, ela_path=None):
    """Return {'findings': [...], 'caution_score': int, 'caution_band': str, 'ela': {...}|None}."""
    findings = []

    def add(fid, title, detail, weight_key=None, kind="caution"):
        weight = WEIGHTS.get(weight_key, 0) if weight_key else 0
        findings.append({"id": fid, "title": title, "detail": detail, "weight": weight,
                         "kind": kind if weight else "info"})

    kf = metadata["key_fields"]
    texts = metadata.get("embedded_text", {})
    software = kf.get("software") or ""

    gen_hit = _match(software, GENERATOR_PATTERNS)
    gen_keys = sorted(k for k in texts if k.lower() in GENERATOR_TEXT_KEYS)
    if not gen_hit:
        for value in texts.values():
            gen_hit = _match(str(value), GENERATOR_PATTERNS)
            if gen_hit:
                break
    if gen_hit or gen_keys:
        what = []
        if gen_hit:
            what.append(f"text matching '{gen_hit}'")
        if gen_keys:
            what.append(f"embedded chunk(s) {', '.join(gen_keys)} used by image-generation tools")
        add("generator_signature", "AI generator signature in metadata",
            "The file contains " + " and ".join(what) + ". Metadata can be added or forged, but "
            "this is a strong clue that a generation tool produced or processed the file.",
            "generator_signature")

    editor_hit = _match(software, EDITOR_PATTERNS)
    if editor_hit:
        add("editing_software", "Editing software recorded",
            f"The Software tag is '{software}'. Editing software is routinely used for legitimate "
            "adjustments (cropping, exposure, export); this indicates processing, not deception.",
            "editing_software")
    elif software and not gen_hit:
        add("software_tag", "Software tag present",
            f"The Software tag is '{software}'. It is not a known editor or generator (often camera "
            "or phone firmware).")

    captured = _parse_exif_dt(kf.get("capture_datetime"))
    modified = _parse_exif_dt(kf.get("modified_datetime"))
    if captured and modified and abs((modified - captured).total_seconds()) > 60:
        add("modified_after_capture", "Modification date differs from capture date",
            f"DateTimeOriginal is {kf['capture_datetime']} but DateTime (last change) is "
            f"{kf['modified_datetime']}. The file was likely re-saved or processed after capture.",
            "modified_after_capture")

    if not file_info["extension_matches_content"]:
        add("extension_mismatch", "File extension does not match content",
            f"The name ends in '.{file_info['extension']}' but the content decodes as "
            f"{file_info['detected_format']}. This often happens after conversion or renaming.",
            "extension_mismatch")

    has_camera = bool(kf.get("camera_make") or kf.get("camera_model"))
    if not metadata.get("exif_present"):
        add("no_camera_metadata", "No EXIF metadata",
            "No EXIF data was found. Social networks, messaging apps, screenshots and many export "
            "tools strip EXIF, so this is common for legitimate images; it only means less "
            "provenance information is available.", "no_camera_metadata")
    elif not has_camera:
        add("no_camera_metadata", "EXIF present but no camera make/model",
            "EXIF exists but does not identify a capture device, which limits provenance checks.",
            "no_camera_metadata")
    else:
        add("camera_metadata", "Camera metadata present",
            f"Device recorded as {device_name(kf)}. EXIF can be copied or edited, so this is "
            "supporting context only.")

    w, h = img.size
    if w in GENERATOR_SIDES and h in GENERATOR_SIDES and not has_camera:
        add("generator_typical_dimensions", "Dimensions common for image generators",
            f"{w}x{h} matches a resolution frequently produced by diffusion models. Many ordinary "
            "images share these sizes; this is a weak clue only.", "generator_typical_dimensions")

    quality = estimate_jpeg_quality(img)
    if quality is not None:
        if quality < 70:
            add("low_jpeg_quality", "Low JPEG quality",
                f"Estimated JPEG quality is about {quality}. Strong compression (often from "
                "re-sharing) removes fine detail and reduces the reliability of every other check.",
                "low_jpeg_quality")
        else:
            add("jpeg_quality", "JPEG quality estimate",
                f"Estimated JPEG quality is about {quality} (from the quantization table).")

    if kf.get("gps_present"):
        add("gps_present", "GPS location recorded",
            "The file contains GPS tags. Location can be compared against the claimed location "
            "of the event, but GPS tags can also be edited.")
    containers = metadata.get("containers", {})
    if containers.get("c2pa_marker_present"):
        add("c2pa_marker", "Possible Content Credentials (C2PA) data",
            "Byte markers for a C2PA manifest were found. This application does not validate "
            "the signature; use an official C2PA verification tool.")
    if containers.get("xmp_present"):
        add("xmp_present", "XMP metadata present",
            "XMP packets are usually written by editing or cataloguing software and may contain "
            "edit history.")

    ela = None
    if ela_path is not None and img.format in ("JPEG", "MPO"):
        try:
            ela = error_level_analysis(img, ela_path)
            add("ela", "Error Level Analysis generated",
                f"Mean recompression difference {ela['mean_difference']} (max {ela['max_difference']}). "
                "ELA is a visual aid only and is not included in the caution score: bright regions "
                "can indicate different compression history, but edges and textures are also bright.")
        except Exception as exc:
            add("ela_failed", "Error Level Analysis failed", f"{type(exc).__name__}: {exc}")

    score = min(100, sum(f["weight"] for f in findings))
    band = "low" if score < 15 else "moderate" if score < 40 else "elevated"
    return {"findings": findings, "caution_score": score, "caution_band": band,
            "jpeg_quality_estimate": quality, "ela": ela, "weights": WEIGHTS}
