"""Evidence consistency analysis.

Cross-checks the AI model output, metadata, file properties and file identity
with explicit, readable rules. It produces observations and follow-up checks.
It deliberately does not merge anything into a single authenticity number.
"""
from metadata import device_name

ALWAYS_RECOMMENDED = [
    "Obtain the original file directly from the source device or account and compare SHA-256 hashes.",
    "Run a reverse image search to find earlier or differently-cropped copies.",
    "Corroborate the depicted event with independent sources (other photos, witnesses, records).",
    "For legal or high-stakes use, refer the file to a qualified digital forensic examiner.",
]


def analyse(ai, metadata, file_info, heuristics, identity):
    obs = []

    def add(oid, category, kind, title, detail, follow_up=None):
        obs.append({"id": oid, "category": category, "kind": kind, "title": title,
                    "detail": detail, "follow_up": follow_up})

    kf = metadata["key_fields"]
    finding_ids = {f["id"] for f in heuristics["findings"]}
    has_camera = bool(kf.get("camera_make") or kf.get("camera_model"))
    has_exif = metadata.get("exif_present")
    generator_sig = "generator_signature" in finding_ids
    ai_ok = ai.get("status") == "ok" and ai.get("ai_score") is not None
    ai_flagged = ai_ok and ai.get("predicted_class") == "ai_generated"

    # --- File identity --------------------------------------------------------
    add("hash_identity", "identity", "info", "SHA-256 fingerprint recorded",
        f"SHA-256 {identity['sha256']} was computed from the bytes received by this application. "
        "It uniquely identifies this exact file: any change to a single byte (including re-saving "
        "or metadata stripping) produces a different hash. It says nothing about whether the "
        "depicted scene is genuine, and it does not establish the file's history before upload.",
        "Share the hash with anyone holding a copy so they can confirm they have the identical file.")
    if identity.get("stored_copy_verified"):
        add("storage_verified", "identity", "info", "Stored copy verified",
            "The copy written to evidence storage was re-hashed and matches the received bytes.")
    for dup in identity.get("previous_matches", []):
        add(f"duplicate_{dup['id']}", "identity", "info", "Identical file analysed before",
            f"Report {dup['id']} ({dup['original_filename']}, {dup['uploaded_at']}) has the same "
            "SHA-256, so it is byte-for-byte the same file.")

    # --- AI availability ------------------------------------------------------
    if ai.get("status") != "ok":
        add("ai_unavailable", "ai", "caution", "AI detection did not run",
            f"AI status: {ai.get('status')}. {ai.get('message', '')} The observations below rely "
            "on metadata and file structure only. The absence of an AI flag must not be read as "
            "evidence that the image is camera-original.",
            "Re-run the analysis once the model is available.")
    elif ai.get("ai_score") is None:
        add("ai_unmapped", "ai", "caution", "AI output could not be interpreted",
            "The model ran but its labels could not be mapped to AI/human classes.")

    # --- AI vs metadata -------------------------------------------------------
    if ai_ok:
        score = ai["ai_score"]
        pct = f"{score * 100:.1f}%"
        if 0.35 <= score <= 0.65:
            add("ai_borderline", "ai", "caution", "Borderline classifier output",
                f"The AI-class score ({pct}) is close to the decision boundary; the model is not "
                "decisive for this image.", "Do not rely on the classifier result for this image.")

        if ai_flagged and generator_sig:
            add("ai_and_metadata_agree_generated", "cross-check", "caution",
                "Classifier and metadata both point to generation",
                f"The classifier favours 'AI-generated' ({pct}) and the metadata contains a "
                "generator signature. Two independent lines of evidence agree, which strengthens "
                "the indication, but metadata can be forged and the classifier can err.",
                "Look for the original prompt/source on the generation platform if available.")
        elif ai_flagged and has_camera:
            add("ai_flag_vs_camera_metadata", "cross-check", "caution",
                "Classifier flag conflicts with camera metadata",
                f"The classifier favours 'AI-generated' ({pct}), yet EXIF names a camera "
                f"({device_name(kf)}). Both findings are "
                "reported as-is: EXIF can be copied onto a generated image, and detectors also "
                "produce false positives on heavily processed real photos (HDR, beauty filters, "
                "upscaling).",
                "Compare with other photos from the same device: sensor noise, lens data and "
                "file structure should be similar.")
        elif ai_flagged:
            add("ai_flag_no_provenance", "cross-check", "caution",
                "Classifier flag with no supporting provenance",
                f"The classifier favours 'AI-generated' ({pct}) and there is no camera metadata. "
                "This is consistent with a generated image, but also with a real photo whose "
                "metadata was stripped by a platform.",
                "Ask the source for the original capture file.")
        elif generator_sig:
            add("metadata_generated_vs_ai_not_flagged", "cross-check", "caution",
                "Metadata indicates a generator but classifier does not",
                f"The classifier does not favour 'AI-generated' ({pct}) although the metadata "
                "contains a generator signature. Detectors frequently miss images from generators "
                "that were not in their training data.",
                "Treat the metadata clue seriously; the classifier result does not override it.")
        elif has_camera:
            add("ai_not_flagged_camera_metadata", "cross-check", "info",
                "Classifier and metadata do not raise generation concerns",
                f"The classifier does not favour 'AI-generated' ({pct}) and camera metadata is "
                "present. This is consistent with a camera image but does not prove it is "
                "unedited or that the scene is genuine.")
        else:
            add("ai_not_flagged_no_metadata", "cross-check", "info",
                "Classifier not flagged, limited provenance",
                f"The classifier does not favour 'AI-generated' ({pct}). With no camera metadata "
                "there is little else to corroborate the result.")

    # --- Metadata context -----------------------------------------------------
    if not has_exif:
        add("missing_exif_context", "metadata", "info", "Missing metadata is not proof of manipulation",
            "Many legitimate images have stripped metadata (messaging apps, social media, "
            "screenshots, privacy tools). Missing EXIF only reduces what can be checked.",
            "If the image came from social media, try to obtain the pre-upload original.")
    if "editing_software" in finding_ids:
        add("editing_is_a_clue", "metadata", "caution", "Editing software is a clue, not proof",
            f"Software '{kf.get('software')}' processed this file. Routine edits are normal; "
            "the question is what was changed.",
            "Ask for the unedited original and an explanation of edits performed.")
    if "modified_after_capture" in finding_ids:
        add("timeline_gap", "metadata", "caution", "File changed after capture",
            "The EXIF modification time is later than the capture time.",
            "Check whether the stated capture time matches the claimed event time.")
    if not file_info["extension_matches_content"]:
        add("format_conversion", "file", "caution", "Possible format conversion or renaming",
            "The extension and actual format differ; conversion re-encodes pixels and usually "
            "discards original metadata.")

    # --- Reliability of the classifier for this input -------------------------
    w, h = metadata["image"]["width"], metadata["image"]["height"]
    if ai_ok:
        if min(w, h) < 224:
            add("small_input", "ai-reliability", "caution", "Image smaller than model input",
                f"The image ({w}x{h}) is upscaled to the model's input size; results on small "
                "images are less reliable.")
        elif max(w, h) > 1024:
            add("downscaled_input", "ai-reliability", "info", "Image downscaled for the model",
                f"The {w}x{h} image is resized to {ai.get('model_input_size')} for the model, so "
                "fine pixel-level artefacts are not visible to it.")
        if (heuristics.get("jpeg_quality_estimate") or 100) < 70:
            add("compression_reduces_reliability", "ai-reliability", "caution",
                "Heavy compression reduces reliability",
                "Strong JPEG compression can hide or mimic generation artefacts.")
        if metadata["image"]["has_alpha"] or metadata["image"]["frames"] > 1:
            add("input_flattened", "ai-reliability", "info", "Model saw a flattened image",
                "Transparency was flattened onto white and only the first frame was analysed.")

    follow_ups = [o["follow_up"] for o in obs if o.get("follow_up")]
    if kf.get("gps_present"):
        follow_ups.append("Compare the GPS coordinates with the claimed location.")
    if metadata.get("containers", {}).get("c2pa_marker_present"):
        follow_ups.append("Validate the embedded Content Credentials with an official C2PA verifier.")
    recommended = list(dict.fromkeys(follow_ups + ALWAYS_RECOMMENDED))

    return {"observations": obs, "recommended_checks": recommended,
            "summary": _summary(ai, ai_flagged, generator_sig, heuristics, has_camera)}


def _summary(ai, ai_flagged, generator_sig, heuristics, has_camera):
    """A qualitative preliminary statement. Never a probability of authenticity."""
    if ai.get("status") != "ok":
        level = "limited"
        text = ("AI detection was unavailable, so this preliminary assessment is based on metadata "
                "and file structure only.")
    elif ai_flagged or generator_sig:
        conflicting = (ai_flagged and has_camera and not generator_sig) or (generator_sig and not ai_flagged)
        level = "conflicting" if conflicting else "review"
        text = ("Indicators of possible AI generation were found"
                + (", but the available evidence conflicts" if conflicting else "")
                + ". Further review is recommended.")
    elif heuristics["caution_band"] != "low":
        level = "review"
        text = "The classifier did not flag the image, but metadata clues warrant further review."
    else:
        level = "no_strong_indicators"
        text = "No strong indicators of AI generation or processing were found by these checks."
    text += (" This is a preliminary, automated assessment. It does not verify authenticity "
             "and does not prove that the image is real or fake.")
    return {"level": level, "text": text}
