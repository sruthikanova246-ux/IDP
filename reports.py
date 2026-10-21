"""Integrity report assembly and TXT / JSON / PDF export."""
import json
import re
import textwrap
from datetime import datetime, timezone

APP_NAME = "Digital Evidence Integrity System"
APP_VERSION = "1.1.0"
SCHEMA_VERSION = "1.1"

ASSESSMENT_TYPE = "Preliminary automated assessment - not a verification of authenticity"

LIMITATIONS = [
    "No automated tool can conclusively prove that an image is real or AI-generated.",
    "The AI classifier's scores are model outputs, not calibrated probabilities, and the model "
    "can be confidently wrong, especially on generators, cameras or processing it was not trained on.",
    "The heuristic caution score is a transparent sum of rule weights. It is not a probability "
    "and is never combined with the AI score.",
    "Metadata (EXIF, XMP, PNG text) is easy to remove, copy or forge. Its absence is common for "
    "legitimate images and its presence does not prove origin.",
    "Error Level Analysis is a visual aid that requires expert interpretation and is not scored.",
    "This application does not validate C2PA / Content Credentials signatures.",
]

CUSTODY_NOTE = ("Hashing begins when this application receives the uploaded file. The SHA-256 value "
                "identifies the exact bytes received; it does not establish where the file came "
                "from or what happened to it before upload. This application does not provide "
                "legally compliant chain-of-custody records.")


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_report(*, report_id, original_filename, received_at, file_info, metadata, identity,
                 ai, heuristics_result, consistency_result, detector_status):
    return {
        "report_id": report_id,
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso(datetime.now(timezone.utc)),
        "application": {"name": APP_NAME, "version": APP_VERSION},
        "assessment_type": ASSESSMENT_TYPE,
        "preliminary_assessment": consistency_result["summary"],
        "file": {
            "original_filename": original_filename,
            "received_at": iso(received_at),
            "size_bytes": metadata["file"]["size_bytes"],
            "detected_format": file_info["detected_format"],
            "mime_type": metadata["image"].get("mime_type"),
            "extension": file_info["extension"],
            "extension_matches_content": file_info["extension_matches_content"],
            "sha256": identity["sha256"],
            "md5": identity["md5"],
            "stored_copy_verified": identity["stored_copy_verified"],
            "previous_matches": identity["previous_matches"],
        },
        "ai_detection": ai,
        "ai_model": {k: detector_status.get(k) for k in
                     ("model_id", "revision", "commit_hash", "architecture", "labels",
                      "torch_version", "transformers_version", "known_limitations", "state", "error")},
        "heuristics": heuristics_result,
        "consistency": {"observations": consistency_result["observations"],
                        "recommended_checks": consistency_result["recommended_checks"]},
        "metadata": metadata,
        "limitations": LIMITATIONS,
        "custody_note": CUSTODY_NOTE,
    }


def to_json(report):
    return json.dumps(report, indent=2, ensure_ascii=False)


def score_of(entry):
    """Per-output score; reports created before v1.1 stored it under 'softmax'."""
    return entry.get("score", entry.get("softmax"))


def pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _kv_lines(d, indent="  "):
    lines = []
    for key, value in d.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{indent}{key}: {value}")
    return lines or [f"{indent}(none)"]


def to_txt(report):
    f, ai, heur = report["file"], report["ai_detection"], report["heuristics"]
    md = report["metadata"]
    out = []

    def section(title):
        out.extend(["", "=" * 78, title, "=" * 78])

    out += [APP_NAME.upper() + " - INTEGRITY REPORT",
            f"Report ID:       {report['report_id']}",
            f"Generated (UTC): {report['generated_at']}",
            f"Assessment type: {report['assessment_type']}"]

    section("1. PRELIMINARY ASSESSMENT")
    out += textwrap.wrap(report["preliminary_assessment"]["text"], 78)

    section("2. FILE IDENTITY")
    out += [f"Original filename:  {f['original_filename']}",
            f"Received at (UTC):  {f['received_at']}",
            f"Size:               {f['size_bytes']:,} bytes",
            f"Detected format:    {f['detected_format']} ({f['mime_type']})",
            f"Extension matches:  {'yes' if f['extension_matches_content'] else 'NO'} (.{f['extension']})",
            f"SHA-256:            {f['sha256']}",
            f"MD5 (reference):    {f['md5']}",
            f"Stored copy check:  {'hash verified' if f['stored_copy_verified'] else 'MISMATCH'}"]
    for m in f["previous_matches"]:
        out.append(f"Identical file seen: report {m['id']} ({m['original_filename']}, {m['uploaded_at']})")

    section("3. AI-GENERATED IMAGE DETECTION")
    model = report["ai_model"]
    # Name the model that produced this stored result (falls back to the detector status).
    out += [f"Model:     {ai.get('model_id') or model.get('model_id')}",
            f"Revision:  {ai.get('commit_hash') or ai.get('revision') or model.get('commit_hash') or model.get('revision')}",
            f"Status:    {ai['status']}"]
    if ai["status"] == "ok":
        out += [f"Prediction:          {ai['prediction_label']}",
                f"AI-generated score:  {pct(ai.get('ai_score'))}",
                f"Real-image score:    {pct(ai.get('real_score'))}",
                "Model confidence:    Not independently calibrated (only class scores are available)",
                f"Decision rule:       {ai['decision_rule']}",
                f"Output mapping:      {ai.get('output_mapping', 'softmax over the model classes')}",
                f"Raw model output ({ai.get('output_type', 'softmax')}):"]
        for s in ai["scores"]:
            out.append(f"  output {s['index']} '{s['label']}': logit={s['logit']:.4f}  "
                       f"score={score_of(s):.6f}")
        out.append(f"Inference time: {ai['inference_ms']} ms; model input {ai.get('model_input_size')}")
        out += textwrap.wrap("Note: " + ai["score_note"], 78)
    else:
        out.append("AI analysis unavailable. No AI prediction was produced.")
        out += textwrap.wrap("Reason: " + str(ai.get("message")), 78)
    out.append("Known model limitations:")
    for item in ai.get("known_limitations") or model.get("known_limitations") or []:
        out += textwrap.wrap(item, 76, initial_indent="- ", subsequent_indent="  ")

    section("4. HEURISTIC FINDINGS")
    out.append(f"Heuristic caution score: {heur['caution_score']}/100 ({heur['caution_band']}) - "
               "rule-weight sum, not a probability")
    for item in heur["findings"]:
        tag = f"+{item['weight']}" if item["weight"] else "info"
        out.append(f"[{tag:>4}] {item['title']}")
        out += textwrap.wrap(item["detail"], 78, initial_indent="       ", subsequent_indent="       ")

    section("5. EVIDENCE CONSISTENCY OBSERVATIONS")
    for o in report["consistency"]["observations"]:
        out.append(f"[{o['kind']}] {o['title']}")
        out += textwrap.wrap(o["detail"], 78, initial_indent="    ", subsequent_indent="    ")

    section("6. RECOMMENDED NEXT STEPS")
    for i, step in enumerate(report["consistency"]["recommended_checks"], 1):
        out += textwrap.wrap(step, 78, initial_indent=f"{i:>2}. ", subsequent_indent="    ")

    section("7. METADATA")
    out.append("Image properties:")
    out += _kv_lines(md["image"])
    out.append("Key fields:")
    out += _kv_lines(md["key_fields"])
    out.append(f"EXIF (IFD0) - {len(md['exif'])} tags:")
    out += _kv_lines(md["exif"])
    out.append(f"EXIF sub-IFD - {len(md['exif_sub_ifd'])} tags:")
    out += _kv_lines(md["exif_sub_ifd"])
    out.append(f"GPS - {len(md['gps'])} tags:")
    out += _kv_lines(md["gps"])
    out.append("Embedded text chunks:")
    out += _kv_lines(md["embedded_text"])
    out.append("Containers:")
    out += _kv_lines(md["containers"])
    if md["errors"]:
        out.append("Extraction errors: " + "; ".join(md["errors"]))

    section("8. LIMITATIONS")
    for item in report["limitations"]:
        out += textwrap.wrap(item, 78, initial_indent="- ", subsequent_indent="  ")

    section("9. CUSTODY NOTE")
    out += textwrap.wrap(report["custody_note"], 78)
    out.append("")
    return "\n".join(out)


def to_pdf(report):
    """Render the TXT report as a minimal A4 PDF (built-in Courier fonts, no extra dependency)."""
    lines = []
    for line in to_txt(report).split("\n"):
        safe = line.encode("latin-1", "replace").decode("latin-1")
        bold = bool(re.match(r"^\d\. [A-Z]", safe)) or safe.startswith(APP_NAME.upper())
        for chunk in textwrap.wrap(safe, 100, replace_whitespace=False, drop_whitespace=False) or [""]:
            lines.append((chunk, bold))

    per_page, leading, top = 74, 10.5, 800
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[]]

    def esc(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = [None,
               "<< /Type /Catalog /Pages 2 0 R >>",
               None,  # page tree, filled in below
               "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>"]
    page_ids = []
    for n, page in enumerate(pages, 1):
        ops = ["BT", f"{leading} TL", f"40 {top} Td"]
        for text, bold in page:
            ops.append(f"/{'F2' if bold else 'F1'} 8.5 Tf ({esc(text)}) Tj T*")
        ops += ["/F1 7 Tf", f"1 0 0 1 40 30 Tm (Report {esc(report['report_id'])} - page {n} of {len(pages)}) Tj", "ET"]
        stream = "\n".join(ops).encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objects) - 1
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                       f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>")
        page_ids.append(len(objects) - 1)
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>"
    title = esc(f"Integrity report {report['report_id']}")
    objects.append(f"<< /Title ({title}) /Producer ({APP_NAME} {APP_VERSION}) >>")
    info_id = len(objects) - 1

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for num, obj in enumerate(objects[1:], 1):
        offsets.append(len(out))
        body = obj if isinstance(obj, bytes) else obj.encode("latin-1")
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % len(objects)
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects), info_id, xref))
    return bytes(out)
