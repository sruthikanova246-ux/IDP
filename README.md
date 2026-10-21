# Digital Evidence Integrity System

A local Flask web application that produces a **preliminary integrity assessment** of an uploaded
digital image. It combines:

1. A pretrained AI-generated-image classifier (real inference, real scores)
2. Metadata extraction (image properties, EXIF, GPS, PNG/WEBP text chunks, XMP/IPTC/C2PA markers)
3. SHA-256 fingerprinting of the exact bytes received
4. Explainable rule-based heuristic findings (plus Error Level Analysis for JPEGs)
5. Evidence consistency analysis that cross-checks all of the above
6. A stored, downloadable integrity report (TXT, JSON, PDF)

> **This tool does not verify authenticity.** No model or rule can prove that an image is real or fake.
> Every report says so, and the app never merges its findings into an "authenticity probability".

---

## Quick start (Windows)

Requirements: Windows 10/11, Python 3.10+ on PATH (tested with 3.14), about 1.5 GB free disk space,
internet access for the first setup only.

```bat
setup_windows.bat
```

This creates `.venv`, installs the dependencies (CPU-only PyTorch), downloads and smoke-tests the AI model
(about 370 MB, cached in `%USERPROFILE%\.cache\huggingface`) and generates demo images in `samples\`.

```bat
run_windows.bat
```

Manual equivalent:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python download_model.py
.venv\Scripts\python make_samples.py
.venv\Scripts\python app.py
```

Run the tests:

```bat
.venv\Scripts\python -m pytest -v
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DEI_SECRET_KEY` | random, saved to `instance/secret_key` | Flask session/CSRF signing key |
| `DEI_HOST` / `DEI_PORT` | `127.0.0.1` / `5000` | Bind address (keep localhost for the demo) |
| `DEI_MAX_UPLOAD_MB` | `20` | Upload size limit |
| `DEI_MAX_IMAGE_PIXELS` | `80000000` | Decompression-bomb limit |
| `DEI_AI_ENABLED` | `1` | Set `0` to run metadata/hash analysis only |
| `DEI_MODEL_ID` / `DEI_MODEL_REVISION` | see below | Hugging Face model and pinned commit |
| `DEI_MODEL_LOCAL_ONLY` | `0` | Set `1` to never use the network (cache only) |
| `DEI_MODEL_PRELOAD` | `1` | Load the model in the background at startup |
| `DEI_INSTANCE_DIR` | `instance/` | Where the database, uploads and logs are stored |

---

## Pages

| Page | URL | What it shows |
|---|---|---|
| Dashboard | `/` | Counts, AI model status (with retry if loading failed), recent analyses |
| Upload evidence | `/upload` | File picker, client-side validation, preview, upload progress and analysis wait state |
| Analysis results | `/analysis/<id>` | Preliminary assessment, AI scores with bars, caution score, key observations, downloads |
| Detailed report | `/report/<id>` | Everything: identity, live stored-copy re-hash, AI raw output, findings, ELA, observations, full metadata, limitations, delete |
| History | `/history` | All stored analyses, search by filename / SHA-256 / report ID, delete |
| Methodology | `/methodology` | How each step works, rule weights, model details and limitations |

---

## AI model

**`Ateeqq/ai-vs-human-image-detector`**, pinned to commit `60e82406916921b823616bee33397baab38af3f0`.

What was reviewed before choosing it (from its Hugging Face model card and config):

- **Architecture:** `SiglipForImageClassification` (SigLIP vision encoder, 92.9M parameters), input 224×224, normalisation mean/std 0.5.
- **Labels:** `0 = "ai"`, `1 = "hum"` (human). These map to the app's *AI-generated* and *Real-image* scores.
- **License:** Apache-2.0.
- **Training:** fine-tuned on about 60,000 AI-generated and 60,000 human images; the card reports 99.23% accuracy on the author's own held-out split.
- **Caveat stated on the card:** "Some users reported overfitting issues".

Alternatives examined: `haywoodsloan/ai-image-detector-deploy` (SwinV2, Apache-2.0, 780 MB, card contains only AutoTrain metrics),
`umm-maybe/AI-image-detector` (trained mainly on art, 2022-era generators), `Organika/sdxl-detector` (non-commercial license),
`dima806/ai_vs_real_image_detection` (CIFAKE 32×32 data, unsuitable for photographs).

How the app uses it (`ai_detector.py`):

- Loaded **once** per process (background thread at startup, guarded by a lock) and reused.
- EXIF orientation applied, transparency flattened on white, first frame of animations; then the model's own `AutoImageProcessor`.
- Reports: AI-generated score, Real-image score (softmax of the two classes, which always sum to 100%), raw logits,
  prediction label (`Likely AI-generated` / `Likely real` by argmax, i.e. >50%), model ID, revision, inference time.
- **Model confidence** is shown as *"Not independently calibrated"*: the model outputs only class scores, the card reports no
  calibration study, so no separate confidence value is invented, and the AI score is not relabelled as confidence.
- If torch/transformers are missing, the weights cannot be downloaded, or inference throws, the result is
  `unavailable` / `error` and every page and report says **"AI analysis unavailable"**. No scores are stored
  (`ai_score` and `real_score` are `NULL`), and heuristics are never substituted.

### Observed behaviour (real runs on this machine)

| Input | AI-generated score | Real-image score | Comment |
|---|---|---|---|
| `samples/*` (four scenes drawn with Pillow) | 99.92–99.94% | 0.06–0.08% | They are computer-generated graphics, so "not a camera photo" is fair, but they are not diffusion output |
| Flat grey 256×256 image | 99.88% | 0.12% | No content at all, yet a near-certain score |
| Browser-canvas gradient with random noise (JPEG) | 0.2% | 99.8% | Synthetic, yet scored "Likely real" with high certainty |

These are a direct demonstration of why the scores must be read as **model outputs, not probabilities**: on inputs
unlike its training data the model is confidently wrong in either direction. The included samples test the
*workflow*; they cannot be used to measure accuracy. Evaluating accuracy would need a labelled set of real
photographs and images from current generators.

---

## Heuristic caution score

A transparent sum of rule weights (capped at 100), shown **separately** from the AI score and never combined with it.

| Rule | Weight |
|---|---|
| AI-generator signature in metadata (Software tag or PNG `parameters`/`prompt`/`workflow` chunks) | +40 |
| Known editing software in the Software tag | +15 |
| EXIF DateTime later than DateTimeOriginal | +10 |
| File extension does not match decoded format | +10 |
| No EXIF / no camera make & model | +5 |
| Both sides are typical generator resolutions and no camera tags | +5 |
| Estimated JPEG quality below 70 | +5 |

Bands: low < 15, moderate 15–39, elevated ≥ 40. Informational findings (GPS present, XMP present, possible C2PA marker,
JPEG quality estimate, ELA statistics) carry no weight. Missing metadata only adds +5 and is always explained as common
for legitimate images.

## Evidence consistency analysis

`consistency.py` applies readable rules, for example:

- Classifier says AI **and** metadata has a generator signature → "both point to generation", still not conclusive.
- Classifier says AI **but** EXIF names a camera → conflict reported both ways (EXIF can be copied; detectors misfire on processed photos).
- Metadata has a generator signature **but** classifier says real → conflict; detectors miss unseen generators.
- AI unavailable → explicit warning that absence of a flag means nothing.
- Borderline score (35–65%), image smaller than the model input, heavy compression, transparency/animation → reliability notes.
- File identity: hash scope, stored-copy verification, and earlier reports with the same SHA-256.

The result is a qualitative label (`No strong indicators`, `Further review recommended`, `Conflicting evidence`,
`Limited (no AI result)`) plus recommended follow-up checks.

---

## Architecture

```
app.py            Flask app factory, routes, CSRF, security headers, error handlers
analyzer.py       Pipeline orchestration: validate -> store -> hash -> metadata -> heuristics -> AI -> consistency -> report -> DB
ai_detector.py    Model loading (once), preprocessing, inference, failure reporting
integrity.py      SHA-256/MD5, extension/size/decode validation
metadata.py       Image properties, EXIF/GPS, text chunks, container markers
heuristics.py     Rule-based findings, caution score, JPEG quality estimate, Error Level Analysis
consistency.py    Cross-checking rules, observations, recommended checks, qualitative summary
reports.py        Report structure, TXT / JSON / PDF (PDF is written directly, no extra library)
database.py       SQLite schema and parameterised queries
config.py         Settings and environment overrides
download_model.py Pre-download and smoke-test the model
make_samples.py   Reproducible demo images
templates/        Jinja2 pages;  static/  CSS and JS (no external CDNs)
tests/            pytest suite
instance/         Runtime data (created automatically, git-ignored): evidence.db, uploads/, derived/, app.log, secret_key
```

SQLite table `analyses` stores: report ID, original filename, stored filename, upload timestamp, SHA-256, size, MIME type,
AI status, AI label, AI-generated score, real-image score, prediction label, model ID and revision, caution score,
summary level, and the full metadata / analysis / report as JSON.

## Security measures

- Server bound to `127.0.0.1` by default; Flask debug mode off.
- Extension allow-list, size limit (`MAX_CONTENT_LENGTH` → HTTP 413), full Pillow decode, decompression-bomb pixel limit,
  decoded format must be an allowed image format.
- Uploads stored under random names (`DEI-YYYYMMDD-<12 hex>.<ext>`) in `instance/uploads`; files are served only by
  validated report ID, never by user-supplied path; deletion refuses paths outside the storage folders.
- CSRF token on every POST (upload, delete, model retry); session cookie `HttpOnly`, `SameSite=Lax`.
- Content-Security-Policy (`default-src 'self'`, no inline scripts/styles), `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`. Jinja2 auto-escaping for all metadata shown.
- All SQL uses bound parameters (including `LIKE` search with escaping).
- No API keys or credentials anywhere; the model is a public download. Errors are logged to `instance/app.log`;
  users see generic messages.

## Test results

`pytest -v` on Windows 11, Python 3.14.7, torch 2.14.1+cpu, transformers 5.20.2: **27 passed** (the real-model tests
ran; they are skipped only if the model has not been downloaded).

| Requirement | Test |
|---|---|
| Valid image upload, hash + file stored, DB record | `test_app.py::test_valid_upload_persists_and_hashes` |
| Invalid file rejection (non-image, wrong extension, empty, truncated) | `test_invalid_file_rejected`, `test_modules.py::test_validate_rejects_*` |
| Oversized file rejection (HTTP 413) | `test_oversized_file_rejected` |
| EXIF extraction / missing EXIF | `test_exif_extraction`, `test_missing_exif_handled` |
| SHA-256 correctness (NIST "abc" vector) and repeatability | `test_sha256_known_vector_and_repeatability` |
| Successful AI inference (real model): labels, scores sum to 1, deterministic, persisted, shown on page and in TXT/JSON | `test_ai.py::test_real_inference_outputs`, `test_real_inference_end_to_end` |
| Model-unavailable fallback (no fake scores, metadata still works, "AI analysis unavailable" in page and TXT) | `test_model_unavailable_fallback` |
| TXT / JSON / PDF downloads (PDF structure checked) | `test_report_downloads` |
| Persistence across restart, reopening an earlier report, search | `test_history_persists_across_restart` |
| Delete report and its stored + derived files | `test_delete_removes_record_and_files` |
| CSRF, path traversal / bad IDs, security headers | `test_csrf_required`, `test_bad_ids_and_pages` |
| Heuristic + consistency rules (editing, timeline, generator conflict, missing metadata wording, hash wording) | `test_modules.py` |

Also checked by hand on the running server: upload through the browser UI (preview, progress, redirect),
client- and server-side error messages, all downloads (PDF parsed by `pypdf` in strict mode), delete from History,
history still present after restarting the server, phone-width layout.

## Demonstration workflow

1. Run `run_windows.bat`; the dashboard shows "AI model ready" after loading (takes a few seconds from cache).
2. **Upload** `samples\camera_exif.jpg` → *Conflicting evidence*: the classifier flags it while EXIF names a Canon camera;
   Photoshop tag (+15) and capture/modify time gap (+10) give caution score 25. Open the detailed report to show the ELA image
   and the "Stored copy now: re-hashed" line.
3. Upload `samples\generator_tags.png` → generator signature found in the PNG `parameters` chunk (+40) and generator-typical
   1024×1024 size (+5): caution score 50 (elevated), and both lines of evidence agree.
4. Upload `samples\no_metadata.png` → shows that missing EXIF is explained as common, not as manipulation.
5. Upload `samples\renamed_png.jpg` → extension mismatch finding. Upload `samples\not_an_image.jpg` → rejected with a clear message.
6. Upload `camera_exif.jpg` again → "Identical file analysed before" (same SHA-256).
7. Download TXT / JSON / PDF. Open **History**, restart the server, show the records are still there; delete one.
8. Optional: stop the server, start it with `set DEI_MODEL_ID=not/a-model` (or with no internet and an empty cache),
   upload an image and show "AI analysis unavailable" with no scores.

## Limitations

- The classifier's scores are not calibrated probabilities and can be confidently wrong (see observed behaviour above).
  Its accuracy on real-world photographs and current generators was not measured in this project.
- The model sees a 224×224 version of the image; pixel-level artefacts are invisible to it.
- Metadata can be stripped, copied or forged; heuristics are clues only.
- ELA requires expert interpretation and is not scored. C2PA manifests are only detected, not validated.
- **Chain of custody:** hashing starts when this application receives the file. It does not establish the file's
  history before upload, and the app does not provide legally compliant chain-of-custody records.
- Single-user local demo: no authentication, no multi-user access control.
- Windows long paths: if the project folder path is very long, some compiled Python packages can fail to load
  ("filename or extension is too long"). Keep the project in a short path such as `C:\projects\evidence`.
