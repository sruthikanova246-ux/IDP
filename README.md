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
(about 330 MB, cached in `%USERPROFILE%\.cache\huggingface`) and generates demo images in `samples\`.

```bat
run_windows.bat
```

This starts the server on **http://127.0.0.1:5000/** (localhost only) and opens your browser.

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

Optional – rebuild the labelled evaluation set (about 300 MB from Wikimedia Commons) and re-run the model comparison:

```bat
.venv\Scripts\python tests\eval_data\build_eval_set.py
.venv\Scripts\python -m evaluation.compare_models
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

**`Smogy/SMOGY-Ai-images-detector`** (Swin Transformer, 86.7M parameters), pinned to commit
`fb9aadb85642d707f05b40191689d60f75b9a3e6`. Chosen in October 2026 by a measured comparison of eight pretrained
detectors (see [Evaluation](#evaluation)); it replaced `Ateeqq/ai-vs-human-image-detector`, which labelled real
photographs as AI-generated.

From its model card and config:

- **Labels:** `0 = "artificial"`, `1 = "human"`. AI-generated score = softmax of `artificial`; real-image score = softmax of `human`.
- **Training:** fine-tuned from `Organika/sdxl-detector` on about 50,000 images from Reddit, Kaggle and public-domain art.
  The card reports 98.2% on its own test split and 76–91% on generator families outside its training data.
- **License: CC BY-NC 4.0 – non-commercial use only.** Acceptable for this college project; do not use it commercially.
  Its ancestor `umm-maybe/AI-image-detector` states CC BY-ND 4.0 on its card, so the licensing of the derivative chain is
  uncertain. If you need a permissive license, the best Apache-2.0 option on validation was
  `haywoodsloan/ai-image-detector-deploy` (set `DEI_MODEL_ID` / `DEI_MODEL_REVISION`; its test accuracy was not measured).

How the app uses it (`ai_detector.py`):

- Loaded **once** per process (background thread at startup, guarded by a lock) and reused.
- EXIF orientation applied, transparency flattened on white, first frame of animations; then the model's own image
  processor (224×224). The Pillow preprocessing backend is pinned so results do not change if `torchvision` is installed.
- The AI score is always read from the label that **means** AI (by name, never by position). Models with one "real" label
  and several generator labels use 1 − P(real). Single-logit models are only interpreted when a reviewed card documents
  the output; anything else is shown as raw output with no AI score.
- Reports: AI-generated score, Real-image score (they sum to 100%), raw logits and scores, prediction label, model ID,
  revision, inference time.
- **Prediction rule:** AI-generated score above 50% → *Likely AI-generated*, otherwise *Likely real*. The app supports an
  *Inconclusive* band as a display rule, but none is used: on the validation split no band reduced wrong calls, because
  every score from this model was below 5% or above 95%.
- **Model confidence** is shown as *"Not independently calibrated"*: no separate confidence value is invented, and the
  AI score is not relabelled as confidence.
- If torch/transformers are missing, the weights cannot be downloaded, or inference throws, every page and report says
  **"AI analysis unavailable"** and no scores are stored (`ai_score` and `real_score` are `NULL`). Heuristics are never
  substituted.

### Observed behaviour on non-photographic inputs (measured with the current model)

| Input | AI-generated score | Prediction |
|---|---|---|
| `samples/*` (four scenes drawn with Pillow) | 0.00% | Likely real |
| Flat grey 256×256 image | 0.00% | Likely real |
| Uniform random noise 640×480 | 18.58% | Likely real |

These images are computer-made but not produced by an image generator. The model is strongly biased towards "real": it
fixed the false positives on photographs, but it misses about one AI image in four and should not be read as evidence
that an image is camera-original. With `samples/generator_tags.png` the consistency check therefore reports that the
metadata names a generator while the classifier does not.

## Evaluation

**Root cause of the original problem.** Not a bug. The labels were checked (`0 = ai`, `1 = hum`, read by name), and the
preprocessing matched the model card (224×224 bilinear resize, mean/std 0.5, no crop). On the validation photos, changing the
preprocessing (no EXIF rotation, or downscaling first) moved at most 1 of 20 results. The previous model itself was
unreliable on real photographs, and its errors depended on the camera: Fujifilm, Google Pixel and Sony photos scored about 99% "AI",
while Nikon, Olympus, iPhone and Xiaomi photos scored about 0%.

**Dataset** (`tests/eval_data/labels.csv`, built 2026-10-21 by `tests/eval_data/build_eval_set.py`, all from Wikimedia
Commons with recorded source page, author, license and checksums):

- 30 real camera/phone photographs: 3 each from 10 devices (Canon EOS 5D Mark II, Nikon D7000, Sony ILCE-7, Fujifilm X-T1,
  Olympus E-M5, iPhone 11 Pro, Samsung Galaxy S9, Google Pixel, Huawei P20 Pro, Xiaomi Redmi Note 7), taken from the
  "Taken with …" categories, captured before June 2022 and uploaded before 2023.
- 30 stress copies of those photos, made locally and labelled as simulations: WhatsApp-style recompression
  (1600 px, JPEG q60, metadata stripped), phone-screenshot framing (PNG 1080×2340), and plain resizing (1024 px, q85).
- 45 AI-generated images from 10 generators (Stable Diffusion, Midjourney, DALL-E 3, Bing Image Creator, ChatGPT,
  Gemini, Flux, Ideogram, Adobe Firefly, Imagen), taken from the "Images generated by …" categories.
- Images were chosen by fixed rules and title-hash order, without looking at the images or at any model's scores.
  Split per camera/generator: **validation 20 real / 18 AI**, **test 40 real / 27 AI** (a photo and its stress copy stay in
  the same split). The images are not committed to git; `python tests/eval_data/build_eval_set.py` downloads them and
  verifies the checksums.

**Procedure** (`python -m evaluation.compare_models`, rules written before any results were seen):
eight candidates were scored on the validation split only; the model and the inconclusive band were chosen there; the
test split was then scored once for the chosen model and, as the "before" figure, for the previous model.
"Inconclusive" always counts as a non-answer.

**Validation split (used for the choice):**

| Model (pinned) | License | Real acc. | AI acc. | Balanced | FPR on real |
|---|---|---|---|---|---|
| `Ateeqq/ai-vs-human-image-detector` (previous) | Apache-2.0 | 60.0% | 77.8% | 68.9% | 40.0% |
| `buildborderless/CommunityForensics-DeepfakeDet-ViT` | MIT | 100% | 33.3% | 66.7% | 0% |
| `haywoodsloan/ai-image-detector-deploy` | Apache-2.0 | 90.0% | 72.2% | 81.1% | 10.0% |
| `umm-maybe/AI-image-detector` | CC BY / BY-ND | 85.0% | 38.9% | 62.0% | 15.0% |
| `Organika/sdxl-detector` | CC BY-NC 3.0 | 75.0% | 55.6% | 65.3% | 25.0% |
| **`Smogy/SMOGY-Ai-images-detector`** | CC BY-NC 4.0 | **95.0%** | **72.2%** | **83.6%** | **5.0%** |
| `jacoballessio/ai-image-detect-distilled` | MIT | 75.0% | 38.9% | 57.0% | 25.0% |
| `NYUAD-ComNets/NYUAD_AI-generated_images_detector` | Apache-2.0 | 80.0% | 50.0% | 65.0% | 20.0% |

Why the others were rejected: Community Forensics never flagged a real photo but caught only 6 of 18 AI images;
haywoodsloan was second (one more false positive and lower balanced accuracy); the umm-maybe/Organika family,
jacoballessio and NYUAD caught half or fewer of the AI images, or flagged 15–25% of real photos.

**Test split (scored once); target: balanced accuracy ≥ 70% and real-photo accuracy ≥ 70%:**

| Group | Real photos correct | AI images correct | Balanced acc. | FPR | Inconclusive |
|---|---|---|---|---|---|
| **Smogy, all test images** | **40/40 (100%)** | **20/27 (74.1%)** | **87.0%** | **0%** | 0% |
| Smogy, camera originals + AI | 20/20 (100%) | 20/27 (74.1%) | 87.0% | 0% | 0% |
| Smogy, stress group (real only) | 20/20 (100%) | – | – | 0% | 0% |
| Previous model, all test images | 37/40 (92.5%) | 25/27 (92.6%) | 92.5% | 7.5% | 0% |
| Previous model, stress group | 18/20 (90.0%) | – | – | 10.0% | 0% |

**Target met** on the test split. With no inconclusive band, results with and without inconclusive images are the same.
95% Wilson intervals: real-photo accuracy 91–100% (40/40), AI-image accuracy 55–87% (20/27).

Read these figures with care:

- The previous model scored *better* on the test split than on validation, where it called 8 of 20 real photos AI.
  Across all 60 real photos it flagged 11 (18%); Smogy flagged 1 (1.7%). The model choice was fixed on validation
  before the test split was scored and was not changed afterwards. The gap between the two splits shows how much
  results vary with samples this small.
- Smogy missed 7 of 27 AI test images, including both Adobe Firefly images (scored 0.0%).
- The set is small (one test photo ≈ 2.5 percentage points) and comes from one source. Accuracy may be lower on other
  cameras, other generators, heavily edited photos and real WhatsApp/screenshot files. It is also unknown whether any of
  these Commons images were in the model's training data.

Re-run: `python tests/eval_data/build_eval_set.py` then `python -m evaluation.compare_models`. Results with per-image
scores: `evaluation/results.json`. The regression test `tests/test_eval_fpr.py` fails if the configured model labels
more than **10%** of the 40 real test photos as AI-generated (measured: 0%); it is skipped when the images or the model
are not available.

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
evaluate_model.py Score any detector on the labelled set (accuracy, FPR, AUC, stress group)
evaluation/       candidates.py (reviewed models, pinned), compare_models.py (selection rules), results.json (measured)
tests/eval_data/  labels.csv (sources, licenses, splits, checksums) and build_eval_set.py; images/ is git-ignored
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

`pytest -v` on Windows 11, Python 3.14.7, torch 2.14.1+cpu, transformers 5.20.2, 2026-10-21: **31 passed, 0 skipped**
(the real-model tests and the evaluation-set test ran; they are skipped with a message only if the model or the
evaluation images are missing).

| Requirement | Test |
|---|---|
| Valid image upload, hash + file stored, DB record | `test_app.py::test_valid_upload_persists_and_hashes` |
| Invalid file rejection (non-image, wrong extension, empty, truncated) | `test_invalid_file_rejected`, `test_modules.py::test_validate_rejects_*` |
| Oversized file rejection (HTTP 413) | `test_oversized_file_rejected` |
| EXIF extraction / missing EXIF | `test_exif_extraction`, `test_missing_exif_handled` |
| SHA-256 correctness (NIST "abc" vector) and repeatability | `test_sha256_known_vector_and_repeatability` |
| Successful AI inference (real model): AI score read from the `artificial` label, scores sum to 1, deterministic, model ID + revision persisted, shown on page and in TXT/JSON | `test_ai.py::test_real_inference_outputs`, `test_real_inference_end_to_end` |
| AI/real mapping by label name (swapped order, multi-class, unreviewed sigmoid → no guess) and the inconclusive display rule | `test_label_mapping_rules`, `test_inconclusive_band_is_a_display_rule` |
| False-positive rate on the 40 real test photos ≤ 10% (measured 0%) | `test_eval_fpr.py::test_false_positive_rate_on_real_photos` |
| Model-unavailable fallback (no fake scores, metadata still works, "AI analysis unavailable" in page and TXT) | `test_model_unavailable_fallback` |
| TXT / JSON / PDF downloads (PDF structure checked) | `test_report_downloads` |
| Records saved by v1.0 (previous model, old score format) still open and download | `test_old_v1_0_records_still_open` |
| Persistence across restart, reopening an earlier report, search | `test_history_persists_across_restart` |
| Delete report and its stored + derived files | `test_delete_removes_record_and_files` |
| CSRF, path traversal / bad IDs, security headers | `test_csrf_required`, `test_bad_ids_and_pages` |
| Heuristic + consistency rules (editing, timeline, generator conflict, missing metadata wording, hash wording) | `test_modules.py` |

Also checked by hand on the running server: upload through the browser UI (preview, progress, redirect),
client- and server-side error messages, all downloads (PDF parsed by `pypdf` in strict mode), delete from History,
history still present after restarting the server, phone-width layout.

## Demonstration workflow

1. Run `run_windows.bat`; the dashboard shows "AI model ready" after loading (takes a few seconds from cache).
2. **Upload** `samples\camera_exif.jpg` → classifier *Likely real*; Photoshop tag (+15) and capture/modify time gap
   (+10) give caution score 25, so the assessment is *Further review recommended*. Open the detailed report to show the
   ELA image and the "Stored copy now: re-hashed" line.
3. Upload `samples\generator_tags.png` → generator signature in the PNG `parameters` chunk (+40) and generator-typical
   1024×1024 size (+5): caution score 50 (elevated). The classifier says *Likely real*, so the report shows
   *Conflicting evidence* and explains that detectors miss generators – the classifier never overrides the metadata clue.
4. Upload `samples\no_metadata.png` → shows that missing EXIF is explained as common, not as manipulation.
5. Upload `samples\renamed_png.jpg` → extension mismatch finding. Upload `samples\not_an_image.jpg` → rejected with a clear message.
6. Upload `camera_exif.jpg` again → "Identical file analysed before" (same SHA-256).
7. Download TXT / JSON / PDF. Open **History**, restart the server, show the records are still there; delete one.
8. Optional: stop the server, start it with `set DEI_MODEL_ID=not/a-model` (or with no internet and an empty cache),
   upload an image and show "AI analysis unavailable" with no scores.

## Limitations

- The classifier's scores are not calibrated probabilities and can be confidently wrong (see observed behaviour above).
  Measured on 2026-10-21 on a small Wikimedia Commons set (test split: 40 real photos incl. 20 simulated
  WhatsApp/screenshot/resized copies, 27 AI images from 10 generators): real-photo accuracy 100%, AI-image accuracy
  74.1%, balanced accuracy 87.0%, false-positive rate 0%. These figures come from a small set and may be lower on
  other kinds of images; about one AI image in four is missed (see [Evaluation](#evaluation)).
- The model sees a 224×224 version of the image; pixel-level artefacts are invisible to it.
- The AI model is licensed CC BY-NC 4.0 (non-commercial use only).
- Metadata can be stripped, copied or forged; heuristics are clues only.
- ELA requires expert interpretation and is not scored. C2PA manifests are only detected, not validated.
- **Chain of custody:** hashing starts when this application receives the file. It does not establish the file's
  history before upload, and the app does not provide legally compliant chain-of-custody records.
- Single-user local demo: no authentication, no multi-user access control.
- Windows long paths: if the project folder path is very long, some compiled Python packages can fail to load
  ("filename or extension is too long"). Keep the project in a short path such as `C:\projects\evidence`.
