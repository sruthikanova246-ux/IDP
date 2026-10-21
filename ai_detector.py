"""Pretrained AI-generated-image classifier (Hugging Face transformers).

The model is loaded once per process and reused. If the libraries, weights or
network are unavailable, every call returns an explicit "unavailable" result;
no prediction is ever fabricated or replaced by a heuristic.

How a model's raw output becomes an "AI-generated score" depends on the model:
  * softmax models with one AI label and one real label: the softmax of the AI label;
  * softmax models with one real label and several generator labels: 1 - P(real);
  * single-logit (sigmoid) models: only when a reviewed model card documents that the
    sigmoid is the probability of "fake" (see REVIEWED_MODELS).
Anything else is reported verbatim as "unmapped" without an AI score.
"""
import logging
import threading
import time

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# Label names (lower-cased) we know how to interpret.
AI_LABELS = {"ai", "artificial", "fake", "generated", "ai-generated", "ai_generated", "synthetic"}
HUMAN_LABELS = {"hum", "human", "real", "authentic", "natural", "photo"}

RETRY_AFTER_SECONDS = 60

# Models whose cards, licenses and output semantics were reviewed for this application.
# Measured results on the project's evaluation set are in evaluation/results.json.
REVIEWED_MODELS = {
    "Smogy/SMOGY-Ai-images-detector": {
        "revision": "fb9aadb85642d707f05b40191689d60f75b9a3e6",
        "output": "labels",
        "display_name": "SMOGY AI-images detector (Swin, fine-tuned from Organika/sdxl-detector)",
        "license": "CC BY-NC 4.0 (non-commercial use only)",
        "band": None,  # no inconclusive band improved validation results (evaluation/results.json)
        "limitations": [
            "Measured on this project's small evaluation set (Wikimedia Commons, October 2026): test split "
            "real-photo accuracy 40/40, AI-image accuracy 20/27 (74%), balanced accuracy 87%. It misses about "
            "one AI image in four, and missed both Adobe Firefly test images.",
            "Fine-tuned (per its model card) on about 50,000 images scraped from Reddit and Kaggle plus "
            "public-domain art; the card reports 76-91% accuracy on generators outside its training data.",
            "License: CC BY-NC 4.0, non-commercial use only. Its base model's card (umm-maybe/AI-image-detector) "
            "states CC BY-ND 4.0, so the licensing of this derivative is uncertain; check before any reuse.",
            "Input is resized to 224x224, so small pixel-level artefacts are not visible to the model.",
            "On all 105 evaluation images its score was below 5% or above 95%, including the images it got "
            "wrong (e.g. 0.0% for both missed Firefly images). The scores are not calibrated probabilities.",
        ],
    },
    "buildborderless/CommunityForensics-DeepfakeDet-ViT": {
        "revision": "ac6ee457bea904a373065754107451793b56db00",
        "output": "sigmoid_fake",
        "display_name": "Community Forensics ViT-S (Park & Owens, CVPR 2025)",
        "license": "MIT",
        "limitations": [
            "ViT-Small trained by the paper authors (Park & Owens, University of Michigan) on 2.7M images "
            "from 4,803 generators; this copy is a third-party upload ('buildborderless') of their checkpoint. "
            "The repository's config and weights were corrected in July 2026; the pinned revision is after that fix.",
            "Single sigmoid output documented on the model card as the probability of 'fake'. It is a model "
            "score, not a calibrated probability, and no calibration study is reported.",
            "The image is resized (shortest edge 440) and centre-cropped to 384x384: content near the "
            "edges of non-square images is not seen, and fine pixel detail is reduced.",
        ],
    },
    "Ateeqq/ai-vs-human-image-detector": {
        "revision": "60e82406916921b823616bee33397baab38af3f0",
        "output": "labels",
        "display_name": "SigLIP ai-vs-human detector (Ateeqq)",
        "license": "Apache-2.0",
        "limitations": [
            "Fine-tuned on about 60,000 AI-generated and 60,000 human images (per the model card); the "
            "exact generators and photo sources are not documented.",
            "The model card notes that some users reported overfitting. On this project's evaluation set it "
            "labelled 8 of 20 validation real photographs as AI-generated (camera-dependent: Nikon D5200, Google "
            "Pixel and Sony ILCE-7 photos scored about 99%), and 3 of 40 on the test split.",
            "Input is squashed to 224x224, so small pixel-level artefacts are not visible to the model.",
        ],
    },
}

SCORE_NOTE = ("The AI-generated score is this model's output for the 'AI-generated' class. It is not a "
              "calibrated probability: it reflects how strongly the model's learned features match its "
              "training data, and it can be confidently wrong on images unlike that data (new generators, "
              "heavy compression, screenshots, resized or edited photos).")
CONFIDENCE_NOTE = ("Not independently calibrated. The AI-generated and real-image scores are two views of "
                   "the same model output (they sum to 100%); the model card reports no calibration study.")


def decision_rule_text(band):
    if band:
        lo, hi = band
        return (f"AI-generated score above {hi * 100:.0f}%: 'Likely AI-generated'; below {lo * 100:.0f}%: "
                f"'Likely real'; from {lo * 100:.0f}% to {hi * 100:.0f}%: 'Inconclusive'. The band is a display "
                "rule chosen on a validation split, not a calibrated confidence.")
    return "AI-generated score above 50%: 'Likely AI-generated'; otherwise 'Likely real'."


def decide(ai_score, band=None):
    """Map an AI score to 'ai', 'real' or 'inconclusive' using the documented display rule."""
    if band and band[0] <= ai_score <= band[1]:
        return "inconclusive"
    return "ai" if ai_score > 0.5 else "real"


PREDICTION_LABELS = {"ai": "Likely AI-generated", "real": "Likely real", "inconclusive": "Inconclusive"}


def prepare_image(img):
    """Convert to the RGB input the model expects (orientation applied, alpha flattened on white)."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


class AIDetector:
    def __init__(self, model_id, revision=None, enabled=True, local_files_only=False, band=None):
        self.model_id = model_id
        self.revision = revision
        self.enabled = enabled
        self.local_files_only = local_files_only
        self.reviewed = REVIEWED_MODELS.get(model_id)
        # The band is a display rule chosen on the validation split for a reviewed model.
        band = band if band is not None else (self.reviewed or {}).get("band")
        self.band = tuple(band) if band else None
        self._lock = threading.RLock()
        self._model = None
        self._processor = None
        self._torch = None
        self.state = "disabled" if not enabled else "not_loaded"
        self.error = None
        self._failed_at = 0.0
        self.info = {}

    # ------------------------------------------------------------------ status
    def known_limitations(self):
        if self.reviewed:
            return self.reviewed["limitations"]
        return ["This model was configured by the user and has not been reviewed for this "
                "application; consult its model card for training data and limitations."]

    def status(self):
        return {
            "enabled": self.enabled,
            "state": self.state,
            "model_id": self.model_id,
            "display_name": (self.reviewed or {}).get("display_name", self.model_id),
            "license": (self.reviewed or {}).get("license"),
            "revision": self.revision,
            "error": self.error,
            "known_limitations": self.known_limitations(),
            "decision_rule": decision_rule_text(self.band),
            "band": self.band,
            **self.info,
        }

    # ----------------------------------------------------------------- loading
    def start_background_load(self):
        if self.enabled and self.state == "not_loaded":
            threading.Thread(target=self.load, name="model-loader", daemon=True).start()

    def request_load(self):
        """Start (or retry after a failure) loading in the background."""
        with self._lock:
            if self.enabled and self.state == "failed":
                self.state, self._failed_at = "not_loaded", 0.0
        self.start_background_load()

    def load(self):
        if not self.enabled:
            return False
        with self._lock:
            if self.state == "ready":
                return True
            if self.state == "failed" and time.time() - self._failed_at < RETRY_AFTER_SECONDS:
                return False
            self.state = "loading"
            self.error = None
            started = time.time()
            try:
                import torch
                import transformers
                from transformers import AutoImageProcessor, AutoModelForImageClassification
            except ImportError as exc:
                return self._fail(f"Required ML library is not installed ({exc.name}). "
                                  "Install the packages in requirements.txt.")
            try:
                kwargs = {"revision": self.revision, "local_files_only": self.local_files_only}
                processor = _load_processor(AutoImageProcessor, self.model_id, kwargs)
                model = AutoModelForImageClassification.from_pretrained(self.model_id, **kwargs)
                model.eval()
            except OSError as exc:
                return self._fail("Model weights could not be loaded. If this is the first run, the "
                                  "computer needs internet access to download them (or run "
                                  f"download_model.py). Details: {str(exc)[:300]}")
            except Exception as exc:
                return self._fail(f"Model failed to load: {type(exc).__name__}: {str(exc)[:300]}")

            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            mapping = _label_mapping(id2label, model.config.num_labels, self.reviewed)

            self._torch = torch
            self._processor = processor
            self._model = model
            self.info = {
                "architecture": (model.config.architectures or [type(model).__name__])[0],
                "model_type": model.config.model_type,
                "commit_hash": getattr(model.config, "_commit_hash", None) or self.revision,
                "labels": id2label,
                "output_mapping": mapping,
                "processor": type(processor).__name__,
                "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "load_seconds": round(time.time() - started, 1),
            }
            self.state = "ready"
            log.info("AI model %s loaded in %.1fs", self.model_id, self.info["load_seconds"])
            return True

    def _fail(self, message):
        self.state = "failed"
        self.error = message
        self._failed_at = time.time()
        log.error("AI model unavailable: %s", message)
        return False

    # --------------------------------------------------------------- inference
    def predict(self, img):
        """Run the classifier on a PIL image. Always returns a dict with a 'status' key."""
        base = {"model_id": self.model_id, "revision": self.revision}
        if not self.enabled:
            return {**base, "status": "disabled",
                    "message": "AI detection is disabled in the configuration (DEI_AI_ENABLED=0)."}
        if not self.load():
            return {**base, "status": "unavailable",
                    "message": self.error or "The AI model is not available."}

        try:
            with self._lock:
                rgb = prepare_image(img)
                started = time.time()
                inputs = self._processor(images=rgb, return_tensors="pt")
                with self._torch.no_grad():
                    logits = self._model(**inputs).logits[0].float()
                mapping = self.info["output_mapping"]
                probs = (self._torch.sigmoid(logits) if mapping["kind"] == "sigmoid"
                         else self._torch.softmax(logits, dim=-1))
                elapsed_ms = round((time.time() - started) * 1000, 1)
        except Exception as exc:
            log.exception("AI inference failed")
            return {**base, "status": "error",
                    "message": f"Inference failed: {type(exc).__name__}: {str(exc)[:300]}"}

        labels = self.info["labels"]
        scores = [{"index": i, "label": labels.get(i, str(i)), "logit": round(float(logits[i]), 6),
                   "score": round(float(probs[i]), 6)} for i in range(len(probs))]
        result = {
            **base,
            "status": "ok",
            "commit_hash": self.info["commit_hash"],
            "architecture": self.info["architecture"],
            "display_name": (self.reviewed or {}).get("display_name", self.model_id),
            "output_type": mapping["kind"],
            "output_mapping": mapping["description"],
            "scores": scores,
            "inference_ms": elapsed_ms,
            "model_input_size": _input_size(self._processor),
            "score_note": SCORE_NOTE,
            "decision_rule": decision_rule_text(self.band),
            "inconclusive_band": list(self.band) if self.band else None,
            "known_limitations": self.known_limitations(),
            # No calibrated confidence exists; model certainty (top_score) is the model's own score.
            "confidence": None,
            "confidence_note": CONFIDENCE_NOTE,
        }
        if mapping["kind"] == "unmapped":
            top = max(scores, key=lambda s: s["score"])
            result.update(ai_score=None, real_score=None, predicted_class="unmapped",
                          prediction_label=f"Raw label: {top['label']}",
                          message="The model's outputs could not be mapped to AI/real classes; raw outputs "
                                  "are shown without interpretation.")
            return result

        if mapping["kind"] == "sigmoid":
            ai_score = float(probs[0])
        else:
            ai_score = sum(float(probs[i]) for i in mapping["ai_indices"])
        ai_score = round(ai_score, 6)
        call = decide(ai_score, self.band)
        result.update(ai_score=ai_score, real_score=round(1 - ai_score, 6),
                      predicted_class={"ai": "ai_generated", "real": "not_ai_generated"}.get(call, "inconclusive"),
                      prediction_label=PREDICTION_LABELS[call])
        if mapping["kind"] == "softmax":
            # Model certainty: the highest class softmax score, exactly as the model produced it.
            top = max(scores, key=lambda s: s["score"])
            result.update(top_score=top["score"], top_label=top["label"])
        return result


def _load_processor(auto_cls, model_id, kwargs):
    """Pin the Pillow preprocessing backend so results do not change if torchvision gets installed."""
    try:
        return auto_cls.from_pretrained(model_id, backend="pil", **kwargs)
    except TypeError:
        return auto_cls.from_pretrained(model_id, **kwargs)
    except ValueError as exc:
        # Older ViT repos name the removed "ViTFeatureExtractor"; its settings load unchanged
        # into ViTImageProcessor (same resize, rescale and normalisation keys).
        if "Unrecognized image processor" not in str(exc):
            raise
        from transformers import AutoConfig, ViTImageProcessor
        if AutoConfig.from_pretrained(model_id, **kwargs).model_type != "vit":
            raise
        return ViTImageProcessor.from_pretrained(model_id, **kwargs)


def _label_mapping(id2label, num_labels, reviewed):
    lower = {i: lbl.lower() for i, lbl in id2label.items()}
    if num_labels == 1:
        if reviewed and reviewed.get("output") == "sigmoid_fake":
            return {"kind": "sigmoid", "ai_indices": [0],
                    "description": "Single logit; sigmoid = probability of 'fake' (per the model card). "
                                   "Real-image score = 1 - AI-generated score."}
        return {"kind": "unmapped", "description": "Single-logit model without reviewed output semantics."}
    ai = [i for i, lbl in lower.items() if lbl in AI_LABELS]
    human = [i for i, lbl in lower.items() if lbl in HUMAN_LABELS]
    if len(ai) == 1 and len(human) == 1 and len(lower) == 2:
        return {"kind": "softmax", "ai_indices": ai,
                "description": f"Softmax over two classes; AI-generated score = P('{id2label[ai[0]]}'), "
                               f"real-image score = P('{id2label[human[0]]}')."}
    if len(human) == 1 and len(lower) > 2:
        others = [i for i in lower if i != human[0]]
        return {"kind": "softmax", "ai_indices": others,
                "description": f"Softmax over {len(lower)} classes; AI-generated score = 1 - P('{id2label[human[0]]}') "
                               f"(sum of {', '.join(id2label[i] for i in others)})."}
    return {"kind": "unmapped", "description": "Labels could not be mapped to AI/real classes."}


def _input_size(processor):
    # transformers 4 uses a dict, transformers 5 a SizeDict object; read both the same way.
    def read(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    crop = getattr(processor, "crop_size", None) if getattr(processor, "do_center_crop", False) else None
    size = getattr(processor, "size", None)
    if crop is not None and read(crop, "height"):
        return f"{read(crop, 'width')}x{read(crop, 'height')} (centre crop after resizing shortest edge to {read(size, 'shortest_edge')})"
    if size is not None and read(size, "height") and read(size, "width"):
        return f"{read(size, 'width')}x{read(size, 'height')}"
    if size is not None and read(size, "shortest_edge"):
        return f"shortest edge {read(size, 'shortest_edge')}"
    return None
