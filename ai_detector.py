"""Pretrained AI-generated-image classifier (Hugging Face transformers).

The model is loaded once per process and reused. If the libraries, weights or
network are unavailable, every call returns an explicit "unavailable" result;
no prediction is ever fabricated or replaced by a heuristic.
"""
import logging
import threading
import time

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# Label names (lower-cased) we know how to interpret. Anything else is reported
# verbatim without being mapped to "AI" or "human".
AI_LABELS = {"ai", "artificial", "fake", "generated", "ai-generated", "ai_generated", "synthetic"}
HUMAN_LABELS = {"hum", "human", "real", "authentic", "natural", "photo"}

RETRY_AFTER_SECONDS = 60
REVIEWED_MODEL_ID = "Ateeqq/ai-vs-human-image-detector"

SCORE_NOTE = ("The score is the softmax output of the classifier for the 'AI-generated' class. "
              "It is not a calibrated probability: it reflects how strongly this model's learned "
              "features match its training data, and it can be confidently wrong on images unlike "
              "that data (new generators, heavy compression, screenshots, resized or edited photos).")
DECISION_RULE = ("The prediction is the class with the highest softmax score (argmax). With two classes "
                 "this equals 'AI-generated if the AI-generated score is above 50%'.")
CONFIDENCE_NOTE = ("Not independently calibrated. The two class scores sum to 100% and are the only "
                   "output of the model; the model card reports no calibration study, so no separate "
                   "confidence value is shown.")
# Limitations from the model card and from how the model is used here.
MODEL_LIMITATIONS = [
    "Fine-tuned on about 60,000 AI-generated and 60,000 human images (per the model card); the "
    "exact generators and photo sources are not fully documented.",
    "The model card notes that some users reported overfitting; reported test accuracy (99.2%) is "
    "on the author's own held-out split and will be lower on real-world images.",
    "Input is resized to 224x224, so small pixel-level artefacts are not visible to the model.",
    "Images from generators released after training, heavy compression, screenshots and edited "
    "photos can be misclassified with high scores.",
]


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
    def __init__(self, model_id, revision=None, enabled=True, local_files_only=False):
        self.model_id = model_id
        self.revision = revision
        self.enabled = enabled
        self.local_files_only = local_files_only
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
        if self.model_id == REVIEWED_MODEL_ID:
            return MODEL_LIMITATIONS
        return ["This model was configured by the user and has not been reviewed for this "
                "application; consult its model card for training data and limitations."]

    def status(self):
        return {
            "enabled": self.enabled,
            "state": self.state,
            "model_id": self.model_id,
            "revision": self.revision,
            "error": self.error,
            "known_limitations": self.known_limitations(),
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
                processor = AutoImageProcessor.from_pretrained(self.model_id, **kwargs)
                model = AutoModelForImageClassification.from_pretrained(self.model_id, **kwargs)
                model.eval()
            except OSError as exc:
                return self._fail("Model weights could not be loaded. If this is the first run, the "
                                  "computer needs internet access to download them (or run "
                                  f"download_model.py). Details: {str(exc)[:300]}")
            except Exception as exc:
                return self._fail(f"Model failed to load: {type(exc).__name__}: {str(exc)[:300]}")

            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            ai_idx = [i for i, lbl in id2label.items() if lbl.lower() in AI_LABELS]
            human_idx = [i for i, lbl in id2label.items() if lbl.lower() in HUMAN_LABELS]

            self._torch = torch
            self._processor = processor
            self._model = model
            self.info = {
                "architecture": (model.config.architectures or [type(model).__name__])[0],
                "model_type": model.config.model_type,
                "commit_hash": getattr(model.config, "_commit_hash", None),
                "labels": id2label,
                "ai_label_index": ai_idx[0] if len(ai_idx) == 1 else None,
                "human_label_index": human_idx[0] if len(human_idx) == 1 else None,
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
                    logits = self._model(**inputs).logits[0]
                probs = self._torch.softmax(logits, dim=-1)
                elapsed_ms = round((time.time() - started) * 1000, 1)
        except Exception as exc:
            log.exception("AI inference failed")
            return {**base, "status": "error",
                    "message": f"Inference failed: {type(exc).__name__}: {str(exc)[:300]}"}

        labels = self.info["labels"]
        scores = [{"index": i, "label": labels.get(i, str(i)),
                   "logit": round(float(logits[i]), 6), "softmax": round(float(probs[i]), 6)}
                  for i in range(len(probs))]
        top = max(scores, key=lambda s: s["softmax"])
        ai_idx = self.info["ai_label_index"]
        human_idx = self.info["human_label_index"]
        result = {
            **base,
            "status": "ok",
            "commit_hash": self.info["commit_hash"],
            "architecture": self.info["architecture"],
            "scores": scores,
            "predicted_label": top["label"],
            "predicted_index": top["index"],
            "top_score": top["softmax"],
            "inference_ms": elapsed_ms,
            "model_input_size": _input_size(self._processor),
            "score_note": SCORE_NOTE,
            "decision_rule": DECISION_RULE,
            "known_limitations": self.known_limitations(),
            # The model emits only class scores; no separate confidence measure exists.
            "confidence": None,
            "confidence_note": CONFIDENCE_NOTE,
        }
        if ai_idx is not None:
            result["ai_score"] = scores[ai_idx]["softmax"]
            result["real_score"] = scores[human_idx]["softmax"] if human_idx is not None else None
            is_ai = top["index"] == ai_idx
            result["predicted_class"] = "ai_generated" if is_ai else "not_ai_generated"
            result["prediction_label"] = "Likely AI-generated" if is_ai else "Likely real"
        else:
            result["ai_score"] = None
            result["real_score"] = None
            result["predicted_class"] = "unmapped"
            result["prediction_label"] = f"Raw label: {top['label']}"
            result["message"] = ("The model's labels could not be mapped to AI/human classes; raw "
                                 "outputs are shown without interpretation.")
        return result


def _input_size(processor):
    # transformers 4 uses a dict, transformers 5 a SizeDict object; read both the same way.
    size = getattr(processor, "size", None)
    get = size.get if isinstance(size, dict) else (lambda k: getattr(size, k, None))
    if get("height") and get("width"):
        return f"{get('width')}x{get('height')}"
    if get("shortest_edge"):
        return f"shortest edge {get('shortest_edge')}"
    return None
