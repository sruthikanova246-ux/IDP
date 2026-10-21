"""Candidate AI-image detectors evaluated for this project, pinned to exact revisions.

Each entry records what was read on the model card before evaluation. Face-only deepfake
detectors were excluded up front (intended use is face swaps, not general images).
"""
CANDIDATES = [
    {"id": "Ateeqq/ai-vs-human-image-detector", "revision": "60e82406916921b823616bee33397baab38af3f0",
     "license": "Apache-2.0", "labels": "ai / hum (softmax)",
     "card": "SigLIP fine-tuned on 60k AI + 60k human images; card notes users reported overfitting."},
    {"id": "buildborderless/CommunityForensics-DeepfakeDet-ViT", "revision": "ac6ee457bea904a373065754107451793b56db00",
     "license": "MIT", "labels": "single logit, sigmoid = P(fake)",
     "card": "ViT-S from the Community Forensics paper (Park & Owens, CVPR 2025): 2.7M images, 4,803 generators. "
             "Third-party upload; config/weights corrected July 2026."},
    {"id": "haywoodsloan/ai-image-detector-deploy", "revision": "4ae1822603cb1ec980c217cfde1ef1b8e0477a16",
     "license": "Apache-2.0", "labels": "artificial / real (softmax)",
     "card": "SwinV2 trained with AutoTrain; card lists only validation metrics, training data undocumented."},
    {"id": "umm-maybe/AI-image-detector", "revision": "c7e223baf11bc40528af364ba7bdea030ef42f9e",
     "license": "CC-BY-4.0 (card text says CC BY-ND 4.0)", "labels": "artificial / human (softmax)",
     "card": "Oct 2022 proof of concept for AI *art*; card says it is not a photo detector."},
    {"id": "Organika/sdxl-detector", "revision": "b37fede8562cb72b89ec201c0987f96ba21b518a",
     "license": "CC-BY-NC-3.0 (non-commercial)", "labels": "artificial / human (softmax)",
     "card": "Fine-tune of umm-maybe on SDXL-era images."},
    {"id": "Smogy/SMOGY-Ai-images-detector", "revision": "fb9aadb85642d707f05b40191689d60f75b9a3e6",
     "license": "CC-BY-NC-4.0 (non-commercial)", "labels": "artificial / human (softmax)",
     "card": "Fine-tune of Organika/sdxl-detector on 50k reddit/kaggle images; card reports out-of-domain accuracy 76-91%."},
    {"id": "jacoballessio/ai-image-detect-distilled", "revision": "4098d40c873a2e38404ea378a8284f33125bd90c",
     "license": "MIT", "labels": "fake / real (softmax)",
     "card": "Distilled 11.8M ViT; card reports 74% validation accuracy."},
    {"id": "NYUAD-ComNets/NYUAD_AI-generated_images_detector", "revision": "b0a2e665f6e40164436cae6c0c95e7e4386f7684",
     "license": "Apache-2.0", "labels": "dalle / real / sd (softmax; AI score = 1 - P(real))",
     "card": "ViT; auto-generated card, training data not described beyond DALL-E and SD classes."},
]
