"""Zero-shot object detection with Grounding DINO."""

from collections import namedtuple

import cv2

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

Detection = namedtuple("Detection", "score box label")


def default_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroundingDino:
    """Callable detector: frame (BGR) + text prompt -> list of Detection."""

    def __init__(self, model_id=MODEL_ID, device=None):
        # Heavy imports kept local so the rest of the package works without torch.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or default_device(torch)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, frame, prompt, text_threshold):
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        # Recent transformers versions return label ids in "labels" and strings in "text_labels".
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        return [Detection(float(s), b, str(l)) for b, s, l in zip(boxes, scores, labels)]


def best_per_label(detections, threshold):
    """Highest-scoring detection of each label above the threshold, sorted by decreasing score."""
    best = {}
    for det in detections:
        if det.score >= threshold and (det.label not in best or det.score > best[det.label].score):
            best[det.label] = det
    return sorted(best.values(), key=lambda d: d.score, reverse=True)
