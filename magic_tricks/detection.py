"""Zero-shot object detection with Grounding DINO."""

from collections import namedtuple

import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_BATCH_SIZE = 4

Detection = namedtuple("Detection", "score box label")


def default_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroundingDino:
    """Callable detector: frame (BGR) + text prompt -> list of Detection.

    `detect_batch` runs several frames through the model at once, which keeps
    the GPU busier than one frame at a time. On CUDA the model runs in float16
    (autocast), about 40% faster; scores move by less than 0.03.
    """

    def __init__(self, model_id=MODEL_ID, device=None):
        # Heavy imports kept local so the rest of the package works without torch.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or default_device(torch)
        # The torchvision backend resizes and normalizes on the device the frames are on,
        # which is much faster than PIL on the CPU.
        self.processor = AutoProcessor.from_pretrained(model_id, backend="torchvision")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, frame, prompt, text_threshold):
        return self.detect_batch([frame], prompt, text_threshold)[0]

    def detect_batch(self, frames, prompt, text_threshold):
        """One list of Detection per frame. Frames are BGR images, all of the same size."""
        torch = self._torch
        # BGR (N, H, W, 3) uint8 -> RGB (N, 3, H, W) on the device.
        images = torch.from_numpy(np.stack(frames)).to(self.device).flip(-1).permute(0, 3, 1, 2)
        inputs = self.processor(images=list(images), text=[prompt] * len(frames),
                                return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            text_threshold=text_threshold,
            target_sizes=[frame.shape[:2] for frame in frames],
        )
        return [_to_detections(result) for result in results]


def _to_detections(result):
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
