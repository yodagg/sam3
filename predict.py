from typing import Any, Dict

import os

import sam3
from cog import BasePredictor, Input, Path
from sam3 import build_sam3_image_model
from sam3.agent.client_sam3 import sam3_inference
from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3.model.sam3_image_processor import Sam3Processor


class Predictor(BasePredictor):
    def setup(self) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3_root = os.path.dirname(sam3.__file__)
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )

        self.processor = Sam3Processor(model, device=device)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Text prompt describing what to segment"),
        confidence_threshold: float = Input(
            description="Minimum score to keep a mask",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        max_masks: int = Input(
            description="Maximum number of masks to return",
            default=20,
            ge=1,
            le=100,
        ),
    ) -> Dict[str, Any]:
        self.processor.set_confidence_threshold(confidence_threshold)

        result = sam3_inference(self.processor, str(image), prompt)
        result = remove_overlapping_masks(result)

        scores = result.get("pred_scores") or []
        boxes = result.get("pred_boxes") or []
        masks = result.get("pred_masks") or []

        if scores:
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            indices = indices[:max_masks]
            scores = [scores[i] for i in indices]
            boxes = [boxes[i] for i in indices]
            masks = [masks[i] for i in indices]

        filtered_scores = []
        filtered_boxes = []
        filtered_masks = []
        for i, rle in enumerate(masks):
            if len(rle) > 4:
                filtered_masks.append(rle)
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[i])

        result["pred_scores"] = filtered_scores
        result["pred_boxes"] = filtered_boxes
        result["pred_masks"] = filtered_masks

        return result

