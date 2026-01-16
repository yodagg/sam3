from typing import Any, Dict, List, Optional

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
            enable_inst_interactivity=True,
        )

        self.processor = Sam3Processor(model, device=device)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Text prompt describing what to segment"),
        points: str = Input(
            description="Point prompts as JSON string [[x,y], ...] in pixels",
            default=None,
        ),
        point_labels: str = Input(
            description="Labels for points as JSON string [1, 0, ...]: 1=foreground, 0=background",
            default=None,
        ),
        box_xyxy: List[float] = Input(
            description="Box prompt [x0,y0,x1,y1] in pixels",
            default=None,
        ),
        multimask_output: bool = Input(
            description="Return multiple masks for ambiguous prompts",
            default=True,
        ),
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
        import numpy as np
        import torch
        from PIL import Image as PILImage
        from sam3.model.box_ops import box_xyxy_to_xywh
        from torchvision.ops import masks_to_boxes
        from sam3.train.masks_ops import rle_encode
        import json

        self.processor.set_confidence_threshold(confidence_threshold)
        
        points_list = None
        if points:
            try:
                points_list = json.loads(points)
            except Exception as e:
                print(f"Error parsing points JSON: {e}")

        point_labels_list = None
        if point_labels:
            try:
                point_labels_list = json.loads(point_labels)
            except Exception as e:
                print(f"Error parsing point_labels JSON: {e}")

        if points_list is not None or box_xyxy is not None:
            img = PILImage.open(str(image))
            state = self.processor.set_image(img)

            pc = None if points_list is None else np.array(points_list, dtype=np.float32)
            pl = None if point_labels_list is None else np.array(point_labels_list, dtype=np.int64)
            bx = None if box_xyxy is None else np.array(box_xyxy, dtype=np.float32)

            if pc is not None and pl is None:
                pl = np.ones((pc.shape[0],), dtype=np.int64)

            masks_np, scores_np, _ = self.processor.model.predict_inst(
                state,
                point_coords=pc,
                point_labels=pl,
                box=bx,
                multimask_output=multimask_output,
            )

            masks_t = torch.from_numpy(masks_np)
            boxes_xyxy = masks_to_boxes(masks_t)

            orig_w, orig_h = img.size
            boxes_xyxy_norm = torch.stack(
                [
                    boxes_xyxy[:, 0] / orig_w,
                    boxes_xyxy[:, 1] / orig_h,
                    boxes_xyxy[:, 2] / orig_w,
                    boxes_xyxy[:, 3] / orig_h,
                ],
                dim=-1,
            )
            boxes_xywh_norm = box_xyxy_to_xywh(boxes_xyxy_norm).tolist()

            rles = rle_encode(masks_t)
            pred_masks = [m["counts"] for m in rles]

            scores = scores_np.tolist()
            boxes = boxes_xywh_norm
            masks = pred_masks

            result = {
                "orig_img_h": orig_h,
                "orig_img_w": orig_w,
                "pred_boxes": boxes,
                "pred_masks": masks,
                "pred_scores": scores,
            }
        else:
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
