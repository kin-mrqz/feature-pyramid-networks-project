import torch
from torch import Tensor
from torchvision.ops import nms as torchvision_nms


class NMS(object):

    @staticmethod
    def suppress(sorted_bboxes: Tensor, threshold: float) -> Tensor:
        # Use PyTorch's built-in NMS from torchvision
        # sorted_bboxes shape: [N, 4] in (x1, y1, x2, y2) format
        # We need scores for torchvision's nms, but since boxes are already sorted,
        # we can use descending indices as scores
        scores = torch.arange(len(sorted_bboxes), 0, -1, dtype=torch.float32, device=sorted_bboxes.device)
        kept_indices = torchvision_nms(sorted_bboxes, scores, threshold)
        return kept_indices
