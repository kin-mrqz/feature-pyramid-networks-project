import math
from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import roi_align as torchvision_roi_align


class Wrapper(object):

    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, mode: Mode, image_width: int, image_height: int) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        proposal_bboxes = proposal_bboxes.detach()

        scale_x = image_width / feature_map_width
        scale_y = image_height / feature_map_height

        if mode == Wrapper.Mode.POOLING:
            pool = []
            for proposal_bbox in proposal_bboxes:
                start_x = max(min(round(proposal_bbox[0].item() / scale_x), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() / scale_y), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() / scale_x) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() / scale_y) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[..., start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=7))
            pool = torch.cat(pool, dim=0)
        elif mode == Wrapper.Mode.ALIGN:
            # Use PyTorch's built-in RoI Align from torchvision
            # torchvision expects boxes in (x1, y1, x2, y2) format (already the case)
            # and spatial_scale = 1 / stride
            spatial_scale = 1.0 / ((scale_x + scale_y) / 2.0)
            pool = torchvision_roi_align(
                features,
                [proposal_bboxes],  # List of boxes per image (we have batch_size=1)
                output_size=(7, 7),
                spatial_scale=spatial_scale,
                sampling_ratio=2
            )
        else:
            raise ValueError

        return pool

