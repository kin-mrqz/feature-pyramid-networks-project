import torch
from torch.autograd import Function

try:
    from ._ext import crop_and_resize as _backend
except Exception:
    _backend = None


class CropAndResizeFunction(Function):
    @staticmethod
    def forward(ctx, image, boxes, box_index, crop_height, crop_width, extrapolation_value=0.0):
        if _backend is None:
            raise RuntimeError('roi.align._ext.crop_and_resize not built')

        crops = _backend.forward(image, boxes, box_index, float(extrapolation_value), int(crop_height), int(crop_width))
        ctx.save_for_backward(boxes, box_index, image)
        ctx.crop_size = (int(crop_height), int(crop_width))
        return crops

    @staticmethod
    def backward(ctx, grad_output):
        boxes, box_index, image = ctx.saved_tensors
        crop_h, crop_w = ctx.crop_size

        batch = int(image.size(0))
        depth = int(image.size(1))
        H = int(image.size(2))
        W = int(image.size(3))

        grads_image = _backend.backward(grad_output.contiguous(), boxes, box_index, batch, depth, H, W, crop_h, crop_w)
        return grads_image, None, None, None, None, None


def crop_and_resize(image, boxes, box_index, crop_height, crop_width, extrapolation_value=0.0):
    return CropAndResizeFunction.apply(image, boxes, box_index, crop_height, crop_width, extrapolation_value)
