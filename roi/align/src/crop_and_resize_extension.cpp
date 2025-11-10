#include <torch/extension.h>
#include <vector>
#include "crop_and_resize.h"

// C++ wrapper that converts between at::Tensor and raw pointers.

at::Tensor crop_and_resize_forward(
    const at::Tensor & image,
    const at::Tensor & boxes,
    const at::Tensor & box_index,
    const float extrapolation_value,
    const int crop_h,
    const int crop_w
){
    const auto batch_size = image.size(0);
    const auto depth = image.size(1);
    const auto image_h = image.size(2);
    const auto image_w = image.size(3);

    const auto num_boxes = boxes.size(0);

    auto crops = at::zeros({num_boxes, depth, crop_h, crop_w}, image.options());

    if (image.is_cuda()) {
        // forward to GPU implementation
        void * stream = at::cuda::getCurrentCUDAStream().stream();
        crop_and_resize_gpu_forward_raw(
            image.data_ptr<float>(),
            (int)batch_size,
            (int)depth,
            (int)image_h,
            (int)image_w,
            boxes.data_ptr<float>(),
            (int*)boxes.data_ptr<int>(),
            (int)num_boxes,
            extrapolation_value,
            crop_h,
            crop_w,
            crops.data_ptr<float>(),
            stream
        );
    } else {
        crop_and_resize_forward_raw(
            image.data_ptr<float>(),
            (int)batch_size,
            (int)depth,
            (int)image_h,
            (int)image_w,
            boxes.data_ptr<float>(),
            (int*)box_index.data_ptr<int>(),
            (int)num_boxes,
            extrapolation_value,
            crop_h,
            crop_w,
            crops.data_ptr<float>()
        );
    }

    return crops;
}

at::Tensor crop_and_resize_backward(
    const at::Tensor & grads,
    const at::Tensor & boxes,
    const at::Tensor & box_index,
    const int batch,
    const int depth,
    const int H,
    const int W,
    const int crop_h,
    const int crop_w
){
    auto grads_image = at::zeros({batch, depth, H, W}, grads.options());
    const auto num_boxes = boxes.size(0);

    if (grads.is_cuda()) {
        void * stream = at::cuda::getCurrentCUDAStream().stream();
        crop_and_resize_gpu_backward_raw(
            grads.data_ptr<float>(),
            boxes.data_ptr<float>(),
            (int*)box_index.data_ptr<int>(),
            (int)num_boxes,
            grads_image.data_ptr<float>(),
            (int)batch,
            (int)depth,
            (int)H,
            (int)W,
            crop_h,
            crop_w,
            stream
        );
    } else {
        crop_and_resize_backward_raw(
            grads.data_ptr<float>(),
            boxes.data_ptr<float>(),
            (int*)box_index.data_ptr<int>(),
            (int)num_boxes,
            grads_image.data_ptr<float>(),
            (int)batch,
            (int)depth,
            (int)H,
            (int)W,
            crop_h,
            crop_w
        );
    }

    return grads_image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &crop_and_resize_forward, "crop_and_resize forward");
    m.def("backward", &crop_and_resize_backward, "crop_and_resize backward");
}
