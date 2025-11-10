#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// declare C functions implemented in C sources
extern "C" {
    void crop_and_resize_forward_raw(
        const float * image_data,
        const int batch_size,
        const int depth,
        const int image_height,
        const int image_width,

        const float * boxes_data,
        const int * box_index_data,
        const int num_boxes,

        const float extrapolation_value,
        const int crop_height,
        const int crop_width,

        float * crops_data
    );

    void crop_and_resize_backward_raw(
        const float * grads_data,
        const float * boxes_data,
        const int * box_index_data,
        const int num_boxes,
        float * grads_image_data,
        const int batch_size,
        const int depth,
        const int image_height,
        const int image_width,
        const int crop_height,
        const int crop_width
    );

    void crop_and_resize_gpu_forward_raw(
        const float * image_ptr,
        const float * boxes_ptr,
        const int * box_ind_ptr,
        int num_boxes,
        int batch,
        int image_height,
        int image_width,
        int crop_height,
        int crop_width,
        int depth,
        float extrapolation_value,
        float * crops_ptr,
        cudaStream_t stream
    );

    void crop_and_resize_gpu_backward_raw(
        const float * grads_ptr,
        const float * boxes_ptr,
        const int * box_ind_ptr,
        int num_boxes,
        int batch,
        int image_height,
        int image_width,
        int crop_height,
        int crop_width,
        int depth,
        float * grads_image_ptr,
        cudaStream_t stream
    );
}


at::Tensor crop_and_resize_forward(
    const at::Tensor & image,
    const at::Tensor & boxes,
    const at::Tensor & box_index,
    const float extrapolation_value,
    const int crop_height,
    const int crop_width
){
    TORCH_CHECK(image.dim() == 4, "image must be 4D (N,C,H,W)");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be Nx4");

    const int batch_size = image.size(0);
    const int depth = image.size(1);
    const int image_height = image.size(2);
    const int image_width = image.size(3);
    const int num_boxes = boxes.size(0);

    auto options = image.options();
    at::Tensor crops = at::zeros({num_boxes, depth, crop_height, crop_width}, options);

    if (image.is_cuda()) {
        // GPU path
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        crop_and_resize_gpu_forward_raw(
            image.data_ptr<float>(),
            boxes.data_ptr<float>(),
            box_index.data_ptr<int>(),
            num_boxes, batch_size, image_height, image_width,
            crop_height, crop_width, depth, extrapolation_value,
            crops.data_ptr<float>(), stream
        );
    } else {
        // CPU path
        crop_and_resize_forward_raw(
            image.data_ptr<float>(),
            batch_size, depth, image_height, image_width,
            boxes.data_ptr<float>(), box_index.data_ptr<int>(), num_boxes,
            extrapolation_value, crop_height, crop_width,
            crops.data_ptr<float>()
        );
    }

    return crops;
}

at::Tensor crop_and_resize_backward(
    const at::Tensor & grads,
    const at::Tensor & boxes,
    const at::Tensor & box_index,
    const int batch_size,
    const int depth,
    const int image_height,
    const int image_width
){
    const int num_boxes = grads.size(0);
    const int crop_height = grads.size(2);
    const int crop_width = grads.size(3);

    auto options = grads.options();
    at::Tensor grads_image = at::zeros({batch_size, depth, image_height, image_width}, options);

    if (grads.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        crop_and_resize_gpu_backward_raw(
            grads.data_ptr<float>(), boxes.data_ptr<float>(), box_index.data_ptr<int>(),
            num_boxes, batch_size, image_height, image_width,
            crop_height, crop_width, depth, grads_image.data_ptr<float>(), stream
        );
    } else {
        crop_and_resize_backward_raw(
            grads.data_ptr<float>(), boxes.data_ptr<float>(), box_index.data_ptr<int>(),
            num_boxes, grads_image.data_ptr<float>(), batch_size, depth, image_height, image_width,
            crop_height, crop_width
        );
    }

    return grads_image;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &crop_and_resize_forward, "CropAndResize forward");
    m.def("backward", &crop_and_resize_backward, "CropAndResize backward");
}
