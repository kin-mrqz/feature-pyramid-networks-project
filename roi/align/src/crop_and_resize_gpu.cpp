#include "cuda/crop_and_resize_kernel.h"

// Raw GPU-facing API: accepts device pointers and a cudaStream_t.
// This avoids depending on the legacy THC API.

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
){
    CropAndResizeLaucher(
        image_ptr, boxes_ptr, box_ind_ptr,
        num_boxes, batch, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        crops_ptr, stream
    );
}

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
){
    CropAndResizeBackpropImageLaucher(
        grads_ptr, boxes_ptr, box_ind_ptr,
        num_boxes, batch, image_height, image_width,
        crop_height, crop_width, depth,
        grads_image_ptr, stream
    );
}