#include "crop_and_resize.h"
#include <stdio.h>

// Forward declare the CUDA kernel launchers implemented in the .cu file
void CropAndResizeForwardLauncher(
    const float* image_data,
    const int batch_size,
    const int depth,
    const int image_height,
    const int image_width,
    const float* boxes_data,
    const int* box_index_data,
    const int num_boxes,
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    float* crops_data,
    void* stream
);

void CropAndResizeBackwardLauncher(
    const float* grads_data,
    const float* boxes_data,
    const int* box_index_data,
    const int num_boxes,
    float* grads_image_data,
    const int batch_size,
    const int depth,
    const int image_height,
    const int image_width,
    const int crop_height,
    const int crop_width,
    void* stream
);


void crop_and_resize_gpu_forward_raw(
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

    float * crops_data,
    void * stream
){
    // call the CUDA launcher; stream is forwarded through as void*
    CropAndResizeForwardLauncher(
        image_data,
        batch_size,
        depth,
        image_height,
        image_width,
        boxes_data,
        box_index_data,
        num_boxes,
        extrapolation_value,
        crop_height,
        crop_width,
        crops_data,
        stream
    );
}

void crop_and_resize_gpu_backward_raw(
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
    const int crop_width,
    void * stream
){
    CropAndResizeBackwardLauncher(
        grads_data,
        boxes_data,
        box_index_data,
        num_boxes,
        grads_image_data,
        batch_size,
        depth,
        image_height,
        image_width,
        crop_height,
        crop_width,
        stream
    );
}
