#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// CPU raw-functions
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

// GPU forward/backward: stream is passed as void* to avoid CUDA headers here
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
);

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
);

#ifdef __cplusplus
}
#endif
