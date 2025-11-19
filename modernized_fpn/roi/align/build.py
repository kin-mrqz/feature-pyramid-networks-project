import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.realpath(__file__))

sources = [os.path.join(this_dir, 'src/crop_and_resize.c')]
extra_objects = []

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources.append(os.path.join(this_dir, 'src/crop_and_resize_gpu.c'))
    extra_objects.append(os.path.join(this_dir, 'src/cuda/crop_and_resize_kernel.cu.o'))

print(this_dir)

setup(
    name='crop_and_resize_ext',
    ext_modules=[
        CUDAExtension(
            name='roi.align._ext.crop_and_resize',
            sources=sources,
            extra_objects=extra_objects,
            extra_compile_args={'cxx': ['-std=c99']},
            define_macros=[('WITH_CUDA', None)] if torch.cuda.is_available() else [],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
