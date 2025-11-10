import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

this_file = os.path.dirname(os.path.realpath(__file__))

sources = ['src/crop_and_resize.cpp']  # Changed from .c to .cpp
headers = ['src/crop_and_resize.h']
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/crop_and_resize_gpu.cpp']  # Changed from .c to .cpp
    headers += ['src/crop_and_resize_gpu.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/crop_and_resize_kernel.cu']  # Remove .o extension
    with_cuda = True

sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

setup(
    name='crop_and_resize',
    ext_modules=[
        CUDAExtension(
            name='roi.align._ext.crop_and_resize',
            sources=sources + extra_objects,
            define_macros=defines,
            extra_compile_args={'cxx': ['-std=c++14'],
                              'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })