import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='nms_ext',
    ext_modules=[
        CUDAExtension(
            name='nms._ext.nms',
            sources=[
                os.path.join(this_dir, 'src/nms.c'),
            ],
            extra_objects=[
                os.path.join(this_dir, 'src/nms_cuda.o')
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
