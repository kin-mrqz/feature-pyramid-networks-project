from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    'src/crop_and_resize_extension.cpp',
    'src/crop_and_resize.c',
    'src/crop_and_resize_gpu.c',
]

cuda_sources = ['src/crop_and_resize_kernel.cu']

ext_modules = []

try:
    import torch
    if torch.cuda.is_available():
        ext_modules.append(
            CUDAExtension(
                name='roi.align._ext.crop_and_resize',
                sources=sources + cuda_sources,
            )
        )
    else:
        from torch.utils.cpp_extension import CppExtension
        ext_modules.append(
            CppExtension(
                name='roi.align._ext.crop_and_resize',
                sources=sources,
            )
        )
except Exception:
    # fallback to CPU-only if torch is missing at build-time
    from torch.utils.cpp_extension import CppExtension
    ext_modules.append(
        CppExtension(
            name='roi.align._ext.crop_and_resize',
            sources=sources,
        )
    )

setup(
    name='roi_align_crop_and_resize',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
