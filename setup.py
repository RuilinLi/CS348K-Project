from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv_cuda',
    include_dirs=['/home/ruilin/cutlass/include'],
    ext_modules=[
        CUDAExtension('conv_cuda', [
            'conv.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })