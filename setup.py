from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv_cuda',
    include_dirs=['/home/ruilin/cutlass/include', "/home/ruilin/cutlass/examples/13_two_tensor_op_fusion"],
    ext_modules=[
        CUDAExtension('conv_cuda', [
            'conv.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

