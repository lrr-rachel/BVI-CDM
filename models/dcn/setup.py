from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deform_conv_ext',
    ext_modules=[
        CUDAExtension(
            name='deform_conv_ext',
            sources=[
                'models/dcn/src/deform_conv_ext.cpp',
                'models/dcn/src/deform_conv_cuda.cpp',
                'models/dcn/src/deform_conv_cuda_kernel.cu',
            ],
            include_dirs=[],  # Add necessary include directories here
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# run with: python setup.py build_ext --inplace