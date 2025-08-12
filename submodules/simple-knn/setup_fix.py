from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Убедимся, что директория существует
os.makedirs('simple_knn', exist_ok=True)

setup(
    name='simple_knn',
    ext_modules=[
        CUDAExtension(
            name='simple_knn._C',
            sources=['spatial.cu', 'simple_knn.cu', 'ext.cpp'],
            extra_compile_args={'nvcc': ['-gencode', 'arch=compute_86,code=sm_86']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
