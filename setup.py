import numpy
from setuptools import setup, Extension

extension = Extension(
    'zeichenformer._tokenizers',
    sources=[
        'src/tokenizers.c',
        'src/binary.c',
        'src/category.c',
        'src/timestamp.c'
    ],
    include_dirs=['src', numpy.get_include()],
    extra_compile_args=[
        '-O0',
        '-Wall',
        '-Werror=incompatible-pointer-types',
        '-Werror=implicit-function-declaration',
        '-fno-strict-aliasing',
        '-fPIC'  # Position Independent Code
    ],
)

setup(
    name='zeichenformer',
    version='1.0.8',
    ext_modules=[extension],
    packages=['zeichenformer'],
    package_dir={'zeichenformer': './zeichenformer'},
    install_requires=['numpy'],
)