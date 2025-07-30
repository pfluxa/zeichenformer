from setuptools import setup, Extension
import numpy

# Define the C extension module
# This is the complex build logic that belongs in a .py file.
extension = Extension(
    # The full name of the module as it will be imported in Python
    'zeichenformer._tokenizers',
    sources=[
        'src/tokenizers.c',
        'src/numerical.c',
        'src/category.c',
        'src/timestamp.c'
    ],
    # Directories to search for header files (.h)
    include_dirs=['src', numpy.get_include()],
    # Extra flags for the C compiler
    extra_compile_args=[
        '-O2',
        '-Wall',
        '-Werror=incompatible-pointer-types',
        '-Werror=implicit-function-declaration',
        '-fno-strict-aliasing',
        '-fPIC'
    ],
)

# The setup() call now only needs to know about the extension modules.
# All other metadata (name, version, dependencies) is in pyproject.toml.
setup(
    ext_modules=[extension],
)
