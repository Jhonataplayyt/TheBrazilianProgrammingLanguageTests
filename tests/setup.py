from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('./tests/__init__.py')
)