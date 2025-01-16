from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import pybind11

# Módulo de extensão
module = Extension(
    "BasBR",  # Nome do módulo Python
    sources=["main.cpp"],  # Arquivos de código-fonte C++
    include_dirs=[pybind11.get_include()],  # Diretório de cabeçalhos do Pybind11
    language="c++",  # Linguagem usada
)

# Configuração
setup(
    name="BasBR",  # Nome do pacote
    version="0.1",
    author="Jhonata",
    description="Brazilian programming language additions",
    ext_modules=[module],
    cmdclass={"build_ext": build_ext},
)