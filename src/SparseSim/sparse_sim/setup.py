from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import platform

system = platform.system()

sources = [
    "sparse_sim/cython/sparse_sim.pyx",
    "sparse_sim/src/wavefunction.c",
    "sparse_sim/src/pauli.c",
]

include_dirs = ["sparse_sim/include", "sparse_sim/src", "sparse_sim/cython"]
extra_compile_args = ["-O3", "-ffast-math", "-funroll-loops", "-flto", ]
extra_link_args = []

if system == "Darwin":
    extra_compile_args += ["-mcpu=apple-m1",
                           "-fvectorize", "-Xpreprocessor", "-fopenmp"]
    omp_root = "/opt/homebrew/opt/libomp"
    include_dirs.append(os.path.join(omp_root, "include"))
    # extra_link_args += [f"-L{os.path.join(omp_root, 'lib')}", "-lomp"]

elif system == "Linux":
    extra_compile_args += ["-march=native", "-fopenmp"]
    extra_link_args += ["-fopenmp"]

# OpenMP paths
OMP_INCLUDE = "/opt/homebrew/opt/libomp/include"
OMP_LIB = "/opt/homebrew/opt/libomp/lib"

extensions = [
    Extension(
        name="sparse_sim.sparse_sim",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3")
)
