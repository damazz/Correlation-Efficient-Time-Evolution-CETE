from pathlib import Path
from setuptools import setup, find_packages, Extension
import os
import platform

root = Path(__file__).parent
pkg = root / "src" / "sparse_sim"
cy = pkg / "cython"
csrc = pkg / "src"
inc = pkg / "include"


def rel(p: Path) -> str:
    return os.path.relpath(p, start=root)


use_c = (cy / "core.c").exists()

sources = [
    rel(cy / ("core.c" if use_c else "core.pyx")),
    rel(csrc / "pauli.c"),
    rel(csrc / "wavefunction.c"),
]

include_dirs = [rel(inc), rel(csrc), rel(cy)]
extra_compile_args, extra_link_args = [
    "-O3", "-ffast-math", "-funroll-loops"], []

sys = platform.system()
if sys == "Linux":
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]
elif sys == "Darwin":
    # enable OpenMP only if Homebrew libomp is present
    omp = Path("/opt/homebrew/opt/libomp")
    if (omp / "include").exists() and (omp / "lib").exists():
        include_dirs.append(str(omp / "include"))
        extra_link_args += [f"-L{omp/'lib'}", "-lomp"]
        extra_compile_args += ["-Xpreprocessor", "-fopenmp"]

ext = Extension(
    name="sparse_sim.cython.core",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)

kwargs = dict(
    name="CETE",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=[
                           "CETE", "CETE.*", "sparse_sim", "sparse_sim.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    package_data={
        "sparse_sim": ["cython/*.pyi", "__init__.pyi", "py.typed"],
    },
)

if use_c:
    setup(ext_modules=[ext], **kwargs)
else:
    from Cython.Build import cythonize
    setup(ext_modules=cythonize(
        [ext], compiler_directives={"language_level": 3}), **kwargs)
