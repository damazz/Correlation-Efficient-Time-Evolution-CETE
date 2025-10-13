from setuptools import setup, find_packages

setup(
    name="sparse_sim",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "qiskit<2.0",
        "qiskit_aer",
        "qiskit_ibm_runtime",
        "mthree",
        "pyscf"
    ]
)
