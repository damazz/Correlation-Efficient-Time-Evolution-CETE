import numpy as np
from pyscf import gto

from sparse_sim.fermion.hamiltonian import *


def test_hf_energy():

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )
    H2.build()

    H = init_Hamiltonian_from_pyscf(H2)

    if False:
        print("SparseSim")
        print(f"{len(H.prods)} prods:")
        for prod in H.prods:
            print(prod)

    if False:
        np.set_printoptions(linewidth=np.inf, precision=2)
        print(H.pSum.matrix.real)

    hf_sdet = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    hf_wfn = Wavefunction(4)
    hf_wfn.append_slater_determinant(hf_sdet)

    tomography = wavefunction_perform_tomography(
        hf_wfn, H.aggregate_measurements())

    print(f"H2 HF energy: {H.energy(tomography)}")

    if False:
        print("Qiskit Nature")
        driver = PySCFDriver(
            atom="H 0 0 0; H 0 0 0.7348654",
            unit=DistanceUnit.ANGSTROM,
            basis="sto3g",
        )
        problem = driver.run()
        hamiltonian = problem.hamiltonian

        print("Fermionic Hamiltonian (Qiskit Output):")

        fermionic_op = hamiltonian.second_q_op()

        if True:
            jw_mapper = JordanWignerMapper()
            qubit_op = jw_mapper.map(fermionic_op)

            matrix = SparsePauliOp(qubit_op).to_matrix()

            print("Matrix representation of the Hamiltonian:")
            print(matrix.real)

        if False:
            print(fermionic_op)


def main():
    print("Testing Hamiltonian class")
    test_hf_energy()


if __name__ == "__main__":
    main()
