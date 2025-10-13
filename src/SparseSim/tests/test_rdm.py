from pyscf import gto

from sparse_sim.fermion.rdm import *
from sparse_sim.fermion.hamiltonian import *


def test_rdm_init_and_freeing():

    N = 2
    p = 2

    trdm = RDM(p, N)
    for prod in trdm.prods:
        print(prod)
    print(f"{len(trdm.prods)} elements in {trdm}")


def test_rdm_measurement():

    ordm = RDM(1, 2)
    for prod in ordm.prods:
        print(prod)
    print(f"{len(ordm.prods)} elements in {ordm}")

    orbitals0 = [1, 0]
    orbitals1 = [0, 1]
    sdet0 = SlaterDeterminant(2, 1 + 0j, orbitals0)
    sdet1 = SlaterDeterminant(2, 0 + 1j, orbitals1)

    wfn = Wavefunction(2)

    wfn.append_slater_determinant(sdet0)
    wfn.append_slater_determinant(sdet1)

    measurements = ordm.aggregate_measurements_recursive()
    print(f"Wavefunction: {wfn}")
    print(f"Measurements: {measurements}")

    tomography = wavefunction_perform_tomography(wfn, measurements)

    print(f"Tomography data:\n {tomography}")
    print(
        f"Expectation value: {ordm.evaluate_expectation(tomography)} = {ordm.prods[0].evaluate_expectation(tomography)} + {ordm.prods[1].evaluate_expectation(tomography)} + {ordm.prods[2].evaluate_expectation(tomography)} + {ordm.prods[3].evaluate_expectation(tomography)}")


def evaluate_rdm(psi: Wavefunction, rdm: RDM):

    new_rdm_prods = []
    bra = psi.adjoint()
    for prod in rdm.prods:
        new_coef = bra * (prod.pSum * psi)
        if prod.coef == 0:
            new_prod = 0 * prod
        else:
            new_prod = (new_coef / prod.coef) * prod
        new_rdm_prods.append(new_prod)

    return RDM(rdm.p, rdm.N,  new_rdm_prods)


def test_rdm_saving_and_loading():
    ordm = RDM(1, 2)

    val = 1
    for prod in ordm.prods:
        prod.coef = val
        val += 1

    for prod in ordm.prods:
        print(prod)

    print("Saving RDM")
    saved_data = ordm.save()
    print(f"Saved data: {saved_data}")

    print("Loading RDM")
    ordm_recovered = load_rdm(saved_data)

    for prod in ordm_recovered.prods:
        print(prod)


def test_rdm_trace():

    H4 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654; H 0.7348654 0 0; H 0.7348654 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )
    eta = 4

    mf = scf.RHF(H4)
    mf.kernel()

    H = init_Hamiltonian_from_pyscf(H4, mf)

    sdet0 = SlaterDeterminant(8, 1, [1, 1, 0, 0, 1, 1, 0, 0])
    wfn0 = Wavefunction(sdet0.N) + sdet0

    wfn = 1 * wfn0
    for t in range(100):
        wfn = wavefunction_pauli_sum_evolution(H.pSum, wfn, -0.1)
        wfn = (1 / wfn.norm()) * wfn

    ordm = RDM(1, sdet0.N)
    trdm = RDM(2, sdet0.N)
    # thrdm = RDM(3, sdet0.N)

    ordm_evaluated = evaluate_rdm(wfn, ordm)
    trdm_evaluated = evaluate_rdm(wfn, trdm)
    # thrdm_evaluated = evaluate_rdm(wfn, thrdm)
    print(f"Trace of the evaluated 1RDM: {ordm_evaluated.trace()} = {eta}")
    print(
        f"Trace of the evaluated 2RDM: {trdm_evaluated.trace()} = {eta * (eta - 1)}")
    # print(f"Trace of the evaluated 3RDM: {thrdm_evaluated.trace()}")


def test_ordered_rdm():
    H4 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654; H 0.7348654 0 0; H 0.7348654 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )
    eta = 4

    mf = scf.RHF(H4)
    mf.kernel()

    H = init_Hamiltonian_from_pyscf(H4, mf)

    sdet0 = SlaterDeterminant(8, 1, [1, 1, 0, 0, 1, 1, 0, 0])
    wfn0 = Wavefunction(sdet0.N) + sdet0

    wfn = 1 * wfn0
    for t in range(100):
        wfn = wavefunction_pauli_sum_evolution(H.pSum, wfn, -0.1)
        wfn = (1 / wfn.norm()) * wfn

    ordm = RDM(1, sdet0.N, ordered=True)
    trdm = RDM(2, sdet0.N, ordered=True)
    # thrdm = RDM(3, sdet0.N)

    ordm_evaluated = evaluate_rdm(wfn, ordm)
    trdm_evaluated = evaluate_rdm(wfn, trdm)
    # thrdm_evaluated = evaluate_rdm(wfn, thrdm)
    print(f"Trace of the evaluated 1RDM: {ordm_evaluated.trace()} = {eta}")
    print(
        f"Trace of the evaluated 2RDM: {trdm_evaluated.trace()} = {eta * (eta - 1) / 2}")


def main():
    print("Testing rdm init and freeing")
    test_rdm_init_and_freeing()
    print("\nTesting RDM measurement")
    test_rdm_measurement()
    print("\nTesting RDM saving and loading")
    test_rdm_saving_and_loading()
    print("\nTesting RDM trace")
    test_rdm_trace()
    print("\nTesting ordered RDM")
    test_ordered_rdm()


if __name__ == "__main__":
    main()
