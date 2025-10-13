from sparse_sim.fermion.projector import *
from sparse_sim.fermion.hamiltonian import *
from sparse_sim.fermion.rdm import *

import numpy as np
from pyscf import gto


def test_projector_H2():
    np.set_printoptions(linewidth=np.inf, precision=4,
                        formatter={'float_kind': '{:e}'.format})

    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    for pair in four_to_one.mapping:
        (original_sdet_right, target_sdet_right) = pair
        print(f"{original_sdet_right} -> {target_sdet_right}")

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    H = init_Hamiltonian_from_pyscf(H2)

    print("Performing mapping")

    for prod in H.prods:
        newline = f"{prod} -> "
        new_prods = prod.map(four_to_one)
        if len(new_prods) > 0:
            for new_prod in new_prods:
                newline += f"{new_prod}"
            print(newline)

    H_1q, inverse_mapping = H.map(four_to_one)
    print("H_1q:")
    for prod in H_1q.prods:
        print(prod)

    print("Inverse mapping:")
    for new_prod_ops, prod_mapping in inverse_mapping.items():
        print(f"{new_prod_ops}:")
        for term in prod_mapping:
            new_prod, prod, weight = term
            print(f"{new_prod} -> {prod} with weight {weight}")

    if False:
        print("As matrices")
        print(H.pSum.matrix.real)
        print(H_1q.pSum.matrix.real)

    if False:
        print("As Pauli sums")
        print(H.pSum)
        print(H_1q.pSum)


def test_projector_H2_time_evolution():
    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    H = init_Hamiltonian_from_pyscf(H2)

    H_1q, inverse_mapping = H.map(four_to_one)

    # Time evolution
    time_step = 0.0001 * 1j
    num_steps = 10000

    # Initial states
    psi_4q = Wavefunction(4) + sdet1010
    psi_1q = Wavefunction(1) + sdet0

    # Time evolution loop
    for step in range(num_steps):
        # Evolve 4-qubit state
        if False:
            for prod in H.prods:
                psi_4q = wavefunction_pauli_sum_evolution(
                    prod.pSum, psi_4q, time_step)
                print(f"{prod}: {psi_4q}")

            for prod in H_1q.prods:
                psi_1q = wavefunction_pauli_sum_evolution(
                    prod.pSum, psi_1q, time_step)
                print(f"{prod}: {psi_1q}")

        if True:
            # Project to 1-qubit space
            psi_4q = wavefunction_pauli_sum_evolution(
                H.pSum, psi_4q, time_step)
            psi_1q = wavefunction_pauli_sum_evolution(
                H_1q.pSum, psi_1q, time_step)
            if step % 1000 == 0:
                print(
                    f"{psi_4q.remove_global_phase()} ~ {psi_1q.remove_global_phase()}")

    print(f"{psi_4q.norm()} ~ {psi_1q.norm()}")


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


def test_projector_expectation_H2():
    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    H2 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    H = init_Hamiltonian_from_pyscf(H2)

    H_1q, inverse_mapping = H.map(four_to_one)

    # Initial states
    psi_4q = Wavefunction(4) + (0.9983) * sdet1010 + (-0.1115) * sdet0101
    psi_1q = Wavefunction(1) + (0.9983) * sdet0 + (-0.1115) * sdet1
    psi_4q = (1 / psi_4q.norm()) * psi_4q
    psi_1q = (1 / psi_1q.norm()) * psi_1q
    print(f"{psi_4q.remove_global_phase()} ~ {psi_1q.remove_global_phase()}")

    tomography_4q = wavefunction_perform_tomography(
        psi_4q, H.aggregate_measurements())
    tomography_1q = wavefunction_perform_tomography(
        psi_1q, H_1q.aggregate_measurements())

    exp_4q = H.energy(tomography_4q)
    exp_1q = H_1q.energy(tomography_1q)

    print(f"Expectation value for the original Hamiltonian: {exp_4q}")
    print(f"Expectation value for the projected Hamiltonian: {exp_1q}")

    ordm_4q = RDM(1, sdet1010.N)
    ordm_1q, inverse_mapping = ordm_4q.map(
        four_to_one, ignore_duplicates=False)

    print("1RDM Inverse Mapping:")
    for prod_ops, prod_mapping in inverse_mapping.items():
        print(f"\t{prod_ops}:")
        for term in prod_mapping:
            new_prod, prod, weight = term
            print(f"\t\t{new_prod} -> {prod} with weight {weight}")

    evaluated_ordm_4q = evaluate_rdm(psi_4q, ordm_4q)
    evaluated_ordm_1q = evaluate_rdm(psi_1q, ordm_1q)
    evaluated_ordm_4q_unmapped = evaluated_ordm_1q.unmap(inverse_mapping)

    ordm_dict = {}

    print("Evaluated 4-qubit 1RDM:")
    for i in range(len(evaluated_ordm_4q.prods)):
        ordm_element_4q = evaluated_ordm_4q.prods[i]
        ordm_dict[ordm_element_4q.ops_to_string()] = ordm_element_4q.coef
        if False:
            if ordm_element_4q.coef != 0:
                print(f"\t{ordm_element_4q}")
    print(f"\tTrace: {evaluated_ordm_4q.trace()}")

    if False:
        print("Evaluated 1-qubit 1RDM:")
        for i in range(len(evaluated_ordm_1q.prods)):
            ordm_element_1q = evaluated_ordm_1q.prods[i]
            if ordm_element_1q.coef != 0:
                print(f"\t{ordm_element_1q}")

    print("Evaluated 4-qubit 1RDM unmapped:")
    for i in range(len(evaluated_ordm_4q_unmapped.prods)):
        ordm_element_4q_unmapped = evaluated_ordm_4q_unmapped.prods[i]
        if np.abs(ordm_dict[ordm_element_4q_unmapped.ops_to_string()] - ordm_element_4q_unmapped.coef) > 1e-10:
            print(
                f"\tError in Mapping and Unmapping element {ordm_element_4q_unmapped.ops_to_string()}: {ordm_dict[ordm_element_4q_unmapped.ops_to_string()]} != {ordm_element_4q_unmapped.coef}")
            break
        if False:
            if ordm_element_4q_unmapped.coef != 0:
                print(f"\t{ordm_element_4q_unmapped}")
    print(f"\tTrace: {evaluated_ordm_4q_unmapped.trace()}")
    print("\t1rdm preserved in mapping and unmapping")

    trdm_4q = RDM(2, sdet1010.N)
    trdm_1q, inverse_mapping = trdm_4q.map(
        four_to_one, ignore_duplicates=True)

    print("2RDM Inverse Mapping:")
    for prod_ops, prod_mapping in inverse_mapping.items():
        print(f"\t{prod_ops}:")
        for term in prod_mapping:
            new_prod, prod, weight = term
            print(f"\t\t{new_prod} -> {prod} with weight {weight}")

    evaluated_trdm_4q = evaluate_rdm(psi_4q, trdm_4q)
    evaluated_trdm_1q = evaluate_rdm(psi_1q, trdm_1q)
    evaluated_trdm_4q_unmapped = evaluated_trdm_1q.unmap(inverse_mapping)

    trdm_dict = {}

    print("Evaluated 4-qubit 2RDM:")
    for i in range(len(evaluated_trdm_4q.prods)):
        trdm_element_4q = evaluated_trdm_4q.prods[i]
        trdm_dict[trdm_element_4q.ops_to_string()] = trdm_element_4q.coef
        if False:
            if trdm_element_4q.coef != 0:
                print(f"\t{trdm_element_4q}")
    print(f"\tTrace: {evaluated_trdm_4q.trace()}")

    if False:
        print("Evaluated 1-qubit 2RDM:")
        for i in range(len(evaluated_trdm_1q.prods)):
            trdm_element_1q = evaluated_trdm_1q.prods[i]
            if trdm_element_1q.coef != 0:
                print(f"\t{trdm_element_1q}")

    print("Evaluated 4-qubit 2RDM unmapped:")
    for i in range(len(evaluated_trdm_4q_unmapped.prods)):
        trdm_element_4q_unmapped = evaluated_trdm_4q_unmapped.prods[i]
        if np.abs(trdm_dict[trdm_element_4q_unmapped.ops_to_string()] - trdm_element_4q_unmapped.coef) > 1e-10:
            print(
                f"\tError in Mapping and Unmapping element {trdm_element_4q_unmapped.ops_to_string()}: {trdm_dict[trdm_element_4q_unmapped.ops_to_string()]} != {trdm_element_4q_unmapped.coef}")
            break
        if False:
            if trdm_element_4q_unmapped.coef != 0:
                print(f"\t{trdm_element_4q_unmapped}")
    print(f"\tTrace: {evaluated_trdm_4q_unmapped.trace()}")
    print("\t2rdm preserved in mapping and unmapping")


def main():
    print("Testing Projector class")
    test_projector_H2()

    print("\nTesting Projector class with time evolution")
    test_projector_H2_time_evolution()

    print("\nTesting Projector class with expectation values")
    test_projector_expectation_H2()


if __name__ == "__main__":
    main()
