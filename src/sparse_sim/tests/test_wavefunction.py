from math import log2
import random
import time

from sparse_sim.sparse_sim import *


def test_initialization_scaling_freeing():
    orbitals0 = [0, 1, 0, 1]
    sdet0 = SlaterDeterminant(4, 1 + 0j, orbitals0)

    orbitals1 = [1, 0, 1, 0]
    sdet1 = SlaterDeterminant(4, 1j, orbitals1)

    wfn = Wavefunction(4)

    wfn.append_slater_determinant(sdet0)
    wfn.append_slater_determinant(sdet1)

    print(
        f"Wavefunction initialized with {wfn.s} sdets: {sdet0}, {sdet1}\n")

    print(wfn)

    norm = wfn.norm()
    print(f"Norm: {norm}")

    normalized_wfn = (1 / norm) * wfn
    print(f"Normalized: {normalized_wfn}")

    adjoint_wfn = normalized_wfn.adjoint()
    print(f"Adjoint: {adjoint_wfn}")


def test_inner_product():
    orbitals0 = [0, 1, 0, 1]
    sdet0 = SlaterDeterminant(4, 1 + 0j, orbitals0)

    orbitals1 = [1, 0, 1, 0]
    sdet1 = SlaterDeterminant(4, 1j, orbitals1)

    ket = Wavefunction(4)
    ket.append_slater_determinant(sdet0)
    ket.append_slater_determinant(sdet1)

    bra = ket.adjoint()

    inner_prod = bra * ket

    print(f"({bra}) * ({ket}) = {inner_prod}")


def test_appending_slater_determinants():
    print("Order of Adding:")

    wfn1 = Wavefunction(4)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    orbitals = [i, j, k, l]
                    sdet = SlaterDeterminant(4, 1 + 0j, orbitals)
                    wfn1.append_slater_determinant(sdet)

    wfn2 = Wavefunction(4)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    orbitals = [l, k, j, i]
                    sdet = SlaterDeterminant(4, 1 + 0j, orbitals)
                    wfn2.append_slater_determinant(sdet)

    print(wfn1)
    print()
    print(wfn2)

    wfn1 = wfn1.adjoint()

    product = wfn1 * wfn2
    print(f"Product: {product}")


def test_wavefunction_pauli_sum_multiplication():
    orbitals0 = [0, 1, 0, 1]
    orbitals1 = [1, 0, 1, 0]

    sdet0 = SlaterDeterminant(4, 1 + 0j, orbitals0)
    sdet1 = SlaterDeterminant(4, 1 + 0j, orbitals1)

    wfn = Wavefunction(4)
    wfn.append_slater_determinant(sdet0)
    wfn.append_slater_determinant(sdet1)

    paulis0 = ["I", "Y", "I", "X"]
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ["X", "I", "X", "I"]
    pString1 = PauliString(4, 1 + 0j, paulis1)

    pSum = PauliSum(4)
    pSum.append_pauli_string(pString0)
    pSum.append_pauli_string(pString1)

    new_wfn = pSum * wfn

    print(f"({pSum}) * ({wfn}) = {new_wfn}")


def test_wavefunction_pauli_string_evolution():
    orbitals0 = [0, 1, 0, 1]
    sdet0 = SlaterDeterminant(4, 1 + 0j, orbitals0)

    wfn = Wavefunction(4)
    wfn.append_slater_determinant(sdet0)

    paulis0 = ["X", "X", "X", "X"]
    pString = PauliString(4, 1j, paulis0)

    old_wfn = wfn

    for _ in range(100000):
        wfn = wavefunction_pauli_string_evolution(pString, wfn, 0.01)

    norm = wfn.norm()
    print(f"exp(0.01 * ({pString})) * ({old_wfn}) = {wfn}")
    print(f"Norm: {norm}")


def test_wavefunction_pauli_sum_evolution():
    orbitals0 = [0, 1, 0, 1]
    sdet0 = SlaterDeterminant(4, 1 + 0j, orbitals0)

    wfn = Wavefunction(4)
    wfn.append_slater_determinant(sdet0)

    paulis0 = ["X", "X", "X", "X"]
    pString0 = PauliString(4, 0.5j, paulis0)
    pString1 = PauliString(4, 0.5j, paulis0)

    pSum = PauliSum(4)
    pSum.append_pauli_string(pString0)
    pSum.append_pauli_string(pString1)

    old_wfn = wfn

    for _ in range(100000):
        wfn = wavefunction_pauli_sum_evolution(pSum, wfn, 0.01)

    norm = wfn.norm()
    print(f"exp(0.01 * ({pSum})) * ({old_wfn}) = {wfn}")
    print(f"Norm: {norm}")


def test_wavefunction_speed():
    N = 20  # Number of qubits
    num_sdets = 2  # Number of Slater determinants
    num_operations = 20  # Number of Pauli sum operations

    print(
        f"\nRunning Wavefunction Speed Test with {N} qubits, {num_sdets} Slater determinants, {num_operations} Pauli sum operations.")

    # Step 1: Initialize a large wavefunction
    wfn = Wavefunction(N)

    for _ in range(num_sdets):
        orbitals = [random.randint(0, 1) for _ in range(N)]
        coef = random.random() + random.random() * 1j
        sdet = SlaterDeterminant(N, coef, orbitals)
        wfn.append_slater_determinant(sdet)

    wfn = wavefunction_scalar_multiplication(wfn, 1 / wfn.norm())

    print("Wavefunction initialized.")

    # Step 2: Generate multiple Pauli sums
    pauli_sums = []
    for _ in range(num_operations):
        pSum = PauliSum(N)
        for _ in range(3):
            paulis = [random.choice(["I", "X", "Y", "Z"]) for _ in range(N)]
            coef = random.random() * 1j
            pString = PauliString(N, coef, paulis)
            pSum.append_pauli_string(pString)
        pauli_sums.append(pSum)

    print("Pauli sums generated.")

    # Step 3: Apply multiple Pauli sum multiplications
    start_time = time.time()

    for i, pSum in enumerate(pauli_sums):
        print(f"{i}: {pSum}")
        wfn = wavefunction_pauli_sum_evolution(pSum, wfn, 0.01)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"Completed {num_operations} Pauli sum multiplications in {elapsed_time:.4f} seconds.")
    print(f"Final wavefunction norm: {wfn.norm()}")
    print(
        f"Final number of Slater determinants is {wfn.s} showing a 2^{log2(wfn.s / num_sdets)} increase.")


def test_wavefunction_cleaning():
    orbitals0 = [0, 1, 1, 0]
    sdet0 = SlaterDeterminant(4, 0.5 + 0j, orbitals0)

    orbitals1 = [0, 1, 0, 1]
    sdet1 = SlaterDeterminant(4, 1e-4, orbitals1)

    orbitals2 = [1, 0, 1, 0]
    sdet2 = SlaterDeterminant(4, 1e-4j, orbitals2)

    orbitals0 = [0, 0, 1, 0]
    sdet3 = SlaterDeterminant(4, 0.5 + 0j, orbitals0)

    wfn = sdet0 + sdet1 + sdet2 + sdet3
    print(f"Initial: {wfn}")

    wfn_clean = wfn.remove_near_zero_terms(1e-3)
    print(f"Cleaned: {wfn_clean}")

    prob_dist = wavefunction_to_probability_distribution(wfn_clean)
    print(f"Probability Distribution: {prob_dist}")


def main():
    print("\nTesting Initialization, Scaling, and Freeing")
    test_initialization_scaling_freeing()

    print("\nTesting Inner Products")
    test_inner_product()

    print("\nTesting Appending Slater Determinants")
    test_appending_slater_determinants()

    print("\nTesting Wavefunction Multiplication by Pauli Sum")
    test_wavefunction_pauli_sum_multiplication()

    print("\nTesting Wavefunction Evolution by Pauli String")
    test_wavefunction_pauli_string_evolution()

    print("\nTesting Wavefunction Evolution by Pauli Sum")
    test_wavefunction_pauli_sum_evolution()

    # print("\nTesting Wavefunction Speed")
    # test_wavefunction_speed()

    print("\nTesting Wavefunction Cleaning")
    test_wavefunction_cleaning()


if __name__ == "__main__":
    main()
