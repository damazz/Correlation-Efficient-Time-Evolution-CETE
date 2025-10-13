from sparse_sim.cython.core import *


def test_pauli_string_initialization_scaling_freeing():
    paulis0 = ['I', 'X', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'X']
    pString1 = PauliString(4, 1 + 1j, paulis1)

    print(pString0)

    pString2 = (2 + 0j) * pString1
    print(f"2 * {pString1} = {pString2}")

    pString3 = pString2.adjoint()
    print(f"Adjoint of {pString2} = {pString3}")


def test_pauli_string_multiplication():
    paulis0 = ["I", "Y", "I", "X"]
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ["I", "X", "I", "I"]
    pString1 = PauliString(4, 1j, paulis1)

    pString2 = pString0 * pString1

    print(f"{pString0} * {pString1} = {pString2}")


def test_pauli_sum_initialization_scaling_freeing():
    paulis0 = ['I', 'Y', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'I']
    pString1 = PauliString(4, 1j, paulis1)

    pSum0 = PauliSum(4)
    pSum0.append_pauli_string(pString0)
    pSum0.append_pauli_string(pString1)

    print(f"{pSum0} contains {pSum0.p} pStrings")
    pSum1 = (2 + 0j) * pSum0
    print(f"2 * {pSum0} = {pSum1}")


def test_pauli_string_to_matrix():
    paulis0 = ['I', 'X']
    pString0 = PauliString(2, 1 + 0j, paulis0)

    paulis1 = ['Y', 'X']
    pString1 = PauliString(2, 0.5j, paulis1)

    print(f"{pString0} ->\n {pString0.matrix}")
    print(f"{pString1} ->\n {pString1.matrix}")


def test_pauli_sum_multiplication():
    paulis0 = ['I', 'Y', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'I']
    pString1 = PauliString(4, 1j, paulis1)

    pSum0 = PauliSum(4)
    pSum0.append_pauli_string(pString0)
    pSum0.append_pauli_string(pString1)

    paulis2 = ['I', 'Y', 'I', 'X']
    pString2 = PauliString(4, 1 + 0j, paulis2)

    paulis3 = ['I', 'X', 'I', 'I']
    pString3 = PauliString(4, 1j, paulis3)

    pSum1 = PauliSum(4)
    pSum1.append_pauli_string(pString2)
    pSum1.append_pauli_string(pString3)

    pSum2 = pSum0 * pSum1

    print(f"({pSum0}) * ({pSum1}) = {pSum2}")


def test_pauli_sum_addition():
    paulis0 = ['I', 'Y', 'I', 'X']
    pString0 = PauliString(4, 1 + 0j, paulis0)

    paulis1 = ['I', 'X', 'I', 'I']
    pString1 = PauliString(4, 1j, paulis1)

    pSum0 = PauliSum(4)
    pSum0.append_pauli_string(pString0)
    pSum0.append_pauli_string(pString1)

    paulis2 = ['I', 'Y', 'I', 'X']
    pString2 = PauliString(4, 1 + 0j, paulis2)

    paulis3 = ['I', 'X', 'I', 'I']
    pString3 = PauliString(4, -1j, paulis3)

    pSum1 = PauliSum(4)
    pSum1.append_pauli_string(pString2)
    pSum1.append_pauli_string(pString3)

    pSum2 = pSum0 + pSum1

    print(f"({pSum0}) + ({pSum1}) = {pSum2}")

    pSum3 = -1 * pSum2
    pSum4 = pSum2 + pSum3
    print(f"{pSum2} + {pSum3} = {pSum4}")


def test_pauli_sum_get_pauli_strings():
    paulis0 = ['Z', 'Y', 'I', 'X']
    paulis2 = ['Y', 'Y', 'I', 'X']
    paulis1 = ['I', 'X', 'Y', 'I']
    paulis3 = ['I', 'X', 'I', 'I']

    pString0 = PauliString(4, 1 + 0j, paulis0)
    print(paulis0)
    pString1 = PauliString(4, 1j, paulis1)

    print(paulis1)
    pString2 = PauliString(4, 1 + 0j, paulis2)

    print(paulis2)
    pString3 = PauliString(4, -1j, paulis3)

    print(paulis3)

    pSum0 = PauliSum(4)
    pSum0.append_pauli_string(pString0)
    pSum0.append_pauli_string(pString1)
    pSum0.append_pauli_string(pString2)
    pSum0.append_pauli_string(pString3)
    pauli_strings = pSum0.get_pauli_strings()
    if not pauli_strings:
        print("Error: Failed to retrieve Pauli strings.")
        return

    print(f"{pSum0} contains {pSum0.p} Pauli strings:")
    for i, p_str in enumerate(pauli_strings):
        print(f"Pauli string {i}: {p_str}")


def test_pauli_sum_to_matrix():
    paulis0 = ['I', 'X']
    pString0 = PauliString(2, 1 + 0j, paulis0)

    paulis1 = ['Y', 'X']
    pString1 = PauliString(2, 0.5j, paulis1)

    pSum0 = pString0 + pString1

    print(f"{pString0} ->\n {pString0.matrix}")
    print(f"{pString1} ->\n {pString1.matrix}")
    print(f"{pSum0} ->\n {pSum0.matrix}")


def main():
    print("\nTesting PauliString initialization, scaling, and freeing")
    test_pauli_string_initialization_scaling_freeing()

    print("\nTesting PauliString to Matrix")
    test_pauli_string_to_matrix()

    print("\nTesting PauliString Multiplication")
    test_pauli_string_multiplication()

    print("\nTesting PauliSum initialization, scaling, and freeing")
    test_pauli_sum_initialization_scaling_freeing()

    print("\nTesting PauliSum Multiplication")
    test_pauli_sum_multiplication()

    print("\nTesting PauliSum addition")
    test_pauli_sum_addition()

    print("\nTesting PauliSum get_pauli_strings")
    test_pauli_sum_get_pauli_strings()

    print("\nTesting PauliSum to Matrix")
    test_pauli_sum_to_matrix()


if __name__ == "__main__":
    main()
