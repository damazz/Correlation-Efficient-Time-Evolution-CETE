import numpy as np

from ..cython.core import *


class BosonicOperator:
    N: int  # Total number of sites / qubits
    op: str  # '+' for creation, '-' for annihilation
    idx: int  # Index of the site
    pSum: PauliSum  # PauliSum object representing the operator as PauliStrings
    # These are technically hardcore bosons as they are restricted to 0 and 1 occupation

    def __init__(self, operator, index, N):

        assert operator in [
            '+', '-'], "op must be '+' (creation) or '-' (annihilation)"
        assert 0 <= index < N, "site must be a valid qubit index"

        self.N = N
        self.op = operator
        self.idx = index
        self.pSum = self.to_pSum()

    def to_pSum(self):

        paulis = ["I"] * self.N
        pSum = PauliSum(self.N)

        paulis[self.idx] = "X"
        pString_x = PauliString(self.N, 0.5 + 0j, paulis)

        if self.op == "+":
            paulis[self.idx] = "Y"
            pString_y = PauliString(self.N, -0.5j, paulis)

        else:
            paulis[self.idx] = "Y"
            pString_y = PauliString(self.N, 0.5j, paulis)

        pSum.append_pauli_string(pString_x)
        pSum.append_pauli_string(pString_y)

        return pSum

    def adjoint(self):
        adjoint_op = "+"
        if self.op == "+":
            adjoint_op = "-"
        return BosonicOperator(adjoint_op, self.idx, self.N)

    def __str__(self):
        return f"{self.op}b_{self.idx}"

    def save(self):
        return np.array(['b', self.N, self.op, self.idx])


def load_bosonic_operator(data):
    assert data[0] == 'b'
    N = int(data[1])
    op = data[2]
    idx = int(data[3])

    return BosonicOperator(op, idx, N)


def slater_determinant_outer_product_to_bosonic_ops(ket: SlaterDeterminant, bra: SlaterDeterminant):

    assert ket.N == bra.N, "Number of sites must be the same for both Slater Determinants"

    N = ket.N
    ket_orbitals = ket.orbitals
    bra_orbitals = bra.orbitals

    bOps = []

    for idx in range(N):
        if ket_orbitals[idx] == 0 and bra_orbitals[idx] == 0:
            bOps.append(BosonicOperator('-', idx, N))
            bOps.append(BosonicOperator('+', idx, N))
        elif ket_orbitals[idx] == 1 and bra_orbitals[idx] == 1:
            bOps.append(BosonicOperator('+', idx, N))
            bOps.append(BosonicOperator('-', idx, N))
        elif ket_orbitals[idx] == 1 and bra_orbitals[idx] == 0:
            bOps.append(BosonicOperator('+', idx, N))
        elif ket_orbitals[idx] == 0 and bra_orbitals[idx] == 1:
            bOps.append(BosonicOperator('-', idx, N))
        else:
            raise ValueError("Invalid orbital configuration")

    return bOps


class Projector:
    mapping: list  # List of Tuples containing the mapping from one basis to another basis of states
    original_N: int  # Number of sites in the original basis
    target_N: int  # Number of sites in the target basis

    def __init__(self, original_N, target_N):
        self.mapping = []
        self.original_N = original_N
        self.target_N = target_N

    def add_mapping(self, mapping_pair):
        self.mapping.append(mapping_pair)
        return self

    def __add__(self, right):
        if isinstance(right, tuple):
            return self.add_mapping(right)
        else:
            TypeError(f"Projector + {type(right)} is not defined")

    def to_string(self):
        output = ""
        for mapping_pair in self.mapping:
            output += f"{mapping_pair[0]} -> {mapping_pair[1]}\n"
        return output

    def __str__(self):
        return self.to_string()
