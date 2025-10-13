import numbers
import numpy as np

from .projector import *
from ..cython.core import *


class FermionicOperator:
    N: int  # Total number of sites / qubits
    op: str  # '+' for creation, '-' for annihilation
    idx: int  # Index of the site
    pSum: PauliSum  # PauliSum object representing the operator as PauliStrings using Jordan-Wigner Transformation

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

        for i in range(self.idx):
            paulis[i] = "Z"

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
        return FermionicOperator(adjoint_op, self.idx, self.N)

    def __str__(self):
        return f"{self.op}a_{self.idx}"

    def save(self):
        return np.array(['f', self.N, self.op, self.idx])


def load_fermionic_operator(data):
    assert data[0] == 'f'
    N = int(data[1])
    op = data[2]
    idx = int(data[3])
    return FermionicOperator(op, idx, N)


class Product:
    N: int  # Total number of sites / qubits
    coef: complex  # Complex coefficient of the product
    ops: list  # List of Operators applied right to left
    pSum: PauliSum  # PauliSum object representing the product as PauliStrings using Jordan-Wigner Transformation

    def __init__(self, coef, ops, N, evaluate_pSum=True):

        self.coef = coef
        self.ops = ops
        self.N = N
        if evaluate_pSum:
            self.pSum = self.to_pSum()

    def to_pSum(self):

        pSum = self.ops[0].pSum
        pSum = pauli_sum_scalar_multiplication(pSum, self.coef)

        for i in range(1, len(self.ops)):
            next_pSum = self.ops[i].pSum
            pSum = pauli_sum_multiplication(pSum, next_pSum)

        return pSum

    def adjoint(self):
        adjoint_coef = np.conj(self.coef)
        adjoint_ops = []
        for op in self.ops[::-1]:
            adjoint_ops.append(op.adjoint())
        return Product(adjoint_coef, adjoint_ops, self.N)

    def evaluate_expectation(self, tomography):
        return pauli_sum_evaluate_expectation(self.pSum, tomography)

    def __str__(self):
        output = f"{self.coef} * ("
        for i, op in enumerate(self.ops):
            if i > 0:
                output += " "
            output = output + f"{op}"
        output = output + ")"
        return output

    def ops_to_string(self):
        output = ""
        for i, op in enumerate(self.ops):
            output = output + f"{op}"
        return output

    def multiply_by_scalar(self, scalar):
        new_prod = Product(
            scalar * self.coef, self.ops, self.N, False)
        new_prod.pSum = pauli_sum_scalar_multiplication(self.pSum, scalar)
        return new_prod

    def __mul__(self, right):
        if isinstance(right, numbers.Number):
            return self.multiply_by_scalar(right)
        else:
            raise TypeError(f"prod * {type(right)} is not defined")

    def __rmul__(self, left):
        if isinstance(left, numbers.Number):
            return self.multiply_by_scalar(left)
        else:
            raise TypeError(f"{type(left)} * prod is not defined")

    def map(self, projector, coef_seperate=False):

        prods = []
        if coef_seperate:
            coefs = []

        for (original_sdet_right, target_sdet_right) in projector.mapping:
            for (original_sdet_left, target_sdet_left) in projector.mapping:
                bra = Wavefunction(
                    self.N) + SlaterDeterminant(self.N, 1, original_sdet_left.orbitals)
                ket = Wavefunction(
                    self.N) + SlaterDeterminant(self.N, 1, original_sdet_right.orbitals)

                coef = bra * (self.pSum * ket)
                if np.abs(coef) > 1e-10:
                    bOps = slater_determinant_outer_product_to_bosonic_ops(
                        target_sdet_left, target_sdet_right)
                    if coef_seperate:
                        coefs.append(coef)
                        prod = Product(1, bOps, projector.target_N)
                    else:
                        prod = Product(coef, bOps, projector.target_N)
                    prods.append(prod)

        if coef_seperate:
            return coefs, prods
        else:
            return prods

    def save(self):
        output = np.array([self.N, self.coef])
        for op in self.ops:
            output = np.concatenate((output, op.save()))
        return output

    def to_tensor(self):
        dims = len(self.ops)
        N = self.N
        tensor = np.zeros((N,) * dims, dtype=np.complex128)
        indices = [op.idx for op in self.ops]
        tensor[tuple(indices)] = self.coef
        return tensor


def load_product(data):
    if data.dtype == '<U64':
        N = int(np.complex128(data[0]).real)
    elif data.dtype == '<U21':
        N = int(data[0])
    else:
        print(f"Data type mismatch: {data.dtype}")

    coef = np.complex128(data[1])

    ops = []
    num_ops = (len(data) - 2) // 4
    for i in range(num_ops):
        idx = 2 + i * 4
        op_data = data[idx:idx+4]
        if op_data[0] == 'f':
            op = load_fermionic_operator(op_data)
        elif op_data[0] == 'b':
            op = load_bosonic_operator(op_data)
        else:
            raise ValueError(f"Unknown operator type: {op_data[0]}")
        ops.append(op)

    prod = Product(coef, ops, N)

    return prod


class Operator:
    N: int  # Total number of sites / qubits
    prods: list  # List of Products
    symbol: str  # Symbol used for printing the operator
    pSum: PauliSum  # PauliSum object representing the Operator as PauliStrings using Jordan-Wigner Transformation

    def __init__(self, prods, N, symbol="Symbol Not Set"):

        self.prods = prods
        self.N = N
        self.symbol = symbol
        self.pSum = self.to_pSum()

    def to_pSum(self):
        pSum = PauliSum(self.N)
        for prod in self.prods:
            pSum = pSum + prod.pSum
        return pSum

    def add_product(self, prod):
        self.prods.append(prod)

    def aggregate_measurements_recursive(self):
        measurements = set()
        for prod in self.prods:
            pSum = prod.pSum
            measurements.update(pauli_sum_collect_measurements(pSum))

        return measurements

    def aggregate_measurements(self):
        return pauli_sum_collect_measurements(self.pSum)

    def evaluate_expectation(self, tomography):
        return pauli_sum_evaluate_expectation(self.pSum, tomography)

    def adjoint(self):
        adjoint_prods = []
        for prod in self.prods[::-1]:
            adjoint_prods.append(prod.adjoint())
        return Operator(adjoint_prods, self.N)

    def __str__(self):
        return self.symbol

    def map(self, projector):
        new_prods_dict = {}

        inverse_mapping = {}

        for prod in self.prods:
            new_coefs, new_prods = prod.map(projector, coef_seperate=True)
            for i, new_prod in enumerate(new_prods):
                new_prod_ops = new_prod.ops_to_string()
                if new_prod_ops in new_prods_dict.keys():
                    new_prods_dict[new_prod_ops][1] += new_coefs[i]
                    inverse_mapping[new_prod_ops].append(
                        [new_prod, prod, new_coefs[i]])
                else:
                    new_prods_dict[new_prod_ops] = [new_prod, new_coefs[i]]
                    inverse_mapping[new_prod_ops] = [
                        [new_prod, prod, new_coefs[i]]]

        new_prods = []
        for new_prod_ops, [new_prod, coef] in new_prods_dict.items():
            new_prod = coef * new_prod
            new_prods.append(new_prod)

        for new_prod_ops, prod_mapping in inverse_mapping.items():
            total_coef = sum(coef for _, __, coef in prod_mapping)
            if total_coef != 0:
                inverse_mapping[new_prod_ops] = [
                    [total_coef * new_prod, prod, coef / total_coef] for new_prod, prod, coef in prod_mapping
                ]
            else:
                inverse_mapping[new_prod_ops] = [
                    [total_coef * new_prod, prod, 0.0] for new_prod, prod, coef in prod_mapping
                ]

        return Operator(new_prods, projector.target_N, f"{self.symbol}_mapped"), inverse_mapping

    def unmap(self, inverse_mapping):
        unmapped_prods_dict = {}
        unmapped_N = -1

        for i, prod in enumerate(self.prods):
            prod_ops = prod.ops_to_string()
            # print(f"\t{prod_ops}")
            if prod_ops in inverse_mapping.keys():
                for mapped_prod, original_prod, weight in inverse_mapping[prod_ops]:
                    unmapped_prod_ops = original_prod.ops_to_string()
                    if unmapped_prod_ops not in unmapped_prods_dict.keys():
                        unmapped_prods_dict[unmapped_prod_ops] = [
                            original_prod, weight * prod.coef]
                    else:
                        unmapped_prods_dict[unmapped_prod_ops][1] += weight * prod.coef

        unmapped_prods = []
        for unmapped_prod_ops, (original_prod, coef) in unmapped_prods_dict.items():
            unmapped_prods.append(coef * original_prod)
            if unmapped_N == -1:
                unmapped_N = original_prod.N

        return Operator(unmapped_prods, unmapped_N, f"{self.symbol}_unmapped")

    def diagonal_elements(self):

        diagonal_elements = {}

        for prod in self.prods:
            indices = []
            for i, op in enumerate(prod.ops):
                indices.append(op.idx)
            num_indices = len(indices)

            diagonal = True
            for i in range(num_indices // 2):
                if indices[i] != indices[num_indices - 1 - i]:
                    diagonal = False
                    break

            if diagonal:
                if prod.ops_to_string() in diagonal_elements.keys():
                    print(
                        f"Warning: Product {prod.ops_to_string()} is already in diagonal_elements, overwriting.")
                diagonal_elements[prod.ops_to_string()] = prod

        return diagonal_elements

    def trace(self):
        diagonal_elements = self.diagonal_elements()

        tr = 0.0 + 0.0j
        for prod in diagonal_elements.values():
            tr += prod.coef

        return tr

    def save(self):
        output = [np.array([self.N, self.symbol])]
        for prod in self.prods:
            output.append(prod.save())
        output = np.array(output, dtype=object)

        return output

    # check implementation at some point
    def to_tensor(self):
        tensor = self.prods[0].to_tensor()
        for prod in self.prods[1:]:
            next_tensor = prod.to_tensor()
            # Assuming we want to add tensors together, this will depend on the specific implementation
            tensor += next_tensor
        return tensor


def load_operator(data):
    row0 = data[0]
    N = int(row0[0])
    symbol = row0[1]
    prods = []

    for i in range(1, len(data)):
        prod_data = data[i]
        prod = load_product(prod_data)
        prods.append(prod)
    return Operator(prods, N, symbol)


def save_inverse_mapping(inverse_mapping):

    saveable_inverse_mapping = {}

    for new_prod_ops, terms in inverse_mapping.items():
        saveable_inverse_mapping[new_prod_ops] = [
            [new_prod.save(), prod.save(), coef] for new_prod, prod, coef in terms]

    return saveable_inverse_mapping


def load_inverse_mapping(saveable_inverse_mapping):
    inverse_mapping = {}

    for new_prod_ops, terms in saveable_inverse_mapping.items():
        inverse_mapping[new_prod_ops] = [[load_product(
            new_prod), load_product(prod), coef] for new_prod, prod, coef in terms]

    return inverse_mapping
