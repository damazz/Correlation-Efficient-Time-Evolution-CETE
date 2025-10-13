
import itertools
import numpy as np

from .fermion import *


class RDM(Operator):
    N: int  # Total number of sites / qubits
    p: int  # Order of rdm
    prods: list  # List of FermionicProducts'
    symbol: str  # Symbol used for printing the operator

    def __init__(self, p, N, prods=[], ordered=False):
        self.p = p

        if len(prods) == 0:
            if ordered:
                prods = generate_prods_ordered(p, N)
            else:
                prods = generate_prods(p, N)

        super().__init__(prods, N, f"{p}D")

    def save(self):
        output = [np.array([self.N, self.p])]
        for prod in self.prods:
            output.append(prod.save())
        output = np.array(output, dtype=object)

        return output

    def map(self, projector, ignore_duplicates=True):

        new_prods_dict = {}

        inverse_mapping = {}

        for prod in self.prods:
            new_coefs, new_prods = prod.map(projector, coef_seperate=True)
            for i, new_prod in enumerate(new_prods):
                new_prod_ops = new_prod.ops_to_string()
                if new_prod_ops in new_prods_dict.keys():
                    if not ignore_duplicates:
                        new_prods_dict[new_prod_ops][1] += 1
                    inverse_mapping[new_prod_ops].append(
                        [new_prod, prod, new_coefs[i]])
                else:
                    new_prods_dict[new_prod_ops] = [new_prod, 1]
                    inverse_mapping[new_prod_ops] = [
                        [new_prod, prod, new_coefs[i]]]

        new_prods = []
        for new_prod_ops, (new_prod, coef) in new_prods_dict.items():
            if ignore_duplicates:
                new_prods.append(new_prod)
            else:
                new_prods.append(coef * new_prod)

        if not ignore_duplicates:
            for new_prod_ops, prod_mapping in inverse_mapping.items():
                total_coef = len(prod_mapping)
                inverse_mapping[new_prod_ops] = [
                    [total_coef * new_prod, prod, coef / total_coef] for new_prod, prod, coef in prod_mapping
                ]

        return RDM(self.p, projector.target_N, new_prods), inverse_mapping


def load_rdm(data):
    row0 = data[0]
    N = int(row0[0])
    p = int(row0[1])
    prods = []

    for i in range(1, len(data)):
        prod_data = data[i]
        prod = load_product(prod_data)
        prods.append(prod)
    return RDM(p, N, prods)


def generate_prods(p, N):
    prods = []

    indices = range(N)

    for idx_tuple in itertools.product(itertools.permutations(indices, p), itertools.permutations(indices, p)):
        fOps = []
        for idx, site in enumerate(idx_tuple[0] + idx_tuple[1]):
            sign = '+' if idx < p else '-'
            fOps.append(FermionicOperator(sign, site, N))
        prod = Product(1, fOps, N)
        prods.append(prod)

    return prods


def generate_prods(p, N):
    prods = []

    indices = range(N)

    for idx_tuple in itertools.product(itertools.permutations(indices, p), itertools.permutations(indices, p)):
        fOps = []
        for idx, site in enumerate(idx_tuple[0] + idx_tuple[1]):
            sign = '+' if idx < p else '-'
            fOps.append(FermionicOperator(sign, site, N))
        prod = Product(1, fOps, N)
        prods.append(prod)

    return prods


def generate_prods_ordered(p, N):
    prods = []

    indices = range(N)

    for idx_tuple in itertools.product(itertools.combinations(indices, p), itertools.combinations(indices, p)):
        fOps = []
        for idx, site in enumerate(idx_tuple[0] + idx_tuple[1][::-1]):
            sign = '+' if idx < p else '-'
            fOps.append(FermionicOperator(sign, site, N))
        prod = Product(1, fOps, N)
        prods.append(prod)

    return prods
