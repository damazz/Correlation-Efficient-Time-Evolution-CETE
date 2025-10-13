
from pyscf import gto, scf, ao2mo
from pyscf.gto import Mole

from .fermion import *


class Hamiltonian(Operator):
    prods: list  # List of Products
    N: int  # Total number of sites / qubits
    symbol: str  # Symbol used for printing the operator
    nuc: float  # Nuclear-nuclear repulsion energy

    def __init__(self, prods, nuc, N, symbol="H"):
        super().__init__(prods, N, symbol)
        self.nuc = nuc

    def energy(self, tomography):
        return self.nuc + self.evaluate_expectation(tomography).real

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

        return Hamiltonian(new_prods, self.nuc, projector.target_N, f"{self.symbol}_mapped"), inverse_mapping

    def save(self):
        output = [np.array([self.N, self.nuc, self.symbol])]
        for prod in self.prods:
            output.append(prod.save())
        output = np.array(output, dtype=object)

        return output


def load_hamiltonian(data):
    row0 = data[0]
    N = int(row0[0])
    nuc = float(row0[1])
    symbol = row0[2]
    prods = []

    for i in range(1, len(data)):
        prod_data = data[i]
        prod = load_product(prod_data)
        prods.append(prod)
    return Hamiltonian(prods, nuc, N, symbol)


def init_Hamiltonian_from_pyscf(mol, mf=None):
    if mf is None:
        mf = scf.ROHF(mol)
        mf.verbose = 0
        mf.run()

    C = mf.mo_coeff

    h_core = mf.get_hcore()
    # Convert to MO basis
    h_mo = C.T @ h_core @ C

    Na = h_mo.shape[0]
    Nb = Na
    N = Na + Nb
    # Compute two-electron integrals in AO basis
    eri_ao = mol.intor("int2e")
    # Convert to MO basis, uncompress the tensor to full 4-index form
    eri_mo = ao2mo.incore.full(
        eri_ao, C, compact=False).reshape(Na, Na, Na, Na).transpose(0, 2, 1, 3)

    prods = []
    for i in range(Na):
        for j in range(Na):
            if abs(h_mo[i, j]) > 1e-10:
                fOp1 = FermionicOperator('+', i, N)
                fOp2 = FermionicOperator('-', j, N)
                prod = Product(h_mo[i, j], [fOp1, fOp2], N)
                prods.append(prod)

    for i in range(Nb):
        for j in range(Nb):
            if abs(h_mo[i, j]) > 1e-10:
                fOp1 = FermionicOperator('+', Na + i, N)
                fOp2 = FermionicOperator('-', Na + j, N)
                prod = Product(h_mo[i, j], [fOp1, fOp2], N)
                prods.append(prod)

    for i in range(Na):
        for j in range(Na):
            for k in range(Na):
                for l in range(Na):
                    if (i != j) and (l != k):
                        val = eri_mo[i, j, k, l]
                        if abs(val) > 1e-10:
                            fOp1 = FermionicOperator('+', i, N)
                            fOp2 = FermionicOperator('+', j, N)
                            fOp3 = FermionicOperator('-', l, N)
                            fOp4 = FermionicOperator('-', k, N)
                            prod = Product(
                                0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                            prods.append(prod)

    for i in range(Na):
        for j in range(Nb):
            for k in range(Na):
                for l in range(Nb):
                    val = eri_mo[i, j, k, l]
                    if abs(val) > 1e-10:
                        fOp1 = FermionicOperator('+', i, N)
                        fOp2 = FermionicOperator('+', Na + j, N)
                        fOp3 = FermionicOperator('-', Na + l, N)
                        fOp4 = FermionicOperator('-', k, N)
                        prod = Product(
                            0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                        prods.append(prod)

    for i in range(Nb):
        for j in range(Na):
            for k in range(Nb):
                for l in range(Na):
                    val = eri_mo[i, j, k, l]
                    if abs(val) > 1e-10:
                        fOp1 = FermionicOperator('+', Na + i, N)
                        fOp2 = FermionicOperator('+', j, N)
                        fOp3 = FermionicOperator('-', l, N)
                        fOp4 = FermionicOperator('-', Na + k, N)
                        prod = Product(
                            0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                        prods.append(prod)

    for i in range(Nb):
        for j in range(Nb):
            for k in range(Nb):
                for l in range(Nb):
                    if (i != j) and (l != k):
                        val = eri_mo[i, j, k, l]
                        if abs(val) > 1e-10:
                            fOp1 = FermionicOperator('+', Na + i, N)
                            fOp2 = FermionicOperator('+', Na + j, N)
                            fOp3 = FermionicOperator('-', Na + l, N)
                            fOp4 = FermionicOperator('-', Na + k, N)
                            prod = Product(
                                0.5 * val, [fOp1, fOp2, fOp3, fOp4], N)
                            prods.append(prod)

    return Hamiltonian(prods, mf.energy_nuc(), N)
