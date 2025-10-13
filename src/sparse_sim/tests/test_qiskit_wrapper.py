from mthree import M3Mitigation
import numpy as np
from pyscf import gto
import qiskit as q
import qiskit_aer as Aer
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from sparse_sim.fermion.qiskit_wrapper import *
from sparse_sim.fermion.hamiltonian import *
from sparse_sim.fermion.rdm import *


def test_qiskit_expectation():
    fOp1 = FermionicOperator("+", 0, 2)
    fOp2 = FermionicOperator("-", 0, 2)
    fOp3 = FermionicOperator("+", 1, 2)
    fOp4 = FermionicOperator("-", 1, 2)

    fProd1 = Product(1, [fOp1, fOp2], 2)
    fProd2 = Product(1, [fOp3, fOp4], 2)
    fProd3 = Product(1, [fOp3, fOp2], 2)
    fProd4 = Product(1, [fOp1, fOp4], 2)

    op1 = Operator([fProd1, fProd2, fProd3, fProd4], 2)
    op2 = Operator([fProd1, fProd2], 2)

    orbitals = [1, 0]
    sdet = SlaterDeterminant(2, 1 + 0j, orbitals)

    paulis0 = ["X", "X"]
    pString0 = PauliString(2, 1j * np.pi / 4, paulis0)
    paulis1 = ["I", "I"]
    pString1 = PauliString(2, 1j * 1, paulis1)
    pSum = pString0 + pString1

    circuit = q.QuantumCircuit(2, 2)
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))
    backend = Aer.AerSimulator()
    pm = generate_preset_pass_manager(backend=backend)

    measurements_r = op1.aggregate_measurements_recursive()
    print(f"fProd1: {fProd1.pSum}")
    print(f"fProd2: {fProd2.pSum}")
    print(f"fProd3: {fProd3.pSum}")
    print(f"fProd4: {fProd4.pSum}")

    tomography = qiskit_perform_tomography(
        circuit, measurements_r, backend, pm)
    print(
        f"Tomography data:\n {[f'{key}: {value}' for key, value in tomography.items()]}")

    statevector = qiskit_statevector(circuit)
    print(f"Statevector: {statevector}")

    print(f"{fProd3.evaluate_expectation(tomography)}")

    print(f"Op1 = {op1.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)} + {fProd3.evaluate_expectation(tomography)} + {fProd4.evaluate_expectation(tomography)}")
    print(f"Op2 = {op2.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)}")


def test_qiskit_probability_distribution():
    paulis0 = ["X", "X"]
    pString0 = PauliString(2, 1j * np.pi / 8, paulis0)
    paulis1 = ["I", "I"]
    pString1 = PauliString(2, 1j * 1, paulis1)
    pSum = pString0 + pString1

    sdet = SlaterDeterminant(2, 1 + 0j, [1, 0])

    circuit = q.QuantumCircuit(2, 0)
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))
    backend = Aer.AerSimulator()
    pm = generate_preset_pass_manager(backend=backend)

    shots_list = [2**9, 2**11, 2**13, 2**15]
    statevector = qiskit_statevector(circuit)
    prob_statevector = slater_determinant_probability_from_statevector(
        sdet, statevector)
    for shots in shots_list:
        prob_dist = qiskit_probability_distribution(
            circuit, backend, pm, shots=shots)
        prob = slater_determinant_probability(sdet, prob_dist)
        print(f"Running with {shots} shots:")
        print(f"P_|10> = {prob}~{prob_statevector}")


def test_qiskit_statevector_expectation():
    fOp1 = FermionicOperator("+", 0, 2)
    fOp2 = FermionicOperator("-", 0, 2)
    fOp3 = FermionicOperator("+", 1, 2)
    fOp4 = FermionicOperator("-", 1, 2)

    fProd1 = Product(1, [fOp1, fOp2], 2)
    fProd2 = Product(1, [fOp3, fOp4], 2)
    fProd3 = Product(1, [fOp3, fOp2], 2)
    fProd4 = Product(1, [fOp1, fOp4], 2)

    op1 = Operator([fProd1, fProd2, fProd3, fProd4], 2)
    op2 = Operator([fProd1, fProd2], 2)

    orbitals = [1, 0]
    sdet = SlaterDeterminant(2, 1 + 0j, orbitals)

    paulis0 = ["X", "X"]
    pString0 = PauliString(2, 1j * np.pi / 4, paulis0)
    paulis1 = ["I", "I"]
    pString1 = PauliString(2, 1j * 1, paulis1)
    pSum = pString0 + pString1

    circuit = q.QuantumCircuit(2, 2)
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    circuit = circuit.compose(qiskit_create_pauli_sum_evolution_circuit(pSum))

    measurements_r = op1.aggregate_measurements_recursive()
    print(f"fProd1: {fProd1.pSum}")
    print(f"fProd2: {fProd2.pSum}")
    print(f"fProd3: {fProd3.pSum}")
    print(f"fProd4: {fProd4.pSum}")

    tomography = qiskit_perform_tomography_statevector(circuit, measurements_r)
    print(
        f"Tomography data:\n {[f'{key}: {value}' for key, value in tomography.items()]}")

    print(f"Op1 = {op1.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)} + {fProd3.evaluate_expectation(tomography)} + {fProd4.evaluate_expectation(tomography)}")
    print(f"Op2 = {op2.evaluate_expectation(tomography)} = {fProd1.evaluate_expectation(tomography)} + {fProd2.evaluate_expectation(tomography)}")


def test_qiskit_noisy_probability_distribution():
    paulis0 = ["X"]
    pString0 = PauliString(1, 1j * np.pi / 8, paulis0)
    paulis1 = ["X"]
    pString1 = PauliString(1, -1j * np.pi / 8, paulis1)
    pSum = pString0 + pString1

    sdet = SlaterDeterminant(1, 1 + 0j, [0])

    circuit = q.QuantumCircuit(1, 0)
    circuit = circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(sdet))
    for i in range(3):
        circuit = circuit.compose(
            qiskit_create_pauli_sum_evolution_circuit(pSum))
        circuit.barrier()

    service = QiskitRuntimeService(channel="ibm_quantum")
    full_backend = service.backend("ibm_marrakesh")
    full_noise_model = NoiseModel.from_backend(full_backend)
    trimmed_noise_model = trim_noise_model(full_noise_model, {0: 0})
    gates = ['id', 'sx', 'x', 'cz', 'rz']

    backend = Aer.AerSimulator(
        method='density_matrix', noise_model=trimmed_noise_model, basis_gates=gates, n_qubits=1)
    mit = M3Mitigation(backend)
    cals_from_noise_model(mit, trimmed_noise_model)

    statevector = qiskit_statevector(circuit)
    print(f"Statevector: {statevector}")
    print(
        f"P_|10> = {slater_determinant_probability_from_statevector(sdet, statevector)}")
    prob_dist = qiskit_probability_distribution(circuit, backend, 2**15)
    print(f"Probability Distribution: {prob_dist}")
    print(f"P_|10> = {slater_determinant_probability(sdet, prob_dist)}")


def evaluate_ordm_statevector(psi_circuit: q.QuantumCircuit, ordm: RDM):

    ordm_tomography = qiskit_perform_tomography_statevector(
        psi_circuit, ordm.aggregate_measurements_recursive())

    new_prods = []
    for prod in ordm.prods:
        d = prod.evaluate_expectation(ordm_tomography)
        new_prod = d * prod
        new_prods.append(new_prod)

    return RDM(1, ordm.N, new_prods)


def evaluate_hamiltonian_statevector(psi_circuit: q.QuantumCircuit, H: Hamiltonian):
    hamiltonian_tomography = qiskit_perform_tomography_statevector(
        psi_circuit, H.aggregate_measurements())

    energy = H.evaluate_expectation(hamiltonian_tomography)

    return energy


def test_qiskit_pauli_sum_evolution():
    H4 = gto.M(
        atom="H 0 0 0; H 0 0 0.7348654; H 0 0 1.4697308; H 0 0 2.2045962",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )
    H4.build()

    H = init_Hamiltonian_from_pyscf(H4)

    hf_sdet = SlaterDeterminant(8, 1, [1, 1, 0, 0, 1, 1, 0, 0])

    circ = q.QuantumCircuit(8, 0)
    circ.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(hf_sdet), inplace=True)

    ordm = RDM(1, 8)

    initial_ordm = evaluate_ordm_statevector(circ, ordm)
    initial_energy = evaluate_hamiltonian_statevector(circ, H)

    print(
        f"Initial electron count, energy: {initial_ordm.trace()}, {initial_energy}")

    total_time = 100
    circ_exact = circ.compose(
        qiskit_create_pauli_sum_evolution_circuit_exact(H.pSum, 1j * total_time), inplace=False)

    exact_ordm = evaluate_ordm_statevector(circ_exact, ordm)
    exact_energy = evaluate_hamiltonian_statevector(circ_exact, H)
    print(
        f"Exact electron count, energy: {exact_ordm.trace()}, {exact_energy}")

    trotterization_list = [1, 10, 100, 1000]
    for trotter_steps in trotterization_list:
        circ_trot = circ.copy()
        evolution_gate = qiskit_create_pauli_sum_evolution_circuit(
            H.pSum, 1j * total_time / trotter_steps)
        for _ in range(trotter_steps):
            circ_trot.compose(evolution_gate, inplace=True)
        trotter_ordm = evaluate_ordm_statevector(circ_trot, ordm)
        trotter_energy = evaluate_hamiltonian_statevector(circ_trot, H)
        print(
            f"Trotterization with {trotter_steps} steps: electron count = {trotter_ordm.trace()}, energy = {trotter_energy}")


def main():
    print("Testing Qiskit Expectation")
    test_qiskit_expectation()
    print("\nTest Qiskit Probability Distribution")
    test_qiskit_probability_distribution()
    print("\nTesting Qiskit Statevector Expectation")
    test_qiskit_statevector_expectation()
    # print("\nTesting Qiskit Noisy Probability Distribution")
    # test_qiskit_noisy_probability_distribution()
    print("\nTesting Qiskit Pauli Sum Evolution")
    test_qiskit_pauli_sum_evolution()


if __name__ == "__main__":
    main()
