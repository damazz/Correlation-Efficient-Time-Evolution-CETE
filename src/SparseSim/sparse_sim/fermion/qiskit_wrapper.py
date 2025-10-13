from qiskit.quantum_info import Statevector
import copy
from collections import Counter
from mthree import M3Mitigation
from mthree.utils import final_measurement_mapping
from multiprocessing import Pool
import numpy as np
import os
import qiskit as q
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator, Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import qiskit_aer as Aer
from scipy.optimize import curve_fit
from scipy.linalg import expm
from typing import Set
import warnings
from scipy.optimize import OptimizeWarning

from .fermion import *
from ..sparse_sim import *

warnings.filterwarnings("ignore", category=OptimizeWarning)


class InitOperators:
    ops: list  # List of Operators
    epsilons: list  # List of epsilon values
    N: int  # Total number of sites / qubits
    M: int  # Number of Operators
    direction: str  # Direction of the circuit initialization
    exact: bool  # Whether to use exact pauli sum evolution or not

    def __init__(self, N, exact=False):
        self.N = N
        self.M = 0
        self.ops = []
        self.epsilons = []
        self.pSum_direction = 'forward'
        self.exact = exact

    def add_operator(self, operator, epsilon):
        self.ops.append(operator)
        self.epsilons.append(epsilon)
        self.M += 1

    def add_operator_at_beginning(self, operator, epsilon):
        self.ops.insert(0, operator)
        self.epsilons.insert(0, epsilon)
        self.M += 1

    def adjoint(self):
        adjoint_init_ops = InitOperators(self.N)
        adjoint_init_ops.exact = self.exact
        for m in range(self.M):
            adjoint_init_ops.add_operator(
                self.ops[self.M-m-1].adjoint(), self.epsilons[self.M-m-1].conj())
        if self.pSum_direction == 'forward':
            adjoint_init_ops.pSum_direction = 'backwards'
        elif self.pSum_direction == 'backward':
            adjoint_init_ops.pSum_direction = 'forwards'
        else:
            raise ValueError("Invalid initialization direction")
        return adjoint_init_ops

    def create_initialization_circuit(self):
        N = self.N
        circ = q.QuantumCircuit(N, 0)
        if self.exact:
            if self.pSum_direction == 'forwards':
                for m in range(self.M):
                    circ = circ.compose(
                        qiskit_create_pauli_sum_evolution_circuit_exact(self.ops[m].pSum, self.epsilons[m]))
            elif self.pSum_direction == 'backwards':
                for m in range(self.M):
                    circ = circ.compose(
                        qiskit_create_backwards_pauli_sum_evolution_circuit_exact(self.ops[m].pSum, self.epsilons[m]))
        else:
            if self.pSum_direction == 'forwards':
                for m in range(self.M):
                    circ = circ.compose(
                        qiskit_create_pauli_sum_evolution_circuit(self.ops[m].pSum, self.epsilons[m]))
            elif self.pSum_direction == 'backwards':
                for m in range(self.M):
                    circ = circ.compose(
                        qiskit_create_backwards_pauli_sum_evolution_circuit(self.ops[m].pSum, self.epsilons[m]))

        return circ

    def to_string(self):
        output = ""
        for m in range(self.M):
            output += f"op {m}:\n {self.epsilons[m]} * (\n"
            op = self.ops[m]
            for prod in op.prods:
                output += f"{prod}\n"
            output += ")\n"
        return output

    def __str__(self):
        return self.to_string()


def qiskit_create_initialization_from_slater_determinant_circuit(sDet: SlaterDeterminant):
    N = sDet.N
    circ = q.QuantumCircuit(N, 0)
    orbitals = sDet.orbitals
    for i in range(N):
        if orbitals[i] == 1:
            circ.x(i)
    return circ


def qiskit_create_pauli_string_evolution_circuit(pString: PauliString, epsilon: np.complex128 = 1.0):
    N = pString.N
    circ = q.QuantumCircuit(N, 0)

    coef = pString.coef
    theta = -1 * (coef * epsilon).imag
    string = pString.string

    q_pString = SparsePauliOp(string[::-1])
    rotation = PauliEvolutionGate(q_pString, theta)
    circ.append(rotation, range(N))
    return circ.decompose()


def qiskit_create_pauli_sum_evolution_circuit(pSum: PauliSum, epsilon: np.complex128 = 1.0):

    pStrings = pSum.get_pauli_strings()

    circ = q.QuantumCircuit(pSum.N, 0)
    for pString in pStrings:
        circ.compose(
            qiskit_create_pauli_string_evolution_circuit(pString, epsilon), inplace=True)
    return circ


def qiskit_create_backwards_pauli_sum_evolution_circuit(pSum: PauliSum, epsilon: np.complex128 = 1.0):

    pStrings = pSum.get_pauli_strings()[::-1]

    circ = q.QuantumCircuit(pSum.N, 0)
    for pString in pStrings:
        circ = circ.compose(
            qiskit_create_pauli_string_evolution_circuit(pString, epsilon))
    return circ


def qiskit_create_pauli_sum_evolution_circuit_exact(pSum: PauliSum, epsilon: np.complex128 = 1.0):

    circ = q.QuantumCircuit(pSum.N, 0)

    pStrings = pSum.get_pauli_strings()
    coefs = [pString.coef for pString in pStrings]
    pStrings_little_endian = []
    for pString in pStrings:
        pStrings_little_endian.append(pString.string[::-1])

    q_pSum = SparsePauliOp(pStrings_little_endian, coeffs=coefs)
    q_matrix = q_pSum.to_matrix()

    evolution = expm(q_matrix * epsilon)

    circ.append(UnitaryGate(evolution), circ.qubits)
    return circ


def qiskit_create_backwards_pauli_sum_evolution_circuit_exact(pSum: PauliSum, epsilon: np.complex128 = 1.0):
    circ = q.QuantumCircuit(pSum.N, 0)

    pStrings = pSum.get_pauli_strings()[::-1]
    coefs = [pString.coef for pString in pStrings]
    pStrings_little_endian = []
    for pString in pStrings:
        pStrings_little_endian.append(pString.string[::-1])

    q_pSum = SparsePauliOp(pStrings_little_endian, coeffs=coefs)
    q_matrix = q_pSum.to_matrix()

    evolution = expm(q_matrix * epsilon)

    circ.append(UnitaryGate(evolution), circ.qubits)
    return circ


def qiskit_pauli_string_measurement(circuit: q.QuantumCircuit, pString_as_string: str, backend, shots=2**13):
    estimator = Estimator(mode=backend, options={"default_shots": shots})

    observable = SparsePauliOp(pString_as_string[::-1])

    result = estimator.run([(circuit, observable)]).result()
    return result[0].data.evs


def qiskit_perform_tomography(circuit: q.QuantumCircuit, measurements: Set[str], backend, pm, shots=2**13):

    circuit = pm.run(circuit)

    estimator = Estimator(mode=backend, options={"default_shots": shots})

    id_string = "I" * len(next(iter(measurements)))

    circuits_with_measurements = []
    for measurement in measurements:
        if measurement != id_string:
            observable = SparsePauliOp(measurement[::-1])
            transpiled_observable = observable.apply_layout(circuit.layout)
            circuits_with_measurements.append((circuit, transpiled_observable))

    result = estimator.run(circuits_with_measurements).result()

    tomography = {}

    result_idx = 0
    for measurement in measurements:
        if measurement == id_string:
            tomography[measurement] = 1.0
        else:
            tomography[measurement] = result[result_idx].data.evs
            result_idx += 1

    return tomography


def _qiskit_probability_distribution_and_statevector_helper(circuits, backend, num_qubits, shots=2**13):
    # New instance per process
    job = backend.run(circuits, shots=shots)
    statevectors = [job.result().get_statevector(i)
                    for i in range(len(circuits))]
    prob_dists = []
    for i in range(len(circuits)):
        result = job.result().data(i)
        counts = result.get("counts")
        total_counts = sum(counts.values())
        prob_dist = {bin(int(state, 16))[2:].zfill(num_qubits)[::-1]: count /
                     total_counts for state, count in counts.items()}
        prob_dists.append(prob_dist)
    return statevectors, prob_dists


def qiskit_probability_distribution_and_statevector(circuit_or_circuits, backend, make_copy=True, number_of_processes=1, shots=2**13):

    circuits = []
    num_qubits = -1

    if isinstance(circuit_or_circuits, q.QuantumCircuit):
        if make_copy:
            circuit = circuit_or_circuits.copy()
        else:
            circuit = circuit_or_circuits
        circuit.save_statevector()
        circuit.measure_all()
        circuits.append(circuit)
        num_qubits = circuit.num_qubits
    else:
        for circuit in circuit_or_circuits:
            if make_copy:
                circuit = circuit.copy()
            circuit.save_statevector()
            circuit.measure_all()
            circuits.append(circuit)

            if num_qubits == -1:
                num_qubits = circuit.num_qubits

    if number_of_processes == 1:
        statevectors, prob_dists = _qiskit_probability_distribution_and_statevector_helper(
            circuits, backend, num_qubits, shots)
    else:
        num_circuits = len(circuits)
        batch_size = max(1, num_circuits // number_of_processes)
        batched_circuits = [circuits[i:i + batch_size]
                            for i in range(0, num_circuits, batch_size)]

        with Pool(processes=number_of_processes) as pool:
            results = pool.starmap(_qiskit_probability_distribution_and_statevector_helper,
                                   [(batch, backend, num_qubits, shots) for batch in batched_circuits])

        statevectors = [
            sv for batch_result in results for sv in batch_result[0]]
        prob_dists = [pd for batch_result in results for pd in batch_result[1]]

    if len(statevectors) == 1:
        return prob_dists[0], statevectors[0]
    else:
        return prob_dists, statevectors


def _qiskit_probability_distribution_helper_with_mit(circuits, backend, mit, shots=2**13):
    sampler = Sampler(mode=backend)
    result = sampler.run(circuits, shots=shots).result()
    prob_dists = []
    for circuit_idx in range(len(circuits)):
        circuit_result = result[circuit_idx]
        mapping = final_measurement_mapping([circuits[circuit_idx]])
        counts = Counter(circuit_result.data.meas.get_bitstrings())
        quasis = mit.apply_correction(counts, mapping)
        quasis = {state[::-1]: count for state, count in quasis.items()}
        prob_dists.append(quasis)
    return prob_dists


def _qiskit_probability_distribution_helper(circuits, backend, shots=2**13):
    sampler = Sampler(mode=backend)
    result = sampler.run(circuits, shots=shots).result()
    prob_dists = []
    for circuit_idx in range(len(circuits)):
        circuit_result = result[circuit_idx]
        counts = Counter(circuit_result.data.meas.get_bitstrings())
        total_counts = sum(counts.values())
        prob_dist = {state[::-1]: count /
                     total_counts for state, count in counts.items()}
        prob_dists.append(prob_dist)
    return prob_dists


def qiskit_probability_distribution(circuit_or_circuits, backend, pm, mit=None, make_copy=True, number_of_processes=1, shots=2**13):

    circuits = []

    if isinstance(circuit_or_circuits, q.QuantumCircuit):
        if make_copy:
            circuit = circuit_or_circuits.copy()
        else:
            circuit = circuit_or_circuits
        circuit.measure_all()
        pm_circuit = pm.run(circuit)
        circuits.append(pm_circuit)
    else:
        for circuit in circuit_or_circuits:
            if make_copy:
                circuit = circuit.copy()
            circuit.measure_all()
            pm_circuit = pm.run(circuit)
            circuits.append(pm_circuit)

    if number_of_processes == 1:
        if mit is not None:
            prob_dists = _qiskit_probability_distribution_helper_with_mit(
                circuits, backend, mit, shots)
        else:
            prob_dists = _qiskit_probability_distribution_helper(
                circuits, backend, shots)
    else:
        num_circuits = len(circuits)
        batch_size = max(1, num_circuits // number_of_processes)
        batched_circuits = [circuits[i:i + batch_size]
                            for i in range(0, num_circuits, batch_size)]

        if mit is not None:
            with Pool(processes=number_of_processes) as pool:
                results = pool.starmap(_qiskit_probability_distribution_helper_with_mit,
                                       [(batch, backend, mit, shots) for batch in batched_circuits])
            prob_dists = [
                pd for batch_result in results for pd in batch_result]
        else:
            with Pool(processes=number_of_processes) as pool:
                results = pool.starmap(_qiskit_probability_distribution_helper,
                                       [(batch, backend, shots) for batch in batched_circuits])
            prob_dists = [
                pd for batch_result in results for pd in batch_result]

    if len(prob_dists) == 1:
        return prob_dists[0]
    else:
        return prob_dists


def slater_determinant_probability(sDet: SlaterDeterminant, prob_dist):
    orbitals = sDet.orbitals
    bit_string = ''.join([str(i) for i in orbitals])
    if bit_string not in prob_dist:
        return 0
    return prob_dist[bit_string]


def _qiskit_statevector_helper(circuits):
    # New instance per process
    backend = Aer.AerSimulator(method='statevector')
    job = backend.run(circuits)
    result = job.result()
    return [result.get_statevector(i).data for i in range(len(circuits))]


def qiskit_statevector(circuit_or_circuits, make_copy=True, number_of_processes=1):

    circuits = []

    if isinstance(circuit_or_circuits, q.QuantumCircuit):
        if make_copy:
            circuit = circuit_or_circuits.copy()
        else:
            circuit = circuit_or_circuits
        circuit.save_statevector()
        circuits.append(circuit)
    else:
        for circuit in circuit_or_circuits:
            if make_copy:
                circuit = circuit.copy()
            circuit.save_statevector()
            circuits.append(circuit)

    if number_of_processes == 1:
        statevectors = _qiskit_statevector_helper(circuits)
    else:
        num_circuits = len(circuits)
        batch_size = max(1, num_circuits // number_of_processes)
        batched_circuits = [circuits[i:i + batch_size]
                            for i in range(0, num_circuits, batch_size)]

        with Pool(processes=number_of_processes) as pool:
            results = pool.map(
                _qiskit_statevector_helper, batched_circuits)

        statevectors = [sv for batch in results for sv in batch]

    if len(statevectors) == 1:
        return statevectors[0]
    else:
        return statevectors


def slater_determinant_probability_from_statevector(sDet: SlaterDeterminant, statevector):
    coef = statevector[sDet.encoding]
    return np.abs(coef)**2


def qiskit_perform_tomography_statevector(circuit: q.QuantumCircuit, measurements: Set[str]):
    circuit = circuit.copy()
    backend = Aer.AerSimulator(method='statevector')
    circuit.save_statevector()
    result = backend.run(circuit).result()
    statevector = result.get_statevector()

    tomography = {}
    for measurement in measurements:
        observable = SparsePauliOp(measurement[::-1])
        tomography[measurement] = statevector.expectation_value(observable)

    return tomography


def trim_noise_model(noise_model: Aer.noise.NoiseModel, qubit_mapping: dict, two_qubit_gates: bool = True):
    noise_model_as_dict = noise_model.to_dict()
    new_noise_model_as_dict = {}
    new_noise_model_as_dict["errors"] = []

    qubits_to_use = qubit_mapping.keys()

    for error in noise_model_as_dict["errors"]:
        qubits_in_error = error["gate_qubits"][0]
        if not two_qubit_gates and len(qubits_in_error) > 1:
            continue

        add_to_new_model = True
        for qubit_in_error in qubits_in_error:
            if qubit_in_error not in qubits_to_use:
                add_to_new_model = False
                break

        if add_to_new_model:
            new_error = copy.deepcopy(error)
            new_gate_qubits = tuple(qubit_mapping.get(q, q)
                                    for q in qubits_in_error)
            new_error["gate_qubits"][0] = new_gate_qubits
            new_noise_model_as_dict["errors"].append(new_error)

    new_noise_model = Aer.noise.NoiseModel.from_dict(new_noise_model_as_dict)
    return new_noise_model


def cals_from_noise_model(mitigation: M3Mitigation, noise_model: Aer.noise.NoiseModel):
    noise_model_as_dict = noise_model.to_dict()
    ro_probs = []
    for error in noise_model_as_dict["errors"]:
        if error["type"] == "roerror":
            probabilities = np.array(error["probabilities"])
            ro_probs.append(probabilities)
    mitigation.cals_from_matrices(ro_probs)


def qiskit_probability_distribution_from_id_helper_with_mit(circuits, job_id, pm, mit):
    service = QiskitRuntimeService()
    result = service.job(job_id).result()
    prob_dists = []
    for circuit_idx in range(len(circuits)):
        circuit_result = result[circuit_idx]
        mapping = final_measurement_mapping([pm.run(circuits[circuit_idx])])
        counts = Counter(circuit_result.data.meas.get_bitstrings())
        quasis = mit.apply_correction(counts, mapping)
        quasis = {state[::-1]: count for state, count in quasis.items()}
        prob_dists.append(quasis)
    return prob_dists
