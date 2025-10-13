
import numpy as np
import qiskit as q
import time
import scipy.stats as stats
from scipy.special import erfinv
import sys
from datetime import datetime

from sparse_sim.fermion.hamiltonian import *
from sparse_sim.fermion.qiskit_wrapper import *
from sparse_sim.fermion.rdm import *
from sparse_sim.cython.core import *

MAX_CIRCUITS = 300
FALSE_POSITIVE_REJECTION_RATE = 0.3
NUMBER_PROCESSES = 1
EPSILON = 0.3
EPSILON = np.complex128(EPSILON)  # Ensure type
print_time = True

GATES = ['id', 'sx', 'x', 'cz', 'rz']  # Basis gates for transpilation


def format_statevector(statevector, N):
    output = ""
    for i, sdet in enumerate(statevector):
        if np.abs(sdet) > 1e-4:
            bits = bin(i)[2:].zfill(N)[::-1]
            if len(output) > 0:
                output += " + "
            output += f"{sdet:.4f} * |{bits}>"
    return output


def create_statevector(phi, phi_circuit):
    circ = q.QuantumCircuit(phi.N, 0)
    circ = circ.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(phi), inplace=False)
    circ = circ.compose(phi_circuit, inplace=False)
    return qiskit_statevector(circ)


def calculate_cutoff(false_positive_rejection_rate: float, shots: int, d1: float, d2: float):
    sigma = np.sqrt((d1**2 + d2**2) / (2 * shots))
    z = np.sqrt(2.0) * erfinv(false_positive_rejection_rate)
    return z * sigma


def calculate_residual_adjoints(phi0: SlaterDeterminant, phi_circuit_adjoint: q.QuantumCircuit, psi_circuit: q.QuantumCircuit, parameters: np.array, trdm: RDM, optimization_level: int, shots: int, backend, pm, mit):

    if print_time:
        start_time = time.time()

    theta1, theta2, d1, d2, d_iters1, d_iters2 = parameters
    assert d_iters1 == 1 and d_iters2 == 1

    cutoff = calculate_cutoff(FALSE_POSITIVE_REJECTION_RATE, shots, d1, d2)

    def create_circuit(phi_circuit_adjoint, residual_adjoints_circuit, pSum, psi_circuit, theta):
        circ = psi_circuit.copy()
        circ.compose(qiskit_create_pauli_sum_evolution_circuit(
            pSum, theta), inplace=True)
        circ.compose(residual_adjoints_circuit, inplace=True)
        circ.compose(phi_circuit_adjoint, inplace=True)
        return circ

    def process_circuits(circuits, shots, backend, pm, mit):
        num_procs = NUMBER_PROCESSES
        print(
            f"\tUsing {num_procs} processes for qiskit calculation of {len(circuits)} circuits")
        prob_dists = qiskit_probability_distribution(
            circuits, backend, pm, mit=mit, make_copy=True, number_of_processes=num_procs, shots=shots)
        new_fids = [slater_determinant_probability(
            phi0, prob_dist) for prob_dist in prob_dists]
        statevectors = qiskit_statevector(
            circuits, make_copy=False, number_of_processes=num_procs)
        new_fids_statevector = [slater_determinant_probability_from_statevector(
            phi0, statevector) for statevector in statevectors]
        return new_fids, new_fids_statevector

    def calculate_new_residual_adjoint(circuits, processing_instructions, cutoff, shots, backend, pm, mit):
        if print_time:
            start_time = time.time()

        fids, fids_statevector = process_circuits(
            circuits, shots, backend, pm, mit)

        if print_time:
            print(
                f"\tCalculated {len(fids)} fidelities in {time.time() - start_time:.2f} seconds")
            start_time = time.time()

        new_residual_adjoint_prods = []
        result_idx = 0
        shot_error = 0.0

        for instruction in processing_instructions:
            if instruction[0] == "calculated":
                label = instruction[1]
                fProd = instruction[2]

                dF = 0.0 + 0.0j
                dF_statevector = 0.0 + 0.0j

                if label == 1 or label == 3:
                    fid1 = fids[result_idx]
                    fid1r = fids[result_idx + 1]
                    fid2 = fids[result_idx + 2]
                    fid2r = fids[result_idx + 3]

                    fid1_statevector = fids_statevector[result_idx]
                    fid1r_statevector = fids_statevector[result_idx + 1]
                    fid2_statevector = fids_statevector[result_idx + 2]
                    fid2r_statevector = fids_statevector[result_idx + 3]

                    result_idx += 4

                    dF_real = d1 * (fid1 - fid1r) - \
                        d2 * (fid2 - fid2r)
                    dF_real_statevector = d1 * (fid1_statevector - fid1r_statevector) - \
                        d2 * (fid2_statevector - fid2r_statevector)
                    if np.abs(dF_real) > cutoff:
                        dF += 0.5 * dF_real
                    dF_statevector += 0.5 * dF_real_statevector

                if label == 2 or label == 3:
                    fid3 = fids[result_idx]
                    fid3r = fids[result_idx + 1]
                    fid4 = fids[result_idx + 2]
                    fid4r = fids[result_idx + 3]

                    fid3_statevector = fids_statevector[result_idx]
                    fid3r_statevector = fids_statevector[result_idx + 1]
                    fid4_statevector = fids_statevector[result_idx + 2]
                    fid4r_statevector = fids_statevector[result_idx + 3]

                    result_idx += 4

                    dF_imag = d1 * (fid3 - fid3r) - \
                        d2 * (fid4 - fid4r)
                    dF_imag_statevector = d1 * (fid3_statevector - fid3r_statevector) - \
                        d2 * (fid4_statevector - fid4r_statevector)
                    if np.abs(dF_imag) > cutoff:
                        dF += 0.5 * 1j * dF_imag
                    dF_statevector += 0.5 * 1j * dF_imag_statevector

                if label == -1:
                    print(f"Error: label is -1 for {fProd.ops_to_string()}")

                if fProd.ops_to_string() != fProd.adjoint().ops_to_string():
                    shot_error += np.abs(dF_statevector - dF)**2
                    dF_adjoint = -1 * dF.conjugate()
                    dF_adjoint_statevector = -1 * dF_statevector.conjugate()
                    shot_error += np.abs(dF_adjoint_statevector -
                                         dF_adjoint)**2

                    print(f"\t{fProd} -> {dF}~{dF_statevector}")
                    print(
                        f"\t{fProd.adjoint()} -> {dF_adjoint}~{dF_adjoint_statevector} (from adjoint)")
                    if np.abs(dF) > 1e-10:
                        new_residual_adjoint_prods.append(dF * fProd)
                        new_residual_adjoint_prods.append(
                            dF_adjoint * fProd.adjoint())
                else:
                    shot_error += np.abs(dF_statevector - dF)**2
                    print(f"\t{fProd} -> {dF}~{dF_statevector} (self-adjoint)")
                    if np.abs(dF) > 1e-10:
                        new_residual_adjoint_prods.append(dF * fProd)

        if print_time:
            print(
                f"\tProcessed {len(processing_instructions)} products in {time.time() - start_time:.2f} seconds")

        new_residual_adjoint = Operator(
            new_residual_adjoint_prods, trdm.N, "A")

        return new_residual_adjoint, shot_error

    def update_new_residual_adjoints_circuit(new_residual_adjoint, residual_adjoints_circuit, optimization_level, backend):
        new_residual_adjoint_circuit = qiskit_create_pauli_sum_evolution_circuit(
            new_residual_adjoint.pSum, EPSILON)
        residual_adjoints_circuit = new_residual_adjoint_circuit.compose(
            residual_adjoints_circuit, inplace=False)
        residual_adjoints_circuit = q.transpile(
            residual_adjoints_circuit, basis_gates=GATES, optimization_level=optimization_level)
        return residual_adjoints_circuit

    residual_adjoints_circuit = q.QuantumCircuit(phi0.N, 0)
    residual_adjoints = []
    shot_error = 0.0

    fProds_seen = set()
    processing_instructions = []
    circuits = []
    for i, fProd in enumerate(trdm.prods):
        fProd_adjoint = fProd.adjoint()
        if fProd_adjoint.ops_to_string() not in fProds_seen:

            label = 0

            a_pSum = (1 / fProd.coef) * \
                (fProd.pSum + -1 * fProd.pSum.adjoint())
            if a_pSum.p != 0:
                circ1 = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, a_pSum, psi_circuit, theta1)
                circ1r = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, a_pSum, psi_circuit, -theta1)
                circ2 = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, a_pSum, psi_circuit, theta2)
                circ2r = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, a_pSum, psi_circuit, -theta2)
                circuits.extend([circ1, circ1r, circ2, circ2r])
                label += 1

            h_pSum = (1 / fProd.coef) * (fProd.pSum + fProd.pSum.adjoint())
            if h_pSum.p != 0:
                circ3 = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, h_pSum, psi_circuit, 1j * theta1)
                circ3r = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, h_pSum, psi_circuit, -1j * theta1)
                circ4 = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, h_pSum, psi_circuit, 1j * theta2)
                circ4r = create_circuit(
                    phi_circuit_adjoint, residual_adjoints_circuit, h_pSum, psi_circuit, -1j * theta2)
                circuits.extend([circ3, circ3r, circ4, circ4r])
                label += 2

            processing_instructions.append(("calculated", label, fProd))
            fProds_seen.add(fProd.ops_to_string())

        else:
            processing_instructions.append(("adjoint calculated", -1))
            fProds_seen.add(fProd.adjoint().ops_to_string())

        if len(circuits) >= MAX_CIRCUITS - 8:
            print(f"\n\tOn prod {i} of {len(trdm.prods)}")
            new_residual_adjoint, new_shot_error = calculate_new_residual_adjoint(
                circuits, processing_instructions, cutoff, shots, backend, pm, mit)
            if new_residual_adjoint.pSum.p != 0:
                residual_adjoints.append(new_residual_adjoint)
                residual_adjoints_circuit = update_new_residual_adjoints_circuit(
                    new_residual_adjoint, residual_adjoints_circuit, optimization_level, backend)
            else:
                print(f"\tResidual adjoint is zero, skipping")

            shot_error += new_shot_error

            processing_instructions = []
            circuits = []

    if circuits:
        print(f"\n\tOn prod {i} of {len(trdm.prods)}")
        new_residual_adjoint, new_shot_error = calculate_new_residual_adjoint(
            circuits, processing_instructions, cutoff, shots, backend, pm, mit)
        if new_residual_adjoint.pSum.p != 0:
            residual_adjoints.append(new_residual_adjoint)
            residual_adjoints_circuit = update_new_residual_adjoints_circuit(
                new_residual_adjoint, residual_adjoints_circuit, optimization_level, backend)
        else:
            print(
                f"\tResidual adjoint is zero, skipping")

        shot_error += new_shot_error

    return residual_adjoints, residual_adjoints_circuit, shot_error


def beta_posterior_fidelity(successes, shots, prior_alpha, prior_beta):
    eta = 0.8
    posterior_alpha = (1 - eta) * prior_alpha + eta * successes
    posterior_beta = (1 - eta) * prior_beta + eta * (shots - successes)

    mean = posterior_alpha / (posterior_alpha + posterior_beta)
    lower, upper = stats.beta.interval(0.95, posterior_alpha, posterior_beta)

    return mean, lower, upper


def calculate_initial_fidelity(phi0: SlaterDeterminant, psi_circuit: q.QuantumCircuit, shots: int, backend, pm, mit):

    prob_dist = qiskit_probability_distribution(
        psi_circuit, backend, pm, mit=mit, make_copy=True, shots=shots)
    success_prob = slater_determinant_probability(phi0, prob_dist)
    successes = int(round(success_prob * shots))

    fid, fid_lower, fid_upper = beta_posterior_fidelity(
        successes, shots, 1, 1)

    statevector = qiskit_statevector(psi_circuit, make_copy=True)
    fid_statevector = slater_determinant_probability_from_statevector(
        phi0, statevector)

    return fid, fid_lower, fid_upper, np.abs(fid_statevector)


def calculate_new_fidelity(previous_fidelity: np.float64, phi0: SlaterDeterminant, phi_circuit_adjoint: q.QuantumCircuit, residual_adjoints_circuit: Operator, psi_circuit: q.QuantumCircuit, shots: int, backend, pm, mit):

    circ = psi_circuit.copy()
    circ.compose(residual_adjoints_circuit, inplace=True)
    circ.compose(phi_circuit_adjoint, inplace=True)

    prob_dist = qiskit_probability_distribution(
        circ, backend, pm, mit=mit, make_copy=True, shots=shots)
    success_prob = slater_determinant_probability(phi0, prob_dist)
    successes = int(round(success_prob * shots))

    fid, fid_lower, fid_upper = beta_posterior_fidelity(
        successes, shots, shots * previous_fidelity, shots * (1 - previous_fidelity))

    statevector = qiskit_statevector(circ, make_copy=False)
    fid_statevector = slater_determinant_probability_from_statevector(
        phi0, statevector)

    return fid, fid_lower, fid_upper, np.abs(fid_statevector)


def optimize_phi(phi: SlaterDeterminant, psi_circuit: q.QuantumCircuit, parameters: np.array, trdm: RDM, max_iters: int, convergence: float, optimization_level: int, shots: int, backend, pm, mit):
    print(f"\nOptimizing phi")

    phi_init_adjoint = InitOperators(phi.N)

    phi_circuit_adjoint = q.QuantumCircuit(phi.N, 0)

    fids = []
    fids_statevector = []

    fid0, fid0_lower, fid0_upper, fid0_statevector = calculate_initial_fidelity(
        phi, psi_circuit, shots, backend, pm, mit)
    fids.append(fid0)
    fids_statevector.append(fid0_statevector)
    print(f"Initial fidelity: {fid0}~{fid0_statevector}")

    start_time = time.time()

    for iter in range(max_iters):
        print(f"\nIteration: {iter}")
        residual_adjoints, residual_adjoints_circuit, shot_error = calculate_residual_adjoints(
            phi, phi_circuit_adjoint, psi_circuit, parameters, trdm, optimization_level, shots, backend, pm, mit)
        print(f"\tShot error: {shot_error}")

        if len(residual_adjoints) == 0:
            print(
                f"\tResidual adjoint is near-zero, skipping. Fidelity remains {fids[-1]}~{fids_statevector[-1]}")
            continue

        fid, fid_lower, fid_upper, fid_statevector = calculate_new_fidelity(
            fids[-1], phi, phi_circuit_adjoint, residual_adjoints_circuit, psi_circuit, shots, backend, pm, mit)

        if fid > fids[-1]:
            print(
                f"Fidelity improved to {fid}~{fid_statevector} with epsilon = {EPSILON}")
            fids.append(fid)
            fids_statevector.append(fid_statevector)
            elapsed = time.time() - start_time
            print(f"Time elapsed: {elapsed:.2f} seconds")

            for residual_adjoint in residual_adjoints:
                phi_init_adjoint.add_operator_at_beginning(
                    residual_adjoint, EPSILON)

            phi_circuit_adjoint = residual_adjoints_circuit.compose(
                phi_circuit_adjoint, inplace=False)

            phi_circuit_adjoint = q.transpile(
                phi_circuit_adjoint, basis_gates=GATES, optimization_level=optimization_level)
        else:
            print(
                f"Fidelity did not improve {fid}~{fid_statevector}<{fids[-1]}, residual rejected")
            elapsed = time.time() - start_time
            print(f"Time elapsed: {elapsed:.2f} seconds")

        if 1 - fid < convergence:
            print("Convergence reached")
            break

    return phi_init_adjoint, phi_circuit_adjoint


def evaluate_ordm_statevector(psi_circuit: q.QuantumCircuit, ordm: RDM):

    ordm_tomography = qiskit_perform_tomography_statevector(
        psi_circuit, ordm.aggregate_measurements_recursive())

    new_prods = []
    for prod in ordm.prods:
        d = prod.evaluate_expectation(ordm_tomography)
        new_prod = d * prod
        new_prods.append(new_prod)

    return RDM(1, ordm.N, new_prods)


def evaluate_ordm(psi_circuit: q.QuantumCircuit, ordm: RDM, shots: int, backend, pm):

    ordm_tomography = qiskit_perform_tomography(
        psi_circuit, ordm.aggregate_measurements_recursive(), backend, pm, shots)

    new_prods = []
    for prod in ordm.prods:
        d = prod.evaluate_expectation(ordm_tomography)
        new_prod = d * prod
        new_prods.append(new_prod)

    return RDM(1, ordm.N, new_prods)


def comparison(psi_vector, reference_psi_vector):
    fid = 0
    for i in range(len(psi_vector)):
        fid += psi_vector[i] * reference_psi_vector[i].conjugate()
    return np.abs(fid)**2


def calculate_energy_statevector(psi_circuit: q.QuantumCircuit, H: Hamiltonian):
    tomography = qiskit_perform_tomography_statevector(
        psi_circuit, H.aggregate_measurements())
    energy = H.energy(tomography)
    return energy


def calculate_energy(psi_circuit: q.QuantumCircuit, H: Hamiltonian, shots: int, backend, pm):
    tomography = qiskit_perform_tomography(
        psi_circuit, H.aggregate_measurements(), backend, pm, shots)
    energy = H.energy(tomography)
    return energy


def trotter_error(phi, phi_circuit_adjoint, phi_circuit):
    circ = phi_circuit.copy()
    circ.compose(phi_circuit_adjoint, inplace=True)
    statevector = qiskit_statevector(circ)
    fid_statevector = slater_determinant_probability_from_statevector(
        phi, statevector)
    return fid_statevector


def CETE(phi0: SlaterDeterminant, psi0: SlaterDeterminant, psi_init_circuit: q.QuantumCircuit, H: Hamiltonian, tau: float, tau_iters: int, time_steps: int, max_iters: int, convergence: float, parameters: np.array, trdm: RDM, ordm: RDM, ordm_inverse_mapping: dict, optimization_level: int, shots: int, shots_residual: int, backend, pm, mit):

    label = f"{tau}_{tau_iters}_{time_steps}"
    output_dir = f"results/H2_{label}"

    output_file = f"{output_dir}/summary.txt"
    old_stdout = sys.stdout
    sys.stdout = open(output_file, "w", buffering=1)
    print(f"Ran at {datetime.now()}\n")
    print(f"shots: {shots}")
    print(f"shots residual: {shots_residual}")
    print(f"backend: {backend}")
    print(f"max_iters: {max_iters}")
    print(f"convergence: {convergence}")
    print(f"derivative parameters: {parameters}")
    print(
        f"psi0: {format_statevector(create_statevector(psi0, psi_init_circuit), psi0.N)}")
    print(f"Transpiler optimization level: {optimization_level}\n")
    print(f"Single reference state: {phi0}")

    print("")
    print('-' * 40)
    print("Time Evolution Started with CETE")
    print('-' * 40)
    print(f"\t{MAX_CIRCUITS} Max Circuits")
    print(f"\t{FALSE_POSITIVE_REJECTION_RATE} False Positive Rejection Rate")
    print(f"\t{NUMBER_PROCESSES} Number of Processes")
    print(f"\t{EPSILON} Epsilon")
    print(f"\t{GATES} Basis Gates")
    print("")

    CETE_fids = []

    CETE_ordms = []
    sequential_ordms = []

    CETE_energies = []
    sequential_energies = []

    CETE_depths = []
    sequential_depths = []

    simulation_times = []
    simulation_time = 0

    CETE_psi_circuit = q.QuantumCircuit(psi0.N, 0)
    CETE_psi_circuit.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(psi0), inplace=True)
    CETE_psi_circuit.compose(psi_init_circuit, inplace=True)
    sequential_psi_circuit = CETE_psi_circuit.copy()

    CETE_psi_depth = CETE_psi_circuit.depth()
    sequential_psi_depth = sequential_psi_circuit.depth()
    CETE_depths.append(CETE_psi_depth)
    sequential_depths.append(sequential_psi_depth)

    new_ordm = evaluate_ordm(CETE_psi_circuit, ordm, shots, backend, pm)
    if ordm_inverse_mapping is None:
        CETE_ordms.append(new_ordm.save())
        sequential_ordms.append(new_ordm.save())
    else:
        new_ordm_unmapped = new_ordm.unmap(ordm_inverse_mapping)
        CETE_ordms.append(new_ordm_unmapped.save())
        sequential_ordms.append(new_ordm_unmapped.save())

    energy = calculate_energy(CETE_psi_circuit, H, shots, backend, pm)
    CETE_energies.append(energy)
    sequential_energies.append(energy)
    simulation_times.append(simulation_time)

    evolution_circuit = qiskit_create_pauli_sum_evolution_circuit(
        H.pSum, -1j * tau)
    evolution_circuit = q.transpile(
        evolution_circuit, basis_gates=GATES, optimization_level=optimization_level)

    for time_step in range(time_steps):
        print(f"Time step: {time_step}")

        for tau_iter in range(tau_iters):
            CETE_psi_circuit.compose(evolution_circuit, inplace=True)
            sequential_psi_circuit.compose(evolution_circuit, inplace=True)

        simulation_time += tau * tau_iters
        simulation_times.append(simulation_time)

        phi_init_adjoint, phi_circuit_adjoint = optimize_phi(
            phi0, CETE_psi_circuit, parameters, trdm, max_iters, convergence, optimization_level, shots_residual, backend, pm, mit)

        if False:  # Avoid using phi_init
            psi_init_circuit = phi_circuit_adjoint.inverse()
        else:
            phi_init = phi_init_adjoint.adjoint()
            psi_init_circuit = phi_init.create_initialization_circuit()

        CETE_psi_circuit = q.QuantumCircuit(psi0.N, 0)
        CETE_psi_circuit.compose(
            qiskit_create_initialization_from_slater_determinant_circuit(psi0), inplace=True)
        CETE_psi_circuit.compose(psi_init_circuit, inplace=True)

        CETE_psi_circuit = q.transpile(
            CETE_psi_circuit, basis_gates=GATES, optimization_level=optimization_level)
        CETE_psi_depth = CETE_psi_circuit.depth()

        sequential_psi_circuit = q.transpile(
            sequential_psi_circuit, basis_gates=GATES, optimization_level=optimization_level)
        sequential_psi_depth = sequential_psi_circuit.depth()

        CETE_depths.append(CETE_psi_depth)
        sequential_depths.append(sequential_psi_depth)

        CETE_psi_vec = qiskit_statevector(CETE_psi_circuit)

        start_time = time.time()
        new_CETE_ordm = evaluate_ordm(
            CETE_psi_circuit, ordm, shots, backend, pm)
        if ordm_inverse_mapping is None:
            CETE_ordms.append(new_CETE_ordm.save())
        else:
            new_CETE_ordm_unmapped = new_CETE_ordm.unmap(ordm_inverse_mapping)
            CETE_ordms.append(new_CETE_ordm_unmapped.save())

        elapsed = time.time() - start_time
        print(f"Evaluated CETE ORDM: {elapsed:.2f} total seconds elapsed")

        sequential_psi_vec = qiskit_statevector(sequential_psi_circuit)

        start_time = time.time()
        new_sequential_ordm = evaluate_ordm(
            sequential_psi_circuit, ordm, shots, backend, pm)
        if ordm_inverse_mapping is None:
            sequential_ordms.append(new_sequential_ordm.save())
        else:
            new_sequential_ordm_unmapped = new_sequential_ordm.unmap(
                ordm_inverse_mapping)
            sequential_ordms.append(new_sequential_ordm_unmapped.save())
        elapsed = time.time() - start_time
        print(
            f"Evaluated Sequential ORDM: {elapsed:.2f} total seconds elapsed")

        fid = comparison(CETE_psi_vec, sequential_psi_vec)
        CETE_fids.append(fid)

        CETE_E = calculate_energy(CETE_psi_circuit, H, shots, backend, pm)
        CETE_energies.append(CETE_E)
        sequential_E = calculate_energy(
            sequential_psi_circuit, H, shots, backend, pm)
        sequential_energies.append(sequential_E)

        if False:
            trot_err = trotter_error(phi0, phi_circuit_adjoint, psi_circuit)
            print(f"Optimization trotter error: {trot_err}")

        print(f"\nTotal Evolution Time: {tau * tau_iters * (time_step+1)}")
        print(
            f"Fidelity of CETE with Reference: {fid} = 1 - 10^{np.log10(1 - fid)}")
        print(f"Depth of CETE circuit: {CETE_psi_depth}")
        print(f"New CETE Electronic State: {CETE_psi_vec}")
        print(f"CETE Energy: {CETE_energies[-1]}")

        print(f"Depth of Sequential circuit: {sequential_psi_depth}")
        print(f"New Sequential Electronic State: {sequential_psi_vec}")
        print(f"Sequential Energy: {sequential_energies[-1]}")

        print("\n\n\n")

    print("Time Evolution Complete")
    sys.stdout = old_stdout

    return CETE_ordms, CETE_energies, CETE_depths, CETE_fids, simulation_times, sequential_ordms, sequential_energies, sequential_depths
