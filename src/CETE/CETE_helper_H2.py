import qiskit as q
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from sparse_sim.fermion.rdm import *
from sparse_sim.fermion.qiskit_wrapper import *

from CETE.CETE import *


def run_H2_1q(psi0, psi_circuit, H2_4q, tau, tau_iters, time_steps, max_iters, convergence, parameters, optimization_level, shots, shots_residual, backend, pm, mit):

    # Hamiltonian is then projected onto 1 qubit subspace
    sdet1010 = SlaterDeterminant(4, 1, [1, 0, 1, 0])
    sdet0101 = SlaterDeterminant(4, 1, [0, 1, 0, 1])
    sdet0 = SlaterDeterminant(1, 1, [0])
    sdet1 = SlaterDeterminant(1, 1, [1])

    four_to_one = Projector(4, 1)
    four_to_one = four_to_one + (sdet1010, sdet0) + (sdet0101, sdet1)

    H2_1q, H_one_to_four = H2_4q.map(four_to_one)

    # Single slater determinant reference state is selected
    phi0 = sdet0

    trdm_4q = RDM(2, sdet1010.N)

    trdm_1q, trdm_one_to_four = trdm_4q.map(
        four_to_one, ignore_duplicates=False)

    ordm_4q = RDM(1, sdet1010.N)

    ordm_1q, ordm_one_to_four = ordm_4q.map(
        four_to_one, ignore_duplicates=True)

    # CETE algorithm is run
    result = CETE(phi0, psi0, psi_circuit, H2_1q, tau, tau_iters, time_steps,
                  max_iters, convergence, parameters, trdm_1q, ordm_1q, ordm_one_to_four, optimization_level, shots, shots_residual, backend, pm, mit)

    return result


def generate_reference(psi0, psi0_circuit, H_4q, tau, tau_iters, time_steps):

    ordm_4q = RDM(1, psi0.N)

    reference_ordms = []
    reference_energies = []
    reference_time = 0.0
    reference_times = []

    reference_circ = q.QuantumCircuit(psi0.N, 0)
    reference_circ.compose(
        qiskit_create_initialization_from_slater_determinant_circuit(psi0), inplace=True)
    reference_circ.compose(psi0_circuit, inplace=True)

    reference_times.append(reference_time)

    evolution_circuit = qiskit_create_pauli_sum_evolution_circuit(
        H_4q.pSum, -1j * tau)

    reference_energy = calculate_energy_statevector(
        reference_circ, H_4q)
    reference_energies.append(reference_energy)

    reference_ordm = evaluate_ordm_statevector(
        reference_circ, ordm_4q)
    reference_ordms.append(reference_ordm.save())

    for time_step in range(time_steps):
        print(f"On time step {time_step+1} of {time_steps}")
        for tau_iter in range(tau_iters):
            reference_circ.compose(evolution_circuit, inplace=True)

            reference_energy = calculate_energy_statevector(
                reference_circ, H_4q)
            reference_energies.append(reference_energy)

            reference_ordm = evaluate_ordm_statevector(
                reference_circ, ordm_4q)
            reference_ordms.append(reference_ordm.save())

            reference_time += tau
            reference_times.append(reference_time)

    return reference_ordms, reference_energies, reference_times


def plot_ordms(CETE_ordms, sequential_ordms, times, reference_ordms, reference_times):
    N = 4
    keys = [f'+a_{i}-a_{i}' for i in range(N)]

    CETE_diagonals = []
    for CETE_ordm in CETE_ordms:
        CETE_ordm = load_operator(CETE_ordm)
        diagonal_elements = CETE_ordm.diagonal_elements()
        diagonal = [diagonal_elements[key].coef for key in keys]
        CETE_diagonals.append(diagonal)
    CETE_diagonals = np.array(CETE_diagonals)

    sequential_diagonals = []
    for sequential_ordm in sequential_ordms:
        sequential_ordm = load_operator(sequential_ordm)
        sequential_diagonal_elements = sequential_ordm.diagonal_elements()
        sequential_diagonal = [
            sequential_diagonal_elements[key].coef for key in keys]
        sequential_diagonals.append(sequential_diagonal)
    sequential_diagonals = np.array(sequential_diagonals)

    reference_diagonals = []
    for reference_ordm in reference_ordms:
        reference_ordm = load_operator(reference_ordm)
        reference_diagonal_elements = reference_ordm.diagonal_elements()
        reference_diagonal = [
            reference_diagonal_elements[key].coef for key in keys]
        reference_diagonals.append(reference_diagonal)
    reference_diagonals = np.array(reference_diagonals)

    fig, ax = plt.subplots(figsize=(10, 8))

    markers = ['o', 'x']
    colors = ["#643A48", "#AB6B35", "#EAB76A", "#A14743"]
    base_labels = [rf"$\langle a^\dagger_{i} a_{i} \rangle$" for i in range(N)]

    cete_handles = []
    seq_handles = []

    # Plot reference results
    for i, key in enumerate(keys):
        ax.plot(reference_times, reference_diagonals[:, i].real,
                linestyle='-', color="black", label='_nolegend_')

    for i, key in enumerate(keys):
        h_cete, = ax.plot(
            times, CETE_diagonals[:, i].real,
            marker=markers[0], linestyle='', markersize=10,
            color=colors[i], label=base_labels[i], fillstyle='none', markeredgewidth=3, alpha=0.9
        )
        cete_handles.append(h_cete)

    for i, key in enumerate(keys):
        h_seq, = ax.plot(
            times, sequential_diagonals[:, i].real,
            marker=markers[1], linestyle='', markersize=10, fillstyle='none', markeredgewidth=3,
            color=colors[i], label=base_labels[i], alpha=0.9
        )
        seq_handles.append(h_seq)

    ax.set_xlabel(r"Time (Ha$^{-1}$)", fontsize=20)
    ax.set_ylabel(
        r"Expectation Value $\langle a^\dagger_i a_i \rangle$", fontsize=20)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 18, 6))
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.tick_params(axis='both', labelsize=16)

    fig.subplots_adjust(bottom=0.30)

    dummy_cete = Line2D([], [], linestyle="None", marker=None, label="CETE:")
    dummy_seq = Line2D([], [], linestyle="None", marker=None,
                       label="Sequential Evolution:")
    cete_handles_with_label = [dummy_cete] + cete_handles
    seq_handles_with_label = [dummy_seq] + seq_handles

    ctqe_labels = ["CETE:"] + base_labels
    seq_labels = ["Sequential Evolution:"] + base_labels

    leg1 = fig.legend(
        cete_handles_with_label, ctqe_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.10),
        ncol=len(ctqe_labels), frameon=False,
        handlelength=0, handletextpad=0.5, fontsize=18
    )

    leg2 = fig.legend(
        seq_handles_with_label, seq_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.03),
        ncol=len(seq_labels), frameon=False,
        handlelength=0, handletextpad=0.5, fontsize=18
    )

    plt.show()


def plot_energies(CETE_energies, sequential_energies, times, reference_energies, reference_times):
    fig, ax = plt.subplots(figsize=(10, 8))

    cete_color = "#375E97"   # deep slate blue
    seq_color = "#D4A017"   # golden ochre

    # Reference results
    ax.plot(reference_times, reference_energies,
            '-', color='black', label='_nolegend_')

    ax.plot(
        times, CETE_energies,
        marker='o', linestyle='', markersize=10,
        fillstyle='none', markeredgewidth=3,
        color=cete_color, alpha=0.9, label='_nolegend_'
    )
    ax.plot(
        times, sequential_energies,
        marker='x', linestyle='', markersize=10,
        markeredgewidth=3, color=seq_color, alpha=0.9, label='_nolegend_'
    )

    ax.set_xlabel(r"Time (Ha$^{-1}$)", fontsize=20)
    ax.set_xticks(np.linspace(0, 18, 6))
    ax.set_ylabel("Energy (Ha)", fontsize=20)
    ax.set_yticks(np.linspace(-1.0, -0.8, 5))
    ax.set_ylim(-1.0, -0.8)
    ax.tick_params(axis='both', labelsize=16)

    fig.subplots_adjust(bottom=0.20)

    text_cete = Line2D([], [], linestyle="None", label="CETE:")
    marker_cete = Line2D([], [], linestyle="None", marker='o', markersize=12,
                         fillstyle='none', markeredgewidth=2.5, color=cete_color, label="")
    text_seq = Line2D([], [], linestyle="None", label="Sequential Evolution:")
    marker_seq = Line2D([], [], linestyle="None", marker='x', markersize=12,
                        markeredgewidth=2.5, color=seq_color, label="")

    handles = [text_cete, marker_cete, text_seq, marker_seq]
    labels = ["CETE:", "", "Sequential Evolution:", ""]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03), bbox_transform=fig.transFigure,
        ncol=4, frameon=False,
        fontsize=20,
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2
    )

    plt.show()


def plot_circuit_depths(CETE_depths, sequential_depths, times):
    fig, ax = plt.subplots(figsize=(10, 8))

    cete_color = "#FF8C00"
    seq_color = "#556B2F"

    ax.plot(
        times, CETE_depths,
        marker='o', linestyle='', markersize=10,
        fillstyle='none', markeredgewidth=2.5,
        color=cete_color, alpha=0.9, label='_nolegend_'
    )

    ax.plot(
        times, sequential_depths,
        marker='x', linestyle='', markersize=10,
        markeredgewidth=2.5,
        color=seq_color, alpha=0.9, label='_nolegend_'
    )

    ax.set_xlabel(r"Time (Ha$^{-1}$)", fontsize=20)
    ax.set_xticks(np.linspace(0, 18, 6))
    ax.set_ylabel("Circuit Depth", fontsize=20)
    ax.set_yticks(np.linspace(0, 3500, 5))
    ax.tick_params(axis='both', labelsize=16)

    fig.subplots_adjust(bottom=0.22)

    text_cete = Line2D([], [], linestyle="None", label="CETE:")
    marker_cete = Line2D([], [], linestyle="None", marker='o', markersize=12,
                         fillstyle='none', markeredgewidth=2.5, color=cete_color, label="")
    text_seq = Line2D([], [], linestyle="None", label="Sequential Evolution:")
    marker_seq = Line2D([], [], linestyle="None", marker='x', markersize=12,
                        markeredgewidth=2.5, color=seq_color, label="")

    handles = [text_cete, marker_cete, text_seq, marker_seq]
    labels = ["CETE:", "", "Sequential Evolution:", ""]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03), bbox_transform=fig.transFigure,
        ncol=4, frameon=False, fontsize=18,
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2
    )

    plt.show()
