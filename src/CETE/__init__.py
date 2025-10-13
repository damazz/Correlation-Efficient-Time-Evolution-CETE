from importlib.metadata import PackageNotFoundError, version as _pkg_version

from CETE import CETE
from CETE.CETE_helper_H2 import *

__all__ = [
    "format_statevector",
    "create_statevector",
    "calculate_cutoff",
    "calculate_residual_adjoints",
    "beta_posterior_fidelity",
    "calculate_initial_fidelity",
    "calculate_new_fidelity",
    "optimize_phi",
    "evaluate_ordm_statevector",
    "evaluate_ordm",
    "comparison",
    "calculate_energy_statevector",
    "calculate_energy",
    "trotter_error",
    "CETE",
    "run_H2_1q",
    "generate_reference",
    "plot_ordms"]
