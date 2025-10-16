"""
spinq_qv: Simulated Quantum Volume benchmarks for Si/SiGe spin qubits.

A noise-model-first simulator for estimating Quantum Volume under physically
realistic noise models derived from experimental parameters (T1, T2, gate
fidelities, crosstalk, SPAM errors).
"""

__version__ = "0.1.0"
__author__ = "Alessandro Da Ros"

# Package-level imports for convenience
from . import config
from . import utils

__all__ = ["config", "utils", "__version__"]
