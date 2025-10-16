"""
Abstract base class for quantum simulation backends.

Defines the interface that all simulators (statevector, density matrix, etc.)
must implement for use with the QV experiment runner.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np


class SimulatorBackend(ABC):
    """
    Abstract base class for quantum simulators.
    
    All backends must implement methods for:
    - State initialization
    - Unitary gate application
    - Measurement sampling
    """
    
    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        """
        Initialize the simulator backend.
        
        Args:
            n_qubits: Number of qubits in the simulation
            seed: Random seed for measurement sampling (optional)
        """
        self.n_qubits = n_qubits
        self.seed = seed
        
        # Initialize RNG for measurements
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
    
    @abstractmethod
    def init_state(self, state: Optional[np.ndarray] = None) -> None:
        """
        Initialize or reset the quantum state.
        
        Args:
            state: Optional initial state. If None, initialize to |0...0⟩.
                  For statevector: shape (2^n,)
                  For density matrix: shape (2^n, 2^n)
        """
        pass
    
    @abstractmethod
    def apply_unitary(self, unitary: np.ndarray, targets: list[int]) -> None:
        """
        Apply a unitary gate to specified target qubits.
        
        Args:
            unitary: Unitary matrix (2^k × 2^k for k target qubits)
            targets: List of qubit indices to apply gate to
        """
        pass
    
    @abstractmethod
    def get_statevector(self) -> np.ndarray:
        """
        Get the current statevector representation.
        
        Returns:
            Complex amplitude array of shape (2^n,)
            
        Note:
            For density matrix backends, this may compute the statevector
            via eigendecomposition if the state is pure, or raise an error
            if the state is mixed.
        """
        pass
    
    @abstractmethod
    def get_probabilities(self) -> np.ndarray:
        """
        Get measurement probabilities for all computational basis states.
        
        Returns:
            Probability array of shape (2^n,) with real values in [0, 1]
            that sum to 1.0
        """
        pass
    
    @abstractmethod
    def measure(
        self, 
        shots: int, 
        readout_noise: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Perform measurements in the computational basis.
        
        Args:
            shots: Number of measurement samples to take
            readout_noise: Optional dictionary specifying readout error model
                          e.g., {'p0given1': 0.01, 'p1given0': 0.01}
        
        Returns:
            Dictionary mapping bitstrings (e.g., "0101") to counts
        """
        pass
    
    def reset_rng(self, seed: int) -> None:
        """Reset the random number generator with a new seed."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits}, seed={self.seed})"
