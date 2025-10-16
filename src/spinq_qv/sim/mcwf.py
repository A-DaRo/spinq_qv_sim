"""
Monte Carlo Wavefunction (MCWF) simulator.

Implements quantum trajectory simulation by running multiple stochastic
statevector evolutions and averaging results. Approximates density-matrix
evolution with controllable accuracy via number of trajectories.

Memory requirement: Same as statevector (2^n), but repeated n_trajectories times.
"""

from typing import Dict, Optional, Any, List
import numpy as np
from copy import deepcopy

from spinq_qv.sim.backend import SimulatorBackend
from spinq_qv.sim.statevector import StatevectorBackend


class MCWFBackend(SimulatorBackend):
    """
    Monte Carlo Wavefunction simulator using quantum trajectories.
    
    Approximates density matrix evolution by averaging over many stochastic
    statevector runs. Each trajectory applies Kraus operators stochastically
    via quantum jump formalism.
    
    Convergence: Statistical error scales as 1/sqrt(n_trajectories).
    Typical usage: 100-1000 trajectories for reasonable accuracy.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_trajectories: int = 100,
        seed: Optional[int] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize MCWF simulator.
        
        Args:
            n_qubits: Number of qubits
            n_trajectories: Number of quantum trajectories to average over
            seed: Random seed for reproducibility
            use_gpu: Ignored (CPU-only implementation)
        """
        super().__init__(n_qubits, seed)
        
        self.n_trajectories = n_trajectories
        self.use_gpu = False
        
        # Store accumulated measurement probabilities from all trajectories
        self._accumulated_probs: Optional[np.ndarray] = None
        self._n_accumulated = 0
        
        # Current trajectory index
        self._current_trajectory = 0
        
        # Reset to initialize first trajectory
        self.reset()
    
    def reset(self) -> None:
        """Reset MCWF to prepare for new circuit execution."""
        self._accumulated_probs = np.zeros(2 ** self.n_qubits, dtype=np.float64)
        self._n_accumulated = 0
        self._current_trajectory = 0
    
    def run_trajectories(
        self, 
        circuit_func,
        n_trajectories: Optional[int] = None
    ) -> None:
        """
        Run multiple quantum trajectories and accumulate results.
        
        Args:
            circuit_func: Callable that applies gates/noise to a backend.
                         Should accept a StatevectorBackend instance.
            n_trajectories: Number of trajectories (default: self.n_trajectories)
        """
        if n_trajectories is None:
            n_trajectories = self.n_trajectories
        
        self.reset()
        
        for traj in range(n_trajectories):
            # Create fresh statevector backend for this trajectory
            traj_seed = self.rng.integers(0, 2**31) if self.seed is not None else None
            backend = StatevectorBackend(self.n_qubits, seed=traj_seed)
            
            # Apply circuit with stochastic noise
            circuit_func(backend)
            
            # Accumulate probabilities
            probs = backend.get_probabilities()
            self._accumulated_probs += probs
            self._n_accumulated += 1
        
        self._current_trajectory = n_trajectories
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get averaged measurement probabilities from all trajectories.
        
        Returns:
            Average probabilities across trajectories
        """
        if self._n_accumulated == 0:
            # No trajectories run yet, return uniform distribution
            return np.ones(2 ** self.n_qubits) / (2 ** self.n_qubits)
        
        return self._accumulated_probs / self._n_accumulated
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Approximate density matrix from trajectory average.
        
        WARNING: This is a diagonal approximation (only population terms).
        True density matrix reconstruction requires more sophisticated averaging.
        
        Returns:
            Diagonal density matrix with averaged populations
        """
        probs = self.get_probabilities()
        dim = 2 ** self.n_qubits
        rho = np.zeros((dim, dim), dtype=np.complex128)
        np.fill_diagonal(rho, probs)
        return rho
    
    def measure(
        self, 
        shots: int, 
        readout_noise: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Sample measurement outcomes from averaged probabilities.
        
        Args:
            shots: Number of measurement samples
            readout_noise: Optional readout error model (not implemented)
        
        Returns:
            Dictionary mapping bitstrings to counts
        """
        if readout_noise is not None:
            raise NotImplementedError("Readout noise not yet implemented")
        
        # Get averaged probabilities
        probs = self.get_probabilities()
        
        # Ensure valid probabilities
        probs = np.maximum(probs, 0.0)
        probs = probs / np.sum(probs)
        
        # Sample outcomes
        outcomes = self.rng.multinomial(shots, probs)
        
        # Convert to bitstring dictionary
        counts = {}
        for idx in range(2 ** self.n_qubits):
            if outcomes[idx] > 0:
                bitstring = format(idx, f'0{self.n_qubits}b')
                counts[bitstring] = int(outcomes[idx])
        
        return counts
    
    # Stub methods for compatibility with SimulatorBackend interface
    # These are not directly used in MCWF workflow
    
    def init_state(self, state: Optional[np.ndarray] = None) -> None:
        """
        Not used in MCWF (each trajectory starts fresh).
        Included for interface compatibility.
        """
        pass
    
    def apply_unitary(self, unitary: np.ndarray, targets: list[int]) -> None:
        """
        Not used directly (applied within trajectories).
        Included for interface compatibility.
        """
        raise NotImplementedError(
            "MCWF does not support direct gate application. "
            "Use run_trajectories() with a circuit function instead."
        )
    
    def get_statevector(self) -> np.ndarray:
        """
        Cannot return single statevector (MCWF is ensemble average).
        """
        raise NotImplementedError(
            "MCWF represents mixed state; cannot return single statevector. "
            "Use get_probabilities() or get_density_matrix() instead."
        )
    
    def apply_circuit(self, circuit) -> None:
        """
        Apply circuit via trajectory averaging.
        
        This runs n_trajectories independent statevector simulations,
        each with stochastic noise, and averages the results.
        
        Args:
            circuit: CircuitSpec object with gates to apply
        """
        from spinq_qv.io.formats import CircuitSpec
        
        if not isinstance(circuit, CircuitSpec):
            raise TypeError(f"Expected CircuitSpec, got {type(circuit)}")
        
        if circuit.width != self.n_qubits:
            raise ValueError(
                f"Circuit width {circuit.width} does not match "
                f"backend qubits {self.n_qubits}"
            )
        
        # Define circuit application function for trajectories
        def apply_circuit_to_backend(backend: StatevectorBackend):
            """Apply circuit gates to a statevector backend."""
            backend.apply_circuit(circuit)
        
        # Run trajectories
        self.run_trajectories(apply_circuit_to_backend)
