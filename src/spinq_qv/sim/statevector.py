"""
Pure-state statevector simulator.

Implements efficient statevector simulation with:
- Exact unitary evolution
- Measurement sampling from Born rule probabilities
- CPU implementation using NumPy (GPU via CuPy optional)
"""

from typing import Dict, Optional, Any
import numpy as np

from spinq_qv.sim.backend import SimulatorBackend


# CPU-only implementation: use NumPy for all array operations


class StatevectorBackend(SimulatorBackend):
    """
    Statevector simulator using NumPy (CPU) or CuPy (GPU).
    
    Represents quantum state as a complex amplitude vector |ψ⟩ of shape (2^n,).
    Gates are applied by tensor contraction using einsum.
    
    Memory requirement: 2^n complex128 values = 16 * 2^n bytes
    Example: n=20 qubits → 16 MB, n=24 → 256 MB, n=28 → 4 GB
    """
    
    def __init__(
        self,
        n_qubits: int,
        seed: Optional[int] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize statevector simulator.
        
        Args:
            n_qubits: Number of qubits
            seed: Random seed for measurement sampling
            use_gpu: If True and CuPy available, use GPU acceleration
        """
        super().__init__(n_qubits, seed)

        # Force CPU-only NumPy backend (GPU support removed)
        self.use_gpu = False
        self.xp = np

        # Initialize state to |0...0⟩
        self.state: Optional[np.ndarray] = None
        self.init_state()
    
    def init_state(self, state: Optional[np.ndarray] = None) -> None:
        """
        Initialize quantum state.
        
        Args:
            state: Optional initial statevector. If None, initialize to |0...0⟩
        """
        if state is None:
            # Initialize to computational basis state |0...0⟩
            self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
            self.state[0] = 1.0
        else:
            # Validate and copy provided state
            if state.shape != (2**self.n_qubits,):
                raise ValueError(
                    f"State shape {state.shape} does not match "
                    f"expected (2^{self.n_qubits},) = ({2**self.n_qubits},)"
                )
            
            # Normalize if needed
            norm = np.linalg.norm(state)
            if abs(norm - 1.0) > 1e-10:
                state = state / norm
            self.state = np.array(state, dtype=np.complex128)
    
    def apply_unitary(self, unitary: np.ndarray, targets: list[int]) -> None:
        """
        Apply unitary gate to target qubits using tensor contraction.
        
        Args:
            unitary: Unitary matrix (2^k × 2^k for k target qubits)
            targets: List of k qubit indices (0-indexed)
        
        Implementation:
            Reshape state into tensor of shape (2, 2, ..., 2) with n_qubits axes.
            Contract unitary with target axes using einsum.
            Reshape back to vector.
        """
        k = len(targets)
        expected_dim = 2 ** k

        # Validate unitary dimensions
        if unitary.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Unitary shape {unitary.shape} does not match "
                f"expected ({expected_dim}, {expected_dim}) for {k} qubits"
            )

        # Validate target qubit indices
        if not all(0 <= t < self.n_qubits for t in targets):
            raise ValueError(
                f"Target qubits {targets} out of range [0, {self.n_qubits})"
            )

        # Convert unitary to NumPy array
        unitary = np.asarray(unitary, dtype=np.complex128)

        # Reshape state from (2^n,) to (2, 2, ..., 2) with n_qubits axes
        state_tensor = self.state.reshape([2] * self.n_qubits)

        # Reshape unitary from (2^k, 2^k) to (2, 2, ..., 2, 2, 2, ..., 2)
        # with k output indices and k input indices
        unitary_tensor = unitary.reshape([2] * (2 * k))

        # Build einsum string for contraction
        # Input state indices (lowercase letters)
        in_indices = [chr(ord('a') + i) for i in range(self.n_qubits)]

        # Output unitary indices (map targets to new indices)
        out_indices = in_indices.copy()
        new_idx_start = ord('a') + self.n_qubits
        unitary_out = []
        unitary_in = []

        for i, t in enumerate(targets):
            new_label = chr(new_idx_start + i)
            unitary_out.append(new_label)
            unitary_in.append(in_indices[t])
            out_indices[t] = new_label

        # Build einsum expression
        state_idx = ''.join(in_indices)
        unitary_idx = ''.join(unitary_out) + ''.join(unitary_in)
        result_idx = ''.join(out_indices)

        einsum_expr = f"{state_idx},{unitary_idx}->{result_idx}"

        # Apply gate via tensor contraction
        result_tensor = np.einsum(einsum_expr, state_tensor, unitary_tensor)

        # Reshape back to vector
        self.state = result_tensor.reshape(2 ** self.n_qubits)
    
    def get_statevector(self) -> np.ndarray:
        """Return the current statevector as NumPy array."""
        return self.state.copy()
    
    def get_probabilities(self) -> np.ndarray:
        """Compute Born rule probabilities |⟨x|ψ⟩|²."""
        probs = np.abs(self.state) ** 2
        return probs.copy()
    
    def measure(
        self, 
        shots: int, 
        readout_noise: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Sample measurement outcomes in computational basis.
        
        Args:
            shots: Number of measurement samples
            readout_noise: Optional readout error model (not yet implemented)
        
        Returns:
            Dictionary mapping bitstrings to counts
        """
        if readout_noise is not None:
            raise NotImplementedError(
                "Readout noise not yet implemented in Iteration 3"
            )
        
        # Get probabilities (transfer from GPU if needed)
        probs = self.get_probabilities()
        
        # Sample outcomes according to Born rule
        # Use multinomial distribution for efficiency
        outcomes = self.rng.multinomial(shots, probs)
        
        # Convert to bitstring dictionary (only include non-zero counts)
        counts = {}
        for idx in range(2**self.n_qubits):
            if outcomes[idx] > 0:
                bitstring = format(idx, f'0{self.n_qubits}b')
                counts[bitstring] = int(outcomes[idx])
        
        return counts
    
    def apply_circuit(self, circuit) -> None:
        """
        Apply a full circuit specified as CircuitSpec.
        
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
        
        # Apply each gate in sequence
        for gate in circuit.gates:
            gate_type = gate['type']
            qubits = gate['qubits']
            params = gate.get('params', {})
            
            if gate_type == 'u3':
                # Single-qubit U3 gate: U3(θ, φ, λ)
                theta = params['theta']
                phi = params['phi']
                lam = params['lambda']
                unitary = self._u3_matrix(theta, phi, lam)
                self.apply_unitary(unitary, qubits)
                
            elif gate_type == 'su4':
                # Two-qubit SU(4) gate with full 4×4 unitary
                # Stored as separate real and imaginary parts
                real_part = np.array(params['unitary_real'], dtype=np.float64)
                imag_part = np.array(params['unitary_imag'], dtype=np.float64)
                unitary_flat = real_part + 1j * imag_part
                unitary = unitary_flat.reshape(4, 4)
                self.apply_unitary(unitary, qubits)
                
            else:
                raise NotImplementedError(
                    f"Gate type '{gate_type}' not yet implemented"
                )
    
    def apply_pauli_error(self, p: float, targets: list[int]) -> None:
        """
        Apply depolarizing error via stochastic Pauli twirling.
        
        With probability p, applies a random Pauli (X, Y, or Z) to each target qubit.
        This is a stochastic approximation of the depolarizing channel.
        
        Args:
            p: Depolarizing probability (total error rate)
            targets: List of qubit indices
        """
        if p <= 0:
            return  # No error to apply
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        paulis = [I, X, Y, Z]
        
        # For each target qubit, decide whether to apply error
        for qubit in targets:
            if self.rng.random() < p:
                # Apply random Pauli (excluding I for efficiency, scale p accordingly)
                # With probability p, choose uniformly from {X, Y, Z}
                pauli_idx = self.rng.integers(1, 4)  # 1, 2, or 3
                pauli = paulis[pauli_idx]
                self.apply_unitary(pauli, [qubit])
    
    def apply_kraus_stochastic(
        self, 
        kraus_ops: list[np.ndarray], 
        targets: list[int]
    ) -> None:
        """
        Apply Kraus channel via stochastic unraveling (quantum jump method).
        
        This is the Monte Carlo wavefunction (MCWF) approach:
        1. Compute probabilities p_k = ⟨ψ|K_k^† K_k|ψ⟩ for each Kraus operator
        2. Sample one Kraus operator k according to probabilities
        3. Apply K_k|ψ⟩ and renormalize
        
        Args:
            kraus_ops: List of Kraus operators (each 2^k × 2^k for k qubits)
            targets: List of k qubit indices
        """
        # Compute probabilities for each Kraus operator
        # p_k = ⟨ψ|K_k^† K_k|ψ⟩
        probs = []
        for K in kraus_ops:
            # Apply K_k^† K_k to get effective measurement operator
            state_copy = self.state.copy()
            
            # Reshape and apply K to copy
            k = len(targets)
            K_arr = np.asarray(K, dtype=np.complex128)
            state_tensor = state_copy.reshape([2] * self.n_qubits)
            K_tensor = K_arr.reshape([2] * (2 * k))
            
            # Build einsum for K application (same as apply_unitary)
            in_indices = [chr(ord('a') + i) for i in range(self.n_qubits)]
            out_indices = in_indices.copy()
            new_idx_start = ord('a') + self.n_qubits
            K_out = []
            K_in = []
            
            for i, t in enumerate(targets):
                new_label = chr(new_idx_start + i)
                K_out.append(new_label)
                K_in.append(in_indices[t])
                out_indices[t] = new_label
            
            state_idx = ''.join(in_indices)
            K_idx = ''.join(K_out) + ''.join(K_in)
            result_idx = ''.join(out_indices)
            einsum_expr = f"{state_idx},{K_idx}->{result_idx}"
            
            result_tensor = np.einsum(einsum_expr, state_tensor, K_tensor)
            result_state = result_tensor.reshape(2 ** self.n_qubits)
            
            # Probability is norm squared of result
            prob = float(np.abs(np.vdot(result_state, result_state)))
            probs.append(prob)
        
        # Normalize probabilities (should sum to ~1 for trace-preserving channels)
        probs = np.array(probs)
        probs /= probs.sum()
        
        # Sample which Kraus operator to apply
        k_idx = self.rng.choice(len(kraus_ops), p=probs)
        
        # Apply the selected Kraus operator
        K_selected = kraus_ops[k_idx]
        self.apply_unitary(K_selected, targets)
        
        # Renormalize state
        norm = np.linalg.norm(self.state)
        if norm > 1e-15:
            self.state /= norm
    
    @staticmethod
    def _u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
        """
        Construct U3 gate matrix.
        
        U3(θ, φ, λ) = [[cos(θ/2),           -exp(iλ)sin(θ/2)          ],
                       [exp(iφ)sin(θ/2),     exp(i(φ+λ))cos(θ/2)      ]]
        """
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        return np.array([
            [cos_half, -np.exp(1j * lam) * sin_half],
            [np.exp(1j * phi) * sin_half, np.exp(1j * (phi + lam)) * cos_half]
        ], dtype=np.complex128)
