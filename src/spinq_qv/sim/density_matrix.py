"""
Density-matrix simulator with exact Kraus channel propagation.

Implements exact non-unitary evolution via Kraus operators. Suitable for
small qubit counts (n <= 12-14) due to memory scaling as 2^(2n).

Memory requirement: 2^(2n) complex128 values = 16 * 4^n bytes
Example: n=10 → 16 MB, n=12 → 256 MB, n=14 → 4 GB
"""

from typing import Dict, Optional, Any, List
import numpy as np
import warnings

from spinq_qv.sim.backend import SimulatorBackend


class DensityMatrixBackend(SimulatorBackend):
    """
    Density-matrix simulator using exact Kraus channel propagation.
    
    Represents quantum state as density matrix ρ of shape (2^n, 2^n).
    Gates are applied via ρ -> U ρ U†.
    Noise channels applied via Kraus sum: ρ -> Σ_k K_k ρ K_k†.
    
    WARNING: Memory scales as 4^n. Practical limit around n=14 on typical machines.
    """
    
    # Safety limits
    MAX_QUBITS_ERROR = 16  # Hard error if exceeded
    MAX_QUBITS_WARNING = 12  # Warning if exceeded
    
    def __init__(
        self,
        n_qubits: int,
        seed: Optional[int] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize density-matrix simulator.
        
        Args:
            n_qubits: Number of qubits
            seed: Random seed for measurement sampling
            use_gpu: Ignored (CPU-only implementation)
        
        Raises:
            ValueError: If n_qubits exceeds safety limits
        """
        if n_qubits > self.MAX_QUBITS_ERROR:
            raise ValueError(
                f"Density matrix requires too much memory for {n_qubits} qubits. "
                f"Maximum supported: {self.MAX_QUBITS_ERROR}. "
                f"Consider using MCWF backend for larger systems."
            )
        
        if n_qubits > self.MAX_QUBITS_WARNING:
            memory_gb = 16 * (4 ** n_qubits) / (1024 ** 3)
            warnings.warn(
                f"Density matrix for {n_qubits} qubits requires ~{memory_gb:.2f} GB memory. "
                f"Consider using smaller systems or MCWF backend.",
                ResourceWarning
            )
        
        super().__init__(n_qubits, seed)
        
        # CPU-only NumPy backend
        self.use_gpu = False
        
        # Initialize density matrix to pure state |0...0⟩
        self.rho: Optional[np.ndarray] = None
        self.init_state()
    
    def init_state(self, state: Optional[np.ndarray] = None) -> None:
        """
        Initialize density matrix.
        
        Args:
            state: Optional initial density matrix (2^n × 2^n). 
                   If None, initialize to |0...0⟩⟨0...0|.
                   Can also be a statevector (will be converted to density matrix).
        """
        dim = 2 ** self.n_qubits
        
        if state is None:
            # Initialize to pure state |0...0⟩⟨0...0|
            self.rho = np.zeros((dim, dim), dtype=np.complex128)
            self.rho[0, 0] = 1.0
        else:
            if state.ndim == 1:
                # Convert statevector to density matrix: ρ = |ψ⟩⟨ψ|
                if state.shape[0] != dim:
                    raise ValueError(
                        f"Statevector shape {state.shape} does not match "
                        f"expected ({dim},)"
                    )
                # Normalize if needed
                norm = np.linalg.norm(state)
                if abs(norm - 1.0) > 1e-10:
                    state = state / norm
                self.rho = np.outer(state, state.conj())
            
            elif state.ndim == 2:
                # Use provided density matrix
                if state.shape != (dim, dim):
                    raise ValueError(
                        f"Density matrix shape {state.shape} does not match "
                        f"expected ({dim}, {dim})"
                    )
                # Normalize trace if needed
                trace = np.trace(state)
                if abs(trace - 1.0) > 1e-10:
                    state = state / trace
                self.rho = state.astype(np.complex128)
            else:
                raise ValueError("state must be 1D (statevector) or 2D (density matrix)")
    
    def apply_unitary(self, unitary: np.ndarray, targets: list[int]) -> None:
        """
        Apply unitary gate: ρ -> U ρ U†.
        
        For subsystem targets, embeds unitary in full Hilbert space.
        Uses tensor contraction to correctly handle arbitrary target orderings.
        
        Args:
            unitary: Unitary matrix (2^k × 2^k for k target qubits)
            targets: List of k qubit indices (0-indexed)
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
        
        unitary = np.asarray(unitary, dtype=np.complex128)
        
        # Apply U ρ U† using tensor contraction
        # Density matrix ρ[i,j] represents ⟨i|ρ|j⟩
        # U ρ U† computes: Σ_{k,l} U[i,k] ρ[k,l] U†[l,j] = Σ_{k,l} U[i,k] ρ[k,l] U*[j,l]
        # So U acts on the BRA (first) index, U† acts on the KET (second) index
        
        # Reshape density matrix to tensor: (2,2,...,2) x (2,2,...,2)
        rho_tensor = self.rho.reshape([2] * (2 * self.n_qubits))
        
        # Reshape unitary to tensor: (2,2,...,2) x (2,2,...,2) for k qubits
        U_tensor = unitary.reshape([2] * (2 * k))
        U_dag_tensor = unitary.conj().T.reshape([2] * (2 * k))
        
        # Build indices for einsum
        # Bra indices (row): 0, 1, ..., n-1
        # Ket indices (col): n, n+1, ..., 2n-1  
        n = self.n_qubits
        
        # Initial ρ tensor indices
        rho_bra = list(range(n))
        rho_ket = list(range(n, 2*n))
        
        # First apply U on the BRA (row) side: U[new_bra, old_bra] × ρ[old_bra, ket]
        # U tensor: new output indices and old input indices
        U_out = list(range(2*n, 2*n + k))
        U_in = list(range(2*n + k, 2*n + 2*k))
        
        # After U ρ: replace bra indices at targets with new indices from U
        result_bra = rho_bra.copy()
        rho_bra_contracted = rho_bra.copy()
        
        for i, t in enumerate(targets):
            result_bra[t] = U_out[i]
            rho_bra_contracted[t] = U_in[i]
        
        # Build einsum: U[bra_new, bra_old] × ρ[bra_old, ket] → result[bra_new, ket]
        U_idx = ''.join(chr(ord('a') + i) for i in U_out + U_in)
        rho_idx = ''.join(chr(ord('a') + i) for i in rho_bra_contracted + rho_ket)
        result1_idx = ''.join(chr(ord('a') + i) for i in result_bra + rho_ket)
        
        einsum1 = f"{U_idx},{rho_idx}->{result1_idx}"
        rho_tensor = np.einsum(einsum1, U_tensor, rho_tensor)
        
        # Now apply U† on the KET (column) side: (U ρ)[bra, old_ket] × U†[new_ket, old_ket]
        # After first einsum, we have intermediate[result_bra, rho_ket]
        
        # New indices for U† output
        U_dag_out = list(range(2*n + 2*k, 2*n + 3*k))
        
        # U† input indices: use the current ket labels at target positions
        U_dag_in = [rho_ket[t] for t in targets]
        
        # Build result ket indices (replace contracted positions with new labels)
        result_ket_labels = rho_ket.copy()
        
        for i, t in enumerate(targets):
            result_ket_labels[t] = U_dag_out[i]
        
        # Build einsum: intermediate[result_bra, rho_ket] × U†[U_dag_in, U_dag_out] → result[result_bra, result_ket]
        # Note: U† has input indices first, output second (for matrix mult on right)
        rho2_idx = ''.join(chr(ord('a') + i) for i in result_bra + rho_ket)
        U_dag_idx = ''.join(chr(ord('a') + i) for i in U_dag_in + U_dag_out)
        result2_idx = ''.join(chr(ord('a') + i) for i in result_bra + result_ket_labels)
        
        einsum2 = f"{rho2_idx},{U_dag_idx}->{result2_idx}"
        rho_tensor = np.einsum(einsum2, rho_tensor, U_dag_tensor)
        
        # Reshape back to matrix
        self.rho = rho_tensor.reshape(2 ** self.n_qubits, 2 ** self.n_qubits)
    
    def apply_kraus(self, kraus_ops: List[np.ndarray], targets: list[int]) -> None:
        """
        Apply Kraus channel: ρ -> Σ_k K_k ρ K_k†.
        
        Args:
            kraus_ops: List of Kraus operators (each 2^k × 2^k for k qubits)
            targets: List of k qubit indices
        """
        if len(kraus_ops) == 0:
            return  # No-op
        
        k = len(targets)
        expected_dim = 2 ** k
        
        # Validate dimensions
        for i, K in enumerate(kraus_ops):
            if K.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Kraus operator {i} shape {K.shape} does not match "
                    f"expected ({expected_dim}, {expected_dim})"
                )
        
        # Validate target qubit indices
        if not all(0 <= t < self.n_qubits for t in targets):
            raise ValueError(
                f"Target qubits {targets} out of range [0, {self.n_qubits})"
            )
        
        # Apply Kraus sum
        new_rho = np.zeros_like(self.rho, dtype=np.complex128)
        
        for K in kraus_ops:
            # Embed Kraus operator in full space if needed
            if k == self.n_qubits:
                K_full = K
            else:
                K_full = self._embed_subsystem_operator(K, targets)
            
            # Accumulate: ρ += K ρ K†
            new_rho += K_full @ self.rho @ K_full.conj().T
        
        self.rho = new_rho
    
    def _embed_subsystem_operator(
        self, 
        op: np.ndarray, 
        targets: list[int]
    ) -> np.ndarray:
        """
        Embed a subsystem operator into the full Hilbert space.
        
        For targets [t1, t2, ...], constructs full-space operator by
        tensor product with identity on non-target qubits, handling
        arbitrary (possibly unsorted) target qubit orderings.
        
        Uses the same tensor contraction method as statevector backend
        to ensure consistency.
        
        Args:
            op: Operator on k qubits (2^k × 2^k)
            targets: List of k target qubit indices (can be in any order)
        
        Returns:
            Full-space operator (2^n × 2^n)
        """
        k = len(targets)
        n = self.n_qubits
        
        # For single-qubit gates, use simple Kronecker product
        if k == 1:
            target = targets[0]
            I = np.eye(2, dtype=np.complex128)
            # Build I ⊗ ... ⊗ op ⊗ ... ⊗ I
            full_op = np.array([[1.0]], dtype=np.complex128)
            for q in range(n):
                if q == target:
                    full_op = np.kron(full_op, op)
                else:
                    full_op = np.kron(full_op, I)
            return full_op
        
        # For multi-qubit gates, use tensor contraction approach
        # This matches the statevector backend's einsum logic
        
        # Start with identity operator in full space
        I_full = np.eye(2**n, dtype=np.complex128)
        
        # Reshape to tensor: (2, 2, ..., 2) x (2, 2, ..., 2) with n axes each
        I_tensor = I_full.reshape([2] * (2 * n))
        
        # Reshape operator to tensor: (2, 2, ..., 2) x (2, 2, ..., 2) with k axes each
        op_tensor = op.reshape([2] * (2 * k))
        
        # Build einsum contraction string
        # Identity tensor indices: [a0, a1, ..., a_{n-1}, b0, b1, ..., b_{n-1}]
        # where a_i are output indices and b_i are input indices
        
        # Operator tensor indices: [c0, c1, ..., c_{k-1}, d0, d1, ..., d_{k-1}]
        
        # We want to replace indices at target positions with operator indices
        # Result: output[targets[i]] = op_out[i], input[targets[i]] = op_in[i]
        #         output[non-targets] = input[non-targets] (identity)
        
        out_indices = list(range(n))  # [0, 1, 2, ..., n-1]
        in_indices = list(range(n, 2*n))  # [n, n+1, ..., 2n-1]
        
        # Operator indices start after all identity indices
        op_out_indices = list(range(2*n, 2*n + k))
        op_in_indices = list(range(2*n + k, 2*n + 2*k))
        
        # Build the result indices by replacing target positions
        result_out = out_indices.copy()
        result_in = in_indices.copy()
        
        for i, t in enumerate(targets):
            result_out[t] = op_out_indices[i]
            result_in[t] = op_in_indices[i]
        
        # Build einsum string
        # identity: [out_indices][in_indices]
        # operator: [op_out_indices][op_in_indices]  
        # result: [result_out][result_in]
        
        id_idx = ''.join(chr(ord('a') + i) for i in out_indices + in_indices)
        op_idx = ''.join(chr(ord('a') + i) for i in op_out_indices + op_in_indices)
        res_idx = ''.join(chr(ord('a') + i) for i in result_out + result_in)
        
        einsum_expr = f"{id_idx},{op_idx}->{res_idx}"
        
        # Perform contraction
        result_tensor = np.einsum(einsum_expr, I_tensor, op_tensor)
        
        # Reshape back to matrix
        full_op = result_tensor.reshape(2**n, 2**n)
        
        return full_op
    
    def get_density_matrix(self) -> np.ndarray:
        """Return the current density matrix as NumPy array."""
        return self.rho.copy()
    
    def get_statevector(self) -> np.ndarray:
        """
        Return statevector (only valid for pure states).
        
        Raises:
            ValueError: If state is mixed (not pure)
        """
        # Check purity: Tr(ρ²) ≈ 1
        purity = np.real(np.trace(self.rho @ self.rho))
        if abs(purity - 1.0) > 1e-6:
            raise ValueError(
                f"Cannot extract statevector from mixed state (purity={purity:.6f})"
            )
        
        # Extract statevector from pure density matrix via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(self.rho)
        max_idx = np.argmax(np.abs(eigvals))
        statevector = eigvecs[:, max_idx]
        
        # Normalize
        statevector = statevector / np.linalg.norm(statevector)
        
        return statevector
    
    def get_probabilities(self) -> np.ndarray:
        """Compute measurement probabilities from diagonal of density matrix."""
        return np.real(np.diag(self.rho)).copy()
    
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
                "Readout noise not yet implemented"
            )
        
        # Get probabilities from diagonal
        probs = self.get_probabilities()
        
        # Ensure probabilities are valid (non-negative and sum to 1)
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
                # Single-qubit U3 gate
                theta = params['theta']
                phi = params['phi']
                lam = params['lambda']
                unitary = self._u3_matrix(theta, phi, lam)
                self.apply_unitary(unitary, qubits)
                
            elif gate_type == 'su4':
                # Two-qubit SU(4) gate
                real_part = np.array(params['unitary_real'], dtype=np.float64)
                imag_part = np.array(params['unitary_imag'], dtype=np.float64)
                unitary_flat = real_part + 1j * imag_part
                unitary = unitary_flat.reshape(4, 4)
                self.apply_unitary(unitary, qubits)
                
            else:
                raise NotImplementedError(
                    f"Gate type '{gate_type}' not yet implemented"
                )
    
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
