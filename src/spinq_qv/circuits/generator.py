"""
IBM-style Quantum Volume circuit generator.

Generates random square circuits (width = depth = m) with random single-qubit
rotations and random two-qubit entangling gates following the IBM QV protocol.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.stats import unitary_group

from spinq_qv.io.formats import CircuitSpec


def generate_qv_circuit(m: int, seed: int) -> CircuitSpec:
    """
    Generate an IBM-style Quantum Volume circuit.
    
    Creates a random square circuit (width = depth = m) with:
    - Random single-qubit unitaries (SU(2) rotations)
    - Random two-qubit entangling unitaries (SU(4))
    - Random permutations of qubit pairings in each layer
    
    Args:
        m: Circuit width and depth (number of qubits)
        seed: Random seed for reproducibility
    
    Returns:
        CircuitSpec containing the circuit structure
    """
    rng = np.random.default_rng(seed)
    
    circuit = CircuitSpec(
        width=m,
        depth=m,
        seed=seed,
        metadata={
            "type": "qv",
            "description": f"IBM QV circuit, m={m}",
        }
    )
    
    layer_idx = 0
    
    for depth_layer in range(m):
        # Alternate between single-qubit and two-qubit layers
        
        # Single-qubit layer: random SU(2) on each qubit
        for qubit in range(m):
            # Generate random SU(2) angles (using Euler decomposition)
            theta, phi, lam = _random_su2_angles(rng)
            
            circuit.add_gate(
                gate_type="u3",
                qubits=[qubit],
                params={"theta": theta, "phi": phi, "lambda": lam},
                layer=layer_idx,
            )
        
        layer_idx += 1
        
        # Two-qubit layer: random pairings + random SU(4) unitaries
        if depth_layer < m - 1 or m == 2:  # Last layer also has 2Q gates for m=2
            pairs = _generate_random_pairing(m, rng)
            
            for q1, q2 in pairs:
                # For each pair, apply a random two-qubit unitary
                # We decompose it into a standard basis (e.g., CZ + single-qubit)
                # For simplicity, use a random SU(4) represented parametrically
                
                # Generate 15 parameters for SU(4) (4x4 unitary has 15 real DOF)
                params = _random_su4_params(rng)
                
                circuit.add_gate(
                    gate_type="su4",
                    qubits=[q1, q2],
                    params=params,
                    layer=layer_idx,
                )
            
            layer_idx += 1
    
    return circuit


def compute_ideal_probabilities(circuit_spec: CircuitSpec) -> np.ndarray:
    """
    Compute ideal (noiseless) output probabilities for a QV circuit.
    
    Uses statevector simulation to compute exact probabilities for all
    2^m computational basis states.
    
    Args:
        circuit_spec: Circuit specification
    
    Returns:
        Array of probabilities (length 2^m), indexed by computational basis states
    """
    m = circuit_spec.width
    
    # Initialize statevector: |0...0>
    statevector = np.zeros(2**m, dtype=np.complex128)
    statevector[0] = 1.0
    
    # Apply gates sequentially
    for gate in circuit_spec.gates:
        gate_type = gate["type"]
        qubits = gate["qubits"]
        params = gate.get("params", {})
        
        if gate_type == "u3":
            # Single-qubit gate
            statevector = _apply_u3(statevector, qubits[0], m, params)
        
        elif gate_type == "su4":
            # Two-qubit gate
            statevector = _apply_su4(statevector, qubits[0], qubits[1], m, params)
        
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    # Compute probabilities
    probabilities = np.abs(statevector) ** 2
    
    # Normalize (should already be normalized, but ensure numerical stability)
    probabilities /= probabilities.sum()
    
    return probabilities


def _random_su2_angles(rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Generate random Euler angles for SU(2) gate (U3).
    
    U3(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
    
    Args:
        rng: Random number generator
    
    Returns:
        (theta, phi, lambda) angles in radians
    """
    # Sample uniformly on SU(2) using Haar measure
    # For Euler angles: θ ∈ [0, π], φ, λ ∈ [0, 2π]
    theta = np.arccos(1 - 2 * rng.random())  # Haar measure on sphere
    phi = rng.uniform(0, 2 * np.pi)
    lam = rng.uniform(0, 2 * np.pi)
    
    return theta, phi, lam


def _random_su4_params(rng: np.random.Generator) -> dict:
    """
    Generate parameters for a random SU(4) unitary.
    
    For simplicity, we represent SU(4) using a canonical decomposition.
    Real implementation would use KAK decomposition or similar.
    
    Args:
        rng: Random number generator
    
    Returns:
        Dictionary of parameters defining the SU(4) gate
    """
    # Generate a random 4x4 unitary using Haar measure
    # Store as matrix elements (for exact simulation)
    # Note: This is memory-intensive but exact
    
    unitary = unitary_group.rvs(4, random_state=rng)
    
    # Store matrix as flattened real/imag parts
    real_part = unitary.real.flatten().tolist()
    imag_part = unitary.imag.flatten().tolist()
    
    return {
        "unitary_real": real_part,
        "unitary_imag": imag_part,
    }


def _generate_random_pairing(m: int, rng: np.random.Generator) -> list:
    """
    Generate random qubit pairings for two-qubit layer.
    
    Creates a random perfect matching (or near-perfect for odd m).
    
    Args:
        m: Number of qubits
        rng: Random number generator
    
    Returns:
        List of (qubit1, qubit2) pairs
    """
    qubits = list(range(m))
    rng.shuffle(qubits)
    
    pairs = []
    for i in range(0, m - 1, 2):
        pairs.append((qubits[i], qubits[i + 1]))
    
    # If odd number of qubits, last one is idle
    # (IBM QV typically uses even m, but we handle odd case)
    
    return pairs


def _apply_u3(
    statevector: np.ndarray,
    qubit: int,
    num_qubits: int,
    params: dict,
) -> np.ndarray:
    """
    Apply U3 gate to statevector.
    
    U3(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
    
    Args:
        statevector: Current statevector (length 2^num_qubits)
        qubit: Target qubit index
        num_qubits: Total number of qubits
        params: Gate parameters {theta, phi, lambda}
    
    Returns:
        Updated statevector
    """
    theta = params["theta"]
    phi = params["phi"]
    lam = params["lambda"]
    
    # U3 matrix
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    
    u3_matrix = np.array([
        [cos_half, -np.exp(1j * lam) * sin_half],
        [np.exp(1j * phi) * sin_half, np.exp(1j * (phi + lam)) * cos_half]
    ], dtype=np.complex128)
    
    # Apply to statevector
    return _apply_single_qubit_gate(statevector, qubit, num_qubits, u3_matrix)


def _apply_su4(
    statevector: np.ndarray,
    qubit1: int,
    qubit2: int,
    num_qubits: int,
    params: dict,
) -> np.ndarray:
    """
    Apply arbitrary SU(4) two-qubit gate.
    
    Args:
        statevector: Current statevector
        qubit1: First qubit index
        qubit2: Second qubit index
        num_qubits: Total number of qubits
        params: Gate parameters (unitary matrix)
    
    Returns:
        Updated statevector
    """
    # Reconstruct unitary from stored real/imag parts
    real_part = np.array(params["unitary_real"]).reshape(4, 4)
    imag_part = np.array(params["unitary_imag"]).reshape(4, 4)
    unitary = real_part + 1j * imag_part
    
    # Apply to statevector
    return _apply_two_qubit_gate(
        statevector, qubit1, qubit2, num_qubits, unitary
    )


def _apply_single_qubit_gate(
    statevector: np.ndarray,
    qubit: int,
    num_qubits: int,
    gate_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply a single-qubit gate to the statevector.
    
    Uses tensor product structure to efficiently apply gate.
    
    Args:
        statevector: Current statevector (length 2^num_qubits)
        qubit: Target qubit (0-indexed)
        num_qubits: Total number of qubits
        gate_matrix: 2x2 gate matrix
    
    Returns:
        Updated statevector
    """
    # Reshape statevector to separate target qubit
    # Shape: (2, 2, ..., 2) with num_qubits axes
    shape = [2] * num_qubits
    sv_tensor = statevector.reshape(shape)
    
    # Move target qubit to first axis
    sv_tensor = np.moveaxis(sv_tensor, qubit, 0)
    
    # Apply gate (Einstein summation)
    # gate[i,j] * sv[j, ...] -> sv_new[i, ...]
    sv_tensor = np.einsum('ij,j...->i...', gate_matrix, sv_tensor)
    
    # Move axis back
    sv_tensor = np.moveaxis(sv_tensor, 0, qubit)
    
    # Flatten back to statevector
    return sv_tensor.flatten()


def _apply_two_qubit_gate(
    statevector: np.ndarray,
    qubit1: int,
    qubit2: int,
    num_qubits: int,
    gate_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply a two-qubit gate to the statevector.
    
    Args:
        statevector: Current statevector (length 2^num_qubits)
        qubit1: First qubit index
        qubit2: Second qubit index
        num_qubits: Total number of qubits
        gate_matrix: 4x4 gate matrix (in computational basis |00>, |01>, |10>, |11>)
    
    Returns:
        Updated statevector
    """
    # Reshape to tensor
    shape = [2] * num_qubits
    sv_tensor = statevector.reshape(shape)
    
    # Move target qubits to first two axes
    sv_tensor = np.moveaxis(sv_tensor, [qubit1, qubit2], [0, 1])
    
    # Reshape gate_matrix to (2, 2, 2, 2) for tensor contraction
    gate_tensor = gate_matrix.reshape(2, 2, 2, 2)
    
    # Apply gate: gate[i,j,k,l] * sv[k,l,...] -> sv_new[i,j,...]
    sv_tensor = np.einsum('ijkl,kl...->ij...', gate_tensor, sv_tensor)
    
    # Move axes back
    sv_tensor = np.moveaxis(sv_tensor, [0, 1], [qubit1, qubit2])
    
    # Flatten
    return sv_tensor.flatten()
