"""
Validation experiments using Randomized Benchmarking (RB).

Runs RB sequences to extract gate fidelities and compare with configured values.
This validates that the noise model reproduces the target fidelities.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.optimize import curve_fit
import logging

from spinq_qv.config.schemas import Config
from spinq_qv.sim.statevector import StatevectorBackend
from spinq_qv.noise.builder import NoiseModelBuilder
from spinq_qv.noise.channels import depolarizing_kraus


logger = logging.getLogger(__name__)


def clifford_1q() -> np.ndarray:
    """
    Return a random single-qubit Clifford gate.
    
    For simplicity, sample from a small set of representative Cliffords:
    {I, X, Y, Z, H, S, S†, sqrt(X), sqrt(Y)}
    
    In a full implementation, sample uniformly from the 24 single-qubit Cliffords.
    """
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    
    cliffords = [I, X, Y, Z, H, S]
    idx = np.random.randint(0, len(cliffords))
    return cliffords[idx]


def clifford_2q() -> np.ndarray:
    """
    Return a random two-qubit Clifford gate.
    
    For simplicity, use CNOT and random single-qubit Cliffords.
    Full RB would sample uniformly from the two-qubit Clifford group.
    """
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)
    
    # With 50% probability, return CNOT or random local Cliffords
    if np.random.random() < 0.5:
        return CNOT
    else:
        # Tensor product of two single-qubit Cliffords
        C1 = clifford_1q()
        C2 = clifford_1q()
        return np.kron(C1, C2)


def run_rb_sequence(
    n_qubits: int,
    sequence_length: int,
    noise_p: float,
    seed: int,
    two_qubit: bool = False
) -> float:
    """
    Run one RB sequence and return survival probability.
    
    Args:
        n_qubits: Number of qubits (1 or 2)
        sequence_length: Number of Clifford gates in sequence
        noise_p: Depolarizing probability per gate
        seed: Random seed
        two_qubit: If True, use two-qubit Cliffords
    
    Returns:
        Survival probability (probability of returning to |0⟩)
    """
    np.random.seed(seed)
    backend = StatevectorBackend(n_qubits, seed=seed)
    
    # Generate random Clifford sequence
    cliffords = []
    for _ in range(sequence_length):
        if two_qubit:
            C = clifford_2q()
        else:
            C = clifford_1q()
        cliffords.append(C)
    
    # Compute inverse to return to |0⟩
    total_unitary = np.eye(2**n_qubits, dtype=np.complex128)
    for C in cliffords:
        total_unitary = C @ total_unitary
    
    inverse = total_unitary.conj().T
    
    # Apply sequence with noise
    targets = list(range(n_qubits))
    for C in cliffords:
        backend.apply_unitary(C, targets)
        # Apply depolarizing noise after each gate
        if noise_p > 0:
            if two_qubit and n_qubits == 2:
                # Use proper 2-qubit depolarizing Kraus channel
                from spinq_qv.noise.channels import depolarizing_kraus_2q
                dep_kraus_2q = depolarizing_kraus_2q(noise_p)
                backend.apply_kraus_stochastic(dep_kraus_2q, targets)
            else:
                # Single-qubit case
                from spinq_qv.noise.channels import depolarizing_kraus
                dep_kraus = depolarizing_kraus(noise_p)
                for qubit in targets:
                    backend.apply_kraus_stochastic(dep_kraus, [qubit])
    
    # Apply inverse
    backend.apply_unitary(inverse, targets)
    if noise_p > 0:
        if two_qubit and n_qubits == 2:
            from spinq_qv.noise.channels import depolarizing_kraus_2q
            dep_kraus_2q = depolarizing_kraus_2q(noise_p)
            backend.apply_kraus_stochastic(dep_kraus_2q, targets)
        else:
            from spinq_qv.noise.channels import depolarizing_kraus
            dep_kraus = depolarizing_kraus(noise_p)
            for qubit in targets:
                backend.apply_kraus_stochastic(dep_kraus, [qubit])
    
    # Measure probability of |0...0⟩
    probs = backend.get_probabilities()
    return float(probs[0])


def rb_decay_model(m: np.ndarray, A: float, p: float, B: float) -> np.ndarray:
    """
    RB decay model: F(m) = A * p^m + B
    
    where p is the depolarizing parameter related to average gate fidelity.
    """
    return A * (p ** m) + B


def extract_fidelity_from_rb(
    sequence_lengths: List[int],
    survival_probs: List[float],
    n_qubits: int
) -> Tuple[float, Dict[str, Any]]:
    """
    Fit RB decay curve and extract average gate fidelity.
    
    Args:
        sequence_lengths: List of Clifford sequence lengths
        survival_probs: List of survival probabilities
        n_qubits: Number of qubits
    
    Returns:
        Tuple of (average_gate_fidelity, fit_info_dict)
    """
    # Fit decay model
    m_arr = np.array(sequence_lengths)
    f_arr = np.array(survival_probs)
    
    # Initial guess: A ≈ 0.5, p ≈ 0.99, B ≈ 0.5/2^n
    p0 = [0.5, 0.99, 0.5 / (2**n_qubits)]
    
    try:
        popt, pcov = curve_fit(rb_decay_model, m_arr, f_arr, p0=p0)
        A, p_fit, B = popt
        
        # Extract fidelity from p
        # For single qubit: F = 1 - (1-p)/2
        # For two qubits: F = 1 - 3(1-p)/4
        if n_qubits == 1:
            F_avg = 1 - (1 - p_fit) / 2
        elif n_qubits == 2:
            F_avg = 1 - 3 * (1 - p_fit) / 4
        else:
            # General formula for n qubits
            d = 2 ** n_qubits
            F_avg = 1 - (d - 1) * (1 - p_fit) / d
        
        fit_info = {
            'A': float(A),
            'p': float(p_fit),
            'B': float(B),
            'fidelity': float(F_avg),
            'success': True
        }
    except Exception as e:
        logger.warning(f"RB fit failed: {e}")
        fit_info = {
            'success': False,
            'error': str(e),
            'fidelity': 0.0
        }
        F_avg = 0.0
    
    return F_avg, fit_info


def validate_rb_single_qubit(config: Config, n_sequences: int = 10, seed: int = 42) -> Dict[str, Any]:
    """
    Run single-qubit RB to validate F1.
    
    Args:
        config: Configuration with target F1
        n_sequences: Number of RB sequences per length
        seed: Random seed
    
    Returns:
        Dictionary with extracted fidelity and comparison to target
    """
    # Get depolarizing probability from config
    F1_target = config.device.F1
    builder = NoiseModelBuilder(config.device.model_dump())
    p_dep = builder.fidelity_to_depolarizing_p(F1_target, two_qubit=False)
    
    # RB sequence lengths
    sequence_lengths = [1, 2, 4, 8, 16, 32]
    
    # Run RB
    survival_probs_per_length = []
    for m in sequence_lengths:
        probs = []
        for i in range(n_sequences):
            p = run_rb_sequence(
                n_qubits=1,
                sequence_length=m,
                noise_p=p_dep,
                seed=seed + i + m * 1000,
                two_qubit=False
            )
            probs.append(p)
        avg_prob = np.mean(probs)
        survival_probs_per_length.append(avg_prob)
    
    # Extract fidelity
    F_extracted, fit_info = extract_fidelity_from_rb(
        sequence_lengths, survival_probs_per_length, n_qubits=1
    )
    
    result = {
        'target_F1': float(F1_target),
        'extracted_F1': float(F_extracted),
        'error': float(abs(F_extracted - F1_target)),
        'relative_error': float(abs(F_extracted - F1_target) / F1_target),
        'fit_info': fit_info,
        'sequence_lengths': sequence_lengths,
        'survival_probs': survival_probs_per_length,
    }
    
    logger.info(f"Single-qubit RB: Target F1={F1_target:.6f}, Extracted F1={F_extracted:.6f}")
    
    return result


def validate_rb_two_qubit(config: Config, n_sequences: int = 10, seed: int = 43) -> Dict[str, Any]:
    """
    Run two-qubit RB to validate F2.
    
    Args:
        config: Configuration with target F2
        n_sequences: Number of RB sequences per length
        seed: Random seed
    
    Returns:
        Dictionary with extracted fidelity and comparison to target
    """
    # Get depolarizing probability from config
    F2_target = config.device.F2
    builder = NoiseModelBuilder(config.device.model_dump())
    p_dep = builder.fidelity_to_depolarizing_p(F2_target, two_qubit=True)
    
    # RB sequence lengths (shorter for two-qubit due to noise accumulation)
    sequence_lengths = [1, 2, 4, 8, 16]
    
    # Run RB
    survival_probs_per_length = []
    for m in sequence_lengths:
        probs = []
        for i in range(n_sequences):
            p = run_rb_sequence(
                n_qubits=2,
                sequence_length=m,
                noise_p=p_dep,
                seed=seed + i + m * 1000,
                two_qubit=True
            )
            probs.append(p)
        avg_prob = np.mean(probs)
        survival_probs_per_length.append(avg_prob)
    
    # Extract fidelity
    F_extracted, fit_info = extract_fidelity_from_rb(
        sequence_lengths, survival_probs_per_length, n_qubits=2
    )
    
    result = {
        'target_F2': float(F2_target),
        'extracted_F2': float(F_extracted),
        'error': float(abs(F_extracted - F2_target)),
        'relative_error': float(abs(F_extracted - F2_target) / F2_target),
        'fit_info': fit_info,
        'sequence_lengths': sequence_lengths,
        'survival_probs': survival_probs_per_length,
    }
    
    logger.info(f"Two-qubit RB: Target F2={F2_target:.6f}, Extracted F2={F_extracted:.6f}")
    
    return result


def main():
    """Run validation experiments and save results."""
    import json
    from pathlib import Path
    
    # Load config
    config = Config.from_yaml("examples/configs/baseline.yaml")
    
    # Run validations
    logger.info("Running single-qubit RB validation...")
    result_1q = validate_rb_single_qubit(config, n_sequences=20, seed=42)
    
    logger.info("Running two-qubit RB validation...")
    result_2q = validate_rb_two_qubit(config, n_sequences=20, seed=43)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output = {
        'single_qubit': result_1q,
        'two_qubit': result_2q,
    }
    
    with open(results_dir / "validate_rb.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {results_dir / 'validate_rb.json'}")
    
    # Print summary
    print("\n=== RB Validation Summary ===")
    print(f"Single-qubit: Target F1={result_1q['target_F1']:.6f}, "
          f"Extracted F1={result_1q['extracted_F1']:.6f}, "
          f"Error={result_1q['error']:.6f} ({result_1q['relative_error']*100:.2f}%)")
    print(f"Two-qubit:    Target F2={result_2q['target_F2']:.6f}, "
          f"Extracted F2={result_2q['extracted_F2']:.6f}, "
          f"Error={result_2q['error']:.6f} ({result_2q['relative_error']*100:.2f}%)")


if __name__ == "__main__":
    from spinq_qv.utils.logging_setup import setup_logging
    setup_logging()
    main()
