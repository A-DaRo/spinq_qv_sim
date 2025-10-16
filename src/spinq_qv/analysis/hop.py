"""
Heavy-output probability (HOP) computation for Quantum Volume.

Implements IBM QV metric: heavy outputs are those with ideal probability
greater than the median probability.
"""

from typing import Dict, List, Tuple
import numpy as np


def identify_heavy_outputs(
    ideal_probabilities: np.ndarray,
    threshold_type: str = "median",
) -> np.ndarray:
    """
    Identify heavy outputs based on ideal probabilities.
    
    Args:
        ideal_probabilities: Ideal (noiseless) output probabilities (length 2^m)
        threshold_type: Method to define heavy outputs ("median" or "mean")
    
    Returns:
        Boolean array where True indicates heavy outputs
    """
    if threshold_type == "median":
        threshold = np.median(ideal_probabilities)
    elif threshold_type == "mean":
        threshold = np.mean(ideal_probabilities)
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")
    
    # Heavy outputs are those with probability > threshold
    is_heavy = ideal_probabilities > threshold
    
    return is_heavy


def compute_hop_from_result(
    measured_counts: Dict[str, int],
    ideal_probabilities: np.ndarray,
    threshold_type: str = "median",
) -> Tuple[float, int, int]:
    """
    Compute heavy-output probability from measurement results.
    
    Args:
        measured_counts: Dictionary mapping bitstrings to counts
        ideal_probabilities: Ideal output probabilities
        threshold_type: How to define heavy outputs ("median" or "mean")
    
    Returns:
        Tuple of (hop, heavy_count, total_shots)
        - hop: Heavy-output probability (fraction in [0, 1])
        - heavy_count: Number of shots that produced heavy outputs
        - total_shots: Total number of measurement shots
    """
    # Identify heavy outputs
    is_heavy = identify_heavy_outputs(ideal_probabilities, threshold_type)
    
    # Count shots landing in heavy outputs
    m = int(np.log2(len(ideal_probabilities)))
    heavy_count = 0
    total_shots = 0
    
    for bitstring, count in measured_counts.items():
        # Convert bitstring to index
        if len(bitstring) != m:
            raise ValueError(
                f"Bitstring length {len(bitstring)} doesn't match width {m}"
            )
        
        index = int(bitstring, 2)
        total_shots += count
        
        if is_heavy[index]:
            heavy_count += count
    
    # Compute HOP
    hop = heavy_count / total_shots if total_shots > 0 else 0.0
    
    return hop, heavy_count, total_shots


def compute_hop_batch(
    measured_counts_list: List[Dict[str, int]],
    ideal_probabilities_list: List[np.ndarray],
    threshold_type: str = "median",
) -> np.ndarray:
    """
    Compute HOP for multiple circuits.
    
    Args:
        measured_counts_list: List of measurement count dictionaries
        ideal_probabilities_list: List of ideal probability arrays
        threshold_type: How to define heavy outputs
    
    Returns:
        Array of HOP values (one per circuit)
    """
    if len(measured_counts_list) != len(ideal_probabilities_list):
        raise ValueError("Counts and probabilities lists must have same length")
    
    hops = []
    
    for counts, probs in zip(measured_counts_list, ideal_probabilities_list):
        hop, _, _ = compute_hop_from_result(counts, probs, threshold_type)
        hops.append(hop)
    
    return np.array(hops)


def theoretical_hop_noiseless() -> float:
    """
    Return theoretical HOP for noiseless ideal circuits.
    
    For noiseless sampling, all shots land in the true distribution,
    so HOP should be close to the fraction of probability mass in
    heavy outputs, which is approximately 0.5 + statistical fluctuations.
    
    Returns:
        Expected HOP for noiseless case (approximately 0.5)
    """
    # For uniform random circuits with median threshold,
    # heavy outputs contain ~50% of probability mass
    return 0.5


def qv_threshold() -> float:
    """
    Return IBM QV success threshold.
    
    A circuit width m achieves Quantum Volume 2^m if:
    - Mean HOP > 2/3
    - Lower 95% confidence interval > 2/3
    
    Returns:
        QV success threshold (2/3)
    """
    return 2.0 / 3.0
