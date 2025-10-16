"""
Statistical analysis for Quantum Volume experiments.

Implements bootstrap confidence intervals, hypothesis testing, and
QV decision rules following IBM protocol.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats


def aggregate_hops(hops: np.ndarray) -> Dict[str, float]:
    """
    Compute aggregate statistics for HOP values.
    
    Args:
        hops: Array of HOP values from multiple circuits
    
    Returns:
        Dictionary with mean_hop, std_hop, median, min, max
    """
    return {
        "mean_hop": float(np.mean(hops)),
        "std_hop": float(np.std(hops, ddof=1)) if len(hops) > 1 else 0.0,
        "median_hop": float(np.median(hops)),
        "min_hop": float(np.min(hops)),
        "max_hop": float(np.max(hops)),
        "n_circuits": len(hops),
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    statistic_fn=np.mean,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Uses percentile bootstrap method for robust CI estimation.
    
    Args:
        data: Array of observed values
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        statistic_fn: Function to compute statistic (default: mean)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (statistic_value, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_seed)
    
    # Compute observed statistic
    observed_stat = statistic_fn(data)
    
    # Bootstrap resampling
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(resample)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return observed_stat, lower_bound, upper_bound


def qv_decision_rule(
    hops: np.ndarray,
    threshold: float = 2.0 / 3.0,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Apply IBM Quantum Volume decision rule.
    
    A width m achieves QV = 2^m if:
    1. Mean HOP > threshold (typically 2/3)
    2. Lower bound of 95% CI > threshold
    
    Args:
        hops: Array of HOP values from multiple circuits
        threshold: Success threshold (default: 2/3)
        confidence_level: Confidence level for CI (default: 0.95)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
        - mean_hop: Mean HOP value
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - pass_mean: Whether mean > threshold
        - pass_ci: Whether lower CI > threshold
        - pass_qv: Whether both criteria met
        - threshold: The threshold used
    """
    # Compute mean and CI
    mean_hop, ci_lower, ci_upper = bootstrap_confidence_interval(
        data=hops,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        statistic_fn=np.mean,
        random_seed=random_seed,
    )
    
    # Check criteria
    pass_mean = mean_hop > threshold
    pass_ci = ci_lower > threshold
    pass_qv = pass_mean and pass_ci
    
    return {
        "mean_hop": mean_hop,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "pass_mean": pass_mean,
        "pass_ci": pass_ci,
        "pass_qv": pass_qv,
        "threshold": threshold,
        "n_circuits": len(hops),
    }


def hypothesis_test_above_threshold(
    hops: np.ndarray,
    threshold: float = 2.0 / 3.0,
    alternative: str = "greater",
) -> Dict[str, Any]:
    """
    Perform one-sample t-test: H0: mean(HOP) <= threshold.
    
    Args:
        hops: Array of HOP values
        threshold: Null hypothesis threshold
        alternative: Alternative hypothesis ("greater", "two-sided", "less")
    
    Returns:
        Dictionary with test statistic, p-value, and decision
    """
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(hops, threshold, alternative=alternative)
    
    # Reject null at alpha=0.05?
    reject_null = p_value < 0.05
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "reject_null": reject_null,
        "threshold": threshold,
        "alternative": alternative,
        "interpretation": (
            f"Mean HOP is {'significantly' if reject_null else 'not significantly'} "
            f"{alternative} than {threshold} (p={p_value:.4f})"
        ),
    }


def compute_statistical_power(
    n_circuits: int,
    true_mean: float,
    true_std: float,
    threshold: float = 2.0 / 3.0,
    alpha: float = 0.05,
) -> float:
    """
    Estimate statistical power for QV experiment.
    
    Power = probability of detecting true_mean > threshold given
    number of circuits and expected variability.
    
    Args:
        n_circuits: Number of circuits
        true_mean: Expected true mean HOP
        true_std: Expected standard deviation of HOP
        threshold: Decision threshold
        alpha: Significance level
    
    Returns:
        Statistical power (probability in [0, 1])
    """
    # Effect size (Cohen's d)
    effect_size = (true_mean - threshold) / true_std if true_std > 0 else 0.0
    
    # Noncentrality parameter for t-distribution
    ncp = effect_size * np.sqrt(n_circuits)
    
    # Critical t-value for one-sided test
    df = n_circuits - 1
    t_critical = stats.t.ppf(1 - alpha, df)
    
    # Power = probability that t > t_critical under alternative
    power = 1 - stats.nct.cdf(t_critical, df, ncp)
    
    return float(power)


def required_circuits_for_power(
    target_power: float,
    true_mean: float,
    true_std: float,
    threshold: float = 2.0 / 3.0,
    alpha: float = 0.05,
    max_circuits: int = 1000,
) -> int:
    """
    Estimate number of circuits needed to achieve target statistical power.
    
    Args:
        target_power: Desired power (e.g., 0.80 for 80%)
        true_mean: Expected true mean HOP
        true_std: Expected standard deviation
        threshold: Decision threshold
        alpha: Significance level
        max_circuits: Maximum to search
    
    Returns:
        Estimated number of circuits needed
    """
    # Binary search for required n
    for n in range(10, max_circuits + 1):
        power = compute_statistical_power(n, true_mean, true_std, threshold, alpha)
        if power >= target_power:
            return n
    
    return max_circuits


def aggregate_results_by_width(
    results: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Aggregate circuit results by width.
    
    Args:
        results: List of circuit result dictionaries, each with 'width' and 'hop'
    
    Returns:
        Dictionary mapping width -> aggregated statistics
    """
    # Group by width
    by_width: Dict[int, List[float]] = {}
    
    for result in results:
        width = result["width"]
        hop = result["hop"]
        
        if width not in by_width:
            by_width[width] = []
        
        by_width[width].append(hop)
    
    # Aggregate each width
    aggregated = {}
    
    for width, hops_list in by_width.items():
        hops_array = np.array(hops_list)
        
        # Compute statistics
        stats_dict = aggregate_hops(hops_array)
        
        # Apply QV decision rule
        qv_result = qv_decision_rule(hops_array)
        
        aggregated[width] = {
            **stats_dict,
            **qv_result,
            "hops": hops_list,
        }
    
    return aggregated
