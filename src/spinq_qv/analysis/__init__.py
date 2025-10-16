"""Analysis package for QV metrics and statistics."""

from spinq_qv.analysis.hop import compute_hop_from_result, identify_heavy_outputs
from spinq_qv.analysis.stats import (
    bootstrap_confidence_interval,
    qv_decision_rule,
    aggregate_hops,
)
from spinq_qv.analysis.ablation import (
    AblationStudy,
    SensitivityAnalysis,
    compute_error_budget,
    export_error_budget_to_json,
    summarize_error_budget,
)

__all__ = [
    "compute_hop_from_result",
    "identify_heavy_outputs",
    "bootstrap_confidence_interval",
    "qv_decision_rule",
    "aggregate_hops",
    "AblationStudy",
    "SensitivityAnalysis",
    "compute_error_budget",
    "export_error_budget_to_json",
    "summarize_error_budget",
]
