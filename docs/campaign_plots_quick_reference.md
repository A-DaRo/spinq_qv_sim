# Campaign Plotting Functions - Quick Reference

## At a Glance

| Function | Purpose | Key Output |
|----------|---------|------------|
| `plot_campaign_comparison()` | Overlay HOP curves | Multi-line comparison |
| `plot_achieved_qv_comparison()` | Show max QV | Horizontal bar chart |
| `plot_qv_waterfall()` | Stacked degradation view | Cascading curves |
| `plot_critical_transitions()` | Pass→fail analysis | 2-panel with margins |
| `plot_parameter_correlation_matrix()` | Parameter-QV correlations | Heatmap |
| `plot_campaign_dashboard()` | Comprehensive overview | 4-panel summary |
| `plot_3d_parameter_surface()` | 3D parameter space | Surface plot |
| `save_all_campaign_plots()` | Batch generation | All plots at once |

---

## Quick Start (3 Lines)

```python
from spinq_qv.analysis.plots import save_all_campaign_plots

# campaign_results = {"config1": {2: {...}, 3: {...}}, "config2": {...}}
save_all_campaign_plots(campaign_results, output_dir="plots", formats=["png"])
```

---

## Common Patterns

### Pattern 1: Quick Comparison
```python
from spinq_qv.analysis.plots import plot_campaign_comparison

plot_campaign_comparison(
    campaign_results,
    output_path="comparison.png",
    show_ci=True
)
```

### Pattern 2: Identify Best Config
```python
from spinq_qv.analysis.plots import plot_achieved_qv_comparison

plot_achieved_qv_comparison(
    campaign_results,
    output_path="winner.png"
)
```

### Pattern 3: Find Critical Parameters
```python
from spinq_qv.analysis.plots import plot_parameter_correlation_matrix

plot_parameter_correlation_matrix(
    campaign_results,
    campaign_configs,
    output_path="correlations.png"
)
```

### Pattern 4: Full Report
```python
from spinq_qv.analysis.plots import save_all_campaign_plots

save_all_campaign_plots(
    campaign_results,
    campaign_configs=configs,
    output_dir="final_report",
    formats=["png", "svg"]
)
```

---

## Data Structure Reference

### Campaign Results
```python
campaign_results = {
    "baseline": {
        2: {
            "mean_hop": 0.95,
            "ci_lower": 0.92,
            "ci_upper": 0.98,
            "pass_qv": True,
            "n_circuits": 50
        },
        3: {...},
        # ...
    },
    "high_fidelity": {...}
}
```

### Campaign Configs (optional, for correlation)
```python
campaign_configs = {
    "baseline": {
        "device": {
            "F1": 0.999,
            "F2": 0.998,
            "T1": 1.0,
            "T2": 99e-6,
            # ...
        }
    },
    "high_fidelity": {...}
}
```

---

## Function Signatures (Minimal)

### 1. Campaign Comparison
```python
plot_campaign_comparison(
    campaign_results: Dict[str, Dict[int, Dict]],
    output_path: Optional[Path] = None,
    show_ci: bool = True,
    threshold: float = 2/3
) -> plt.Figure
```

### 2. Achieved QV
```python
plot_achieved_qv_comparison(
    campaign_results: Dict[str, Dict[int, Dict]],
    output_path: Optional[Path] = None,
    show_values: bool = True
) -> plt.Figure
```

### 3. Waterfall
```python
plot_qv_waterfall(
    campaign_results: Dict[str, Dict[int, Dict]],
    output_path: Optional[Path] = None,
    threshold: float = 2/3
) -> plt.Figure
```

### 4. Critical Transitions
```python
plot_critical_transitions(
    campaign_results: Dict[str, Dict[int, Dict]],
    output_path: Optional[Path] = None,
    threshold: float = 2/3
) -> plt.Figure
```

### 5. Parameter Correlation
```python
plot_parameter_correlation_matrix(
    campaign_results: Dict[str, Dict[int, Dict]],
    campaign_configs: Dict[str, Dict],
    output_path: Optional[Path] = None,
    params_to_analyze: Optional[List[str]] = None
) -> plt.Figure
```

### 6. Dashboard
```python
plot_campaign_dashboard(
    campaign_results: Dict[str, Dict[int, Dict]],
    campaign_configs: Optional[Dict[str, Dict]] = None,
    output_path: Optional[Path] = None,
    threshold: float = 2/3
) -> plt.Figure
```

### 7. 3D Surface
```python
plot_3d_parameter_surface(
    param1_values: np.ndarray,
    param2_values: np.ndarray,
    qv_grid: np.ndarray,
    param1_name: str,
    param2_name: str,
    output_path: Optional[Path] = None
) -> plt.Figure
```

### 8. Batch Save
```python
save_all_campaign_plots(
    campaign_results: Dict[str, Dict[int, Dict]],
    campaign_configs: Optional[Dict[str, Dict]] = None,
    output_dir: Path = Path("campaign_plots"),
    formats: List[str] = ["png", "svg"]
) -> None
```

---

## Import Statement

```python
from spinq_qv.analysis.plots import (
    plot_campaign_comparison,
    plot_achieved_qv_comparison,
    plot_qv_waterfall,
    plot_critical_transitions,
    plot_parameter_correlation_matrix,
    plot_campaign_dashboard,
    plot_3d_parameter_surface,
    save_all_campaign_plots,
)
```

---

## Visual Guide

### When to Use Each Plot

**Want to compare overall performance?**
→ `plot_campaign_comparison()` (overlaid curves)

**Need a single summary metric?**
→ `plot_achieved_qv_comparison()` (QV bars)

**Too many configs cluttering the plot?**
→ `plot_qv_waterfall()` (stacked view)

**Trying to optimize parameters?**
→ `plot_parameter_correlation_matrix()` (heatmap)

**Need to understand failure points?**
→ `plot_critical_transitions()` (margin analysis)

**Want everything in one figure?**
→ `plot_campaign_dashboard()` (4-panel)

**Exploring 2D parameter space?**
→ `plot_3d_parameter_surface()` (3D plot)

**Need all plots for a report?**
→ `save_all_campaign_plots()` (batch)

---

## Common Customizations

### Custom Colors
```python
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']
plot_campaign_comparison(campaign_results, colors=colors, ...)
```

### Focused Parameters
```python
plot_parameter_correlation_matrix(
    campaign_results,
    campaign_configs,
    params_to_analyze=['F1', 'F2', 'T2']
)
```

### Multiple Formats
```python
save_all_campaign_plots(
    campaign_results,
    output_dir="plots",
    formats=["png", "svg", "pdf"]
)
```

### Custom View Angle (3D)
```python
plot_3d_parameter_surface(
    ...,
    elevation=30,
    azimuth=45
)
```

---

## Troubleshooting Checklist

- [ ] Campaign results in correct format? (nested dicts)
- [ ] All widths present in results? (at least some overlap)
- [ ] Config names are strings? (not integers)
- [ ] Device parameters in configs? (for correlation)
- [ ] Output directory exists? (will be created if not)
- [ ] matplotlib installed? (required dependency)

---

## Performance Notes

- **Memory:** Minimal (only aggregated stats)
- **Speed:** ~0.5 sec per plot
- **Scalability:** 2-10 configs recommended
- **File Size:** PNG ~500 KB, SVG ~200 KB

---

## Examples

### Run Demo
```bash
cd examples
python demo_campaign_plots.py
```

### View Generated Plots
```
examples/demo_campaign_output/
├── demo_comparison.png
├── demo_dashboard.png
├── demo_correlation.png
└── ...
```

---

## Documentation Links

- **Full Guide:** `docs/campaign_plotting_guide.md`
- **Implementation:** `docs/campaign_plots_implementation_summary.md`
- **Demo Script:** `examples/demo_campaign_plots.py`
- **Source Code:** `src/spinq_qv/analysis/plots.py`

---

## Contact/Issues

For questions or issues, check:
1. Full guide: `docs/campaign_plotting_guide.md`
2. Demo: `examples/demo_campaign_plots.py`
3. Source: `src/spinq_qv/analysis/plots.py`

---

**Last Updated:** October 2025
