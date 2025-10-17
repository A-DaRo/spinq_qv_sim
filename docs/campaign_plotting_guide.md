# Campaign-Level Plotting Guide

## Overview

This guide covers the advanced campaign-level plotting functions added to `spinq_qv.analysis.plots`. These functions are designed to compare and analyze results across **multiple configurations** in a single Quantum Volume campaign.

### What's a "Campaign"?

A **campaign** refers to running QV experiments with multiple device configurations (e.g., different gate fidelities, coherence times, or gate durations) to understand how parameter variations affect QV performance.

**Key questions campaigns answer:**
- Which configuration achieves the highest QV?
- How do specific parameters (F1, F2, T1, T2) correlate with QV success?
- Where does each configuration start to fail?
- What's the stability margin at critical widths?

---

## New Plotting Functions

### 1. `plot_campaign_comparison()`

**Purpose:** Overlay HOP curves from multiple configurations on a single plot for direct comparison.

**Key Features:**
- Overlaid mean HOP lines with distinct colors
- Optional confidence intervals (semi-transparent bands)
- Stars mark passing widths
- Clear threshold line at 2/3

**When to use:** This should be your **primary campaign visualization**. It directly shows which configurations perform better across all widths.

**Example:**
```python
from spinq_qv.analysis.plots import plot_campaign_comparison

campaign_results = {
    "baseline": {2: {...}, 3: {...}, ...},
    "high_fidelity": {2: {...}, 3: {...}, ...},
    "long_coherence": {2: {...}, 3: {...}, ...},
}

fig = plot_campaign_comparison(
    campaign_results,
    output_path="comparison.png",
    show_ci=True,
    title="Device Configuration Comparison"
)
```

**Visual Insight:** Immediately see which configurations achieve higher HOP and at which width they diverge.

---

### 2. `plot_achieved_qv_comparison()`

**Purpose:** Bar chart showing maximum achieved QV (2^m) for each configuration.

**Key Features:**
- Horizontal bars sorted by QV (best at top)
- Logarithmic scale (base-2) for QV values
- Annotated with QV value and passing width
- Clear visual hierarchy

**When to use:** When you need a **summary metric** – "What's the bottom line for each config?"

**Example:**
```python
fig = plot_achieved_qv_comparison(
    campaign_results,
    output_path="achieved_qv.png",
    show_values=True
)
```

**Visual Insight:** Instantly identify the winning configuration and quantify the gap between them.

---

### 3. `plot_qv_waterfall()`

**Purpose:** Cascading view of HOP degradation with vertical offsets for each configuration.

**Key Features:**
- Stacked curves with offset for clarity
- Gold stars mark critical widths (last passing)
- Color-coded by configuration
- Configuration labels on left side

**When to use:** When you have many configurations and want to show the **degradation pattern** without overlapping lines.

**Example:**
```python
fig = plot_qv_waterfall(
    campaign_results,
    output_path="waterfall.png",
    threshold=2/3
)
```

**Visual Insight:** See the "cascade" of failures as width increases, making it easy to spot configurations that maintain performance longer.

---

### 4. `plot_critical_transitions()`

**Purpose:** Two-panel plot analyzing the transition from passing to failing.

**Key Features:**
- **Upper panel:** Full HOP curves with diamond markers at critical width
- **Lower panel:** Scatter plot of margin-to-threshold vs critical width
- Identifies stability margin (how much cushion above threshold)

**When to use:** When optimizing near the threshold – "How close are we to failure?"

**Example:**
```python
fig = plot_critical_transitions(
    campaign_results,
    output_path="critical_transitions.png"
)
```

**Visual Insight:** Reveals which configurations are "barely passing" (small margin) vs "comfortably passing" (large margin).

---

### 5. `plot_parameter_correlation_matrix()`

**Purpose:** Heatmap showing correlations between device parameters and achieved QV.

**Key Features:**
- Normalized correlation coefficients (-1 to +1)
- Color-coded (blue = positive, red = negative)
- Annotated cells with numerical values
- Includes all device parameters vs maximum width

**When to use:** For **parameter sensitivity analysis** – "Which knobs have the biggest impact?"

**Example:**
```python
campaign_configs = {
    "baseline": {"device": {"F1": 0.999, "F2": 0.998, ...}},
    "high_fidelity": {"device": {"F1": 0.9999, "F2": 0.9995, ...}},
    ...
}

fig = plot_parameter_correlation_matrix(
    campaign_results,
    campaign_configs,
    output_path="correlation.png"
)
```

**Visual Insight:** Strong positive correlations (dark blue) indicate parameters that should be prioritized for improvement.

---

### 6. `plot_campaign_dashboard()`

**Purpose:** Comprehensive 4-panel overview combining multiple views.

**Panels:**
1. HOP comparison (all configs overlaid)
2. Achieved QV bar chart
3. Critical width analysis
4. Statistical summary table

**When to use:** For **presentations and reports** – gives a complete picture in one figure.

**Example:**
```python
fig = plot_campaign_dashboard(
    campaign_results,
    campaign_configs=campaign_configs,
    output_path="dashboard.png"
)
```

**Visual Insight:** Single-page summary suitable for papers or project reviews.

---

### 7. `plot_3d_parameter_surface()`

**Purpose:** 3D surface showing QV as a function of two parameters.

**Key Features:**
- Interactive 3D surface plot
- Contour projection on base plane
- Adjustable viewing angle
- Colormap indicates QV magnitude

**When to use:** For **exploring parameter space** and understanding interaction effects between two parameters.

**Example:**
```python
import numpy as np

F1_values = np.linspace(0.995, 0.9999, 20)
F2_values = np.linspace(0.990, 0.9995, 20)
qv_grid = compute_qv_grid(F1_values, F2_values)  # Your grid data

fig = plot_3d_parameter_surface(
    F1_values,
    F2_values,
    qv_grid,
    param1_name="F1",
    param2_name="F2",
    output_path="3d_surface.png"
)
```

**Visual Insight:** Reveals optimal parameter regions and non-linear interactions.

---

### 8. `save_all_campaign_plots()` (Batch Function)

**Purpose:** Generate all campaign plots at once in multiple formats.

**Features:**
- Automatically creates all plot types
- Multiple output formats (PNG, SVG, PDF)
- Progress reporting
- Organized output directory

**When to use:** At the end of a large campaign for **comprehensive reporting**.

**Example:**
```python
from spinq_qv.analysis.plots import save_all_campaign_plots

save_all_campaign_plots(
    campaign_results,
    campaign_configs=campaign_configs,
    output_dir="campaign_final_plots",
    formats=["png", "svg"]
)
```

Output structure:
```
campaign_final_plots/
├── campaign_comparison.png
├── campaign_comparison.svg
├── achieved_qv_comparison.png
├── achieved_qv_comparison.svg
├── qv_waterfall.png
├── critical_transitions.png
├── campaign_dashboard.png
├── parameter_correlation.png
└── ...
```

---

## Data Structure Requirements

### Campaign Results Format

```python
campaign_results = {
    "config_name1": {
        width1: {
            "mean_hop": float,
            "ci_lower": float,
            "ci_upper": float,
            "n_circuits": int,
            "pass_qv": bool,
            "hops": np.ndarray,  # optional
        },
        width2: {...},
        ...
    },
    "config_name2": {...},
    ...
}
```

### Campaign Configs Format (for correlation analysis)

```python
campaign_configs = {
    "config_name1": {
        "device": {
            "F1": float,
            "F2": float,
            "T1": float,
            "T2": float,
            "T2_star": float,
            "t_single_gate": float,
            "t_two_gate": float,
            "F_readout": float,
            "F_init": float,
        },
        "simulation": {...},  # optional
        "metadata": {...},    # optional
    },
    "config_name2": {...},
    ...
}
```

---

## Typical Workflow

### Step 1: Run Multiple Configurations

```python
from pathlib import Path
from spinq_qv.experiments.run_qv import run_qv_experiment
from spinq_qv.config.schemas import Config

configs = [
    "configs/baseline.yaml",
    "configs/high_fidelity.yaml",
    "configs/long_coherence.yaml",
]

campaign_results = {}
campaign_configs = {}

for config_file in configs:
    config = Config.from_yaml(config_file)
    config_name = config.metadata.get("experiment_name", Path(config_file).stem)
    
    # Run QV experiment
    results = run_qv_experiment(config, output_dir=f"results/{config_name}")
    
    # Store results
    campaign_results[config_name] = results["aggregated"]
    campaign_configs[config_name] = config.model_dump()
```

### Step 2: Generate Campaign Plots

```python
from spinq_qv.analysis.plots import save_all_campaign_plots

save_all_campaign_plots(
    campaign_results,
    campaign_configs=campaign_configs,
    output_dir="campaign_analysis",
    formats=["png", "svg"]
)
```

### Step 3: Focus on Key Insights

```python
# Compare configurations directly
plot_campaign_comparison(
    campaign_results,
    output_path="key_comparison.png",
    title="Si/SiGe Device Comparison: Effect of Gate Fidelities"
)

# Identify critical parameters
plot_parameter_correlation_matrix(
    campaign_results,
    campaign_configs,
    output_path="parameter_importance.png"
)
```

---

## Design Philosophy

### Why These Plots?

1. **Multi-config comparison is non-trivial**: Standard QV plots show one config at a time. Campaign plots enable side-by-side comparison.

2. **Parameter correlation is key**: QV depends on many parameters. Correlation analysis identifies optimization priorities.

3. **Critical transitions matter**: Knowing *where* a config fails is as important as knowing *if* it fails.

4. **Quantum Volume is logarithmic**: QV = 2^m means a small width increase represents exponential improvement. Plots emphasize this.

5. **Statistical rigor**: All plots show confidence intervals and pass/fail status based on rigorous criteria.

### Visual Design Principles

- **Colorblind-friendly palettes** (default colors chosen for accessibility)
- **Clear annotations** (stars, labels, values on plots)
- **Publication quality** (300 DPI, vector formats, serif fonts)
- **Information density** (maximize insight per plot)
- **Consistent styling** (all plots follow same aesthetic)

---

## Advanced Usage

### Custom Color Schemes

```python
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']

plot_campaign_comparison(
    campaign_results,
    colors=colors,
    output_path="custom_colors.png"
)
```

### Focused Parameter Analysis

```python
# Only analyze specific parameters
plot_parameter_correlation_matrix(
    campaign_results,
    campaign_configs,
    params_to_analyze=['F1', 'F2', 'T2'],
    output_path="fidelity_focus.png"
)
```

### 3D Surface from Sensitivity Grid

```python
# If you have a grid sweep
sensitivity_data = run_parameter_grid_sweep(
    param1_range=np.linspace(0.995, 0.9999, 15),
    param2_range=np.linspace(0.990, 0.9995, 15),
    param1_name='F1',
    param2_name='F2'
)

plot_3d_parameter_surface(
    sensitivity_data['param1_values'],
    sensitivity_data['param2_values'],
    sensitivity_data['qv_grid'],
    param1_name='F1',
    param2_name='F2',
    output_path="sensitivity_3d.png",
    elevation=20,
    azimuth=30
)
```

---

## Integration with Existing Tools

### With Error Budgets

```python
from spinq_qv.analysis.ablation import compute_error_budget

# Compute error budget for each config
budgets = {}
for config_name, results in campaign_results.items():
    budgets[config_name] = compute_error_budget(...)

# Compare budgets side-by-side
plot_error_budget_comparison(
    budgets,
    output_path="budget_comparison.png"
)
```

### With Sensitivity Analysis

```python
from spinq_qv.experiments.sensitivity import run_sensitivity_sweep

# Run sweep for multiple configs
sweep_results = {}
for config_name in campaign_configs.keys():
    sweep_results[config_name] = run_sensitivity_sweep(
        config=campaign_configs[config_name],
        param_name='F1',
        param_range=np.linspace(0.995, 0.9999, 20)
    )

# Plot combined sweep
plot_multi_config_sweep(sweep_results, output_path="sweep_comparison.png")
```

---

## Best Practices

### 1. Configuration Naming

Use descriptive names:
- ✅ `baseline_F1_0p999_F2_0p998`
- ✅ `high_fidelity_2xT2`
- ✅ `fast_gates_30ns`
- ❌ `config1`, `test2`, `run_final_v3`

### 2. Consistent Widths

Test the same widths across all configurations for fair comparison:
```python
widths = [2, 3, 4, 5, 6, 7, 8]  # Use for all configs
```

### 3. Statistical Power

Use enough circuits for tight confidence intervals:
```python
n_circuits = 50   # Minimum (quick tests)
n_circuits = 100  # Recommended (IBM standard)
n_circuits = 200  # High precision
```

### 4. Documentation

Always save configs alongside results:
```python
import yaml

for config_name, config_dict in campaign_configs.items():
    with open(f"results/{config_name}_config.yaml", 'w') as f:
        yaml.dump(config_dict, f)
```

### 5. Batch Processing

For large campaigns (>10 configs), use parallel execution:
```python
from multiprocessing import Pool

def run_single_config(config_file):
    # Run QV experiment
    return config_name, results

with Pool(8) as pool:
    results = pool.map(run_single_config, config_files)
```

---

## Troubleshooting

### Issue: "No valid parameters found for correlation analysis"

**Cause:** Config dictionaries don't contain device parameters.

**Solution:** Ensure campaign_configs includes full device parameters:
```python
campaign_configs[name] = config.model_dump()  # Use Pydantic's dump method
```

### Issue: Plots look cluttered with many configurations

**Cause:** Too many configs (>8) on one plot.

**Solution:** Create focused subsets:
```python
# Group 1: Fidelity sweep
fidelity_group = {k: v for k, v in campaign_results.items() 
                  if 'fidelity' in k.lower()}
plot_campaign_comparison(fidelity_group, ...)

# Group 2: Coherence sweep
coherence_group = {k: v for k, v in campaign_results.items() 
                   if 'coherence' in k.lower() or 'T2' in k}
plot_campaign_comparison(coherence_group, ...)
```

### Issue: 3D plot looks flat

**Cause:** Insufficient parameter variation or narrow range.

**Solution:** Expand parameter ranges:
```python
# Too narrow
F1_values = np.linspace(0.999, 0.9995, 10)  # ❌

# Better
F1_values = np.linspace(0.995, 0.9999, 20)  # ✅
```

---

## Performance Notes

- **Memory:** Campaign plots use minimal memory (only aggregated results, not raw samples)
- **Speed:** All plots generate in <1 second on modern hardware
- **Disk:** PNG ~500 KB, SVG ~200 KB per plot
- **Parallel:** Plots can be generated in parallel for large batches

---

## Examples Gallery

See `examples/demo_campaign_plots.py` for a complete working example with mock data.

Run the demo:
```bash
cd examples
python demo_campaign_plots.py
```

This generates all plot types with annotated explanations.

---

## Citation

If you use these campaign plotting tools in a publication, please cite:

```
spinq_qv_sim: Quantum Volume Simulator for Si/SiGe Spin Qubits
Campaign-level analysis and visualization tools
https://github.com/yourusername/spinq_qv_sim
```

---

## Further Reading

- **IBM QV Paper:** [arXiv:1811.12926](https://arxiv.org/abs/1811.12926)
- **Statistical Methods:** `docs/math_foundations.md`
- **Noise Models:** `Technicalities.md`
- **Config Reference:** `src/spinq_qv/config/schemas.py`

---

**Last Updated:** October 2025  
**Author:** spinq_qv_sim development team
