# Parameter Campaign System - User Guide

## Overview

The **Parameter Campaign System** enables systematic exploration of how device parameters affect Quantum Volume performance. It automates:

1. **Configuration Generation**: Create parameter sweeps automatically
2. **Campaign Execution**: Run QV experiments for all configurations
3. **Analysis & Visualization**: Generate comprehensive reports and plots
4. **Sensitivity Analysis**: Identify which parameters matter most

---

## Quick Start

### Step 1: Test the System (Dry Run)

```bash
cd examples

# Generate configs and preview campaign (no execution)
python run_parameter_campaign.py \
    --base-config configs/production.yaml \
    --sweep-type fidelity_focus \
    --n-points 5 \
    --dry-run \
    --output campaigns/test_run
```

This creates configuration files and a manifest without running simulations.

### Step 2: Run Small Test Campaign

```bash
# Run with reduced settings for testing
python run_parameter_campaign.py \
    --base-config configs/test_small.yaml \
    --sweep-type fidelity_focus \
    --n-points 3 \
    --output campaigns/small_test
```

### Step 3: Run Full Production Campaign

```bash
# Full campaign (will take several hours)
python run_parameter_campaign.py \
    --base-config configs/production.yaml \
    --sweep-type comprehensive \
    --n-points 7 \
    --output campaigns/production_run
```

---

## Campaign Types

### 1. Comprehensive Sweep

Sweeps **all major parameters** with balanced coverage:

- **F1, F2**: Gate fidelities (±0.5% around baseline)
- **T1, T2**: Coherence times (0.5× to 2× baseline, log scale)
- **t_single_gate, t_two_gate**: Gate durations (0.5× to 1.5×)

**Use when**: You want full parameter space coverage

**Runtime**: ~6-12 hours with `n_points=5`, production config

```bash
python run_parameter_campaign.py \
    --sweep-type comprehensive \
    --n-points 5
```

### 2. Fidelity Focus

Deep dive into **gate fidelities and SPAM errors**:

- **F1, F2**: Fine-grained sweeps (±0.1% to ±0.2%)
- **F_readout, F_init**: SPAM parameter sweeps
- More points per parameter (2× default)

**Use when**: Optimizing gate calibration, studying fidelity impact

**Runtime**: ~4-8 hours with `n_points=5`

```bash
python run_parameter_campaign.py \
    --sweep-type fidelity_focus \
    --n-points 7
```

### 3. Coherence Focus

Wide-range exploration of **decoherence parameters**:

- **T1**: Amplitude damping (0.2× to 5× baseline)
- **T2**: Phase damping (0.2× to 5× baseline)
- **T2***: Inhomogeneous dephasing sweep
- Logarithmic spacing for time scales

**Use when**: Studying coherence requirements, materials comparison

**Runtime**: ~4-8 hours with `n_points=5`

```bash
python run_parameter_campaign.py \
    --sweep-type coherence_focus \
    --n-points 7
```

### 4. Timing Focus

Analyze **gate speed vs fidelity tradeoffs**:

- **t_single_gate**: Single-qubit gate duration (0.3× to 2×)
- **t_two_gate**: Two-qubit gate duration (0.3× to 2×)
- **t_readout**: Measurement time sweep

**Use when**: Optimizing gate sequences, studying speed limits

**Runtime**: ~3-6 hours with `n_points=5`

```bash
python run_parameter_campaign.py \
    --sweep-type timing_focus \
    --n-points 7
```

---

## Output Structure

After running a campaign, the output directory contains:

```
campaigns/my_campaign/
├── configs/                      # Individual YAML configs
│   ├── baseline.yaml
│   ├── F1_0p99400.yaml
│   ├── F2_0p99500.yaml
│   └── ...
│
├── results/                      # Raw QV results (JSON)
│   ├── baseline_results.json
│   ├── F1_0p99400_results.json
│   └── ...
│
├── plots/                        # All visualizations
│   ├── overview/                 # Campaign-level plots
│   │   ├── campaign_comparison.png
│   │   ├── campaign_dashboard.png
│   │   ├── achieved_qv_comparison.png
│   │   └── ...
│   │
│   ├── param_F1/                 # Parameter-specific plots
│   │   ├── F1_vs_hop.png
│   │   └── F1_vs_qv.png
│   │
│   ├── param_F2/
│   └── ...
│
├── analysis/                     # Quantitative analysis
│   ├── parameter_sensitivity.json
│   ├── F1_sweep_summary.json
│   ├── F2_sweep_summary.json
│   └── ...
│
├── campaign_manifest.json        # Campaign metadata
├── campaign_results.json         # Aggregated results
└── campaign_report.html          # Interactive HTML report
```

---

## Key Output Files

### 1. Campaign Report (`campaign_report.html`)

**Interactive HTML dashboard** with:
- Overview metrics (total configs, max QV achieved)
- Embedded plots (all campaign visualizations)
- Parameter-specific analysis sections
- Key findings summary
- Sensitivity rankings

**Open in browser**: `file:///path/to/campaigns/my_campaign/campaign_report.html`

### 2. Campaign Manifest (`campaign_manifest.json`)

**Complete campaign specification**:
```json
{
  "campaign_name": "my_campaign",
  "timestamp": "2025-10-17T12:00:00",
  "base_config": "configs/production.yaml",
  "n_configurations": 42,
  "swept_parameters": {
    "F1": {
      "min": 0.994,
      "max": 0.9999,
      "n_values": 7,
      "values": [0.994, 0.995, ...]
    },
    ...
  },
  "configurations": {
    "baseline": {...},
    "F1_0p99400": {...}
  }
}
```

### 3. Sensitivity Analysis (`analysis/parameter_sensitivity.json`)

**Quantitative parameter impact**:
```json
{
  "F1": {
    "correlation": 0.85,
    "qv_range": 3,
    "max_qv": 64,
    "min_qv": 8
  },
  "F2": {
    "correlation": 0.72,
    ...
  }
}
```

- **correlation**: How strongly parameter correlates with QV (-1 to +1)
- **qv_range**: Width difference (max - min passing)
- **max_qv, min_qv**: Range of achieved QV values

---

## Understanding the Plots

### Campaign Comparison Plot

![Example](plots/overview/campaign_comparison.png)

**Shows**: All configurations overlaid with HOP vs width
**Interpret**:
- Higher curves = better performance
- Stars = passing widths (QV achieved)
- Bands = confidence intervals
- Divergence point = where configs start to differ

### HOP vs Parameter

![Example](plots/param_F1/F1_vs_hop.png)

**Shows**: How HOP changes with parameter value (one line per width)
**Interpret**:
- Upward slope = parameter improvement helps QV
- Downward slope = parameter increase hurts QV
- Flat line = parameter has minimal impact
- Lines crossing threshold = QV achievement

### QV vs Parameter

![Example](plots/param_F1/F1_vs_qv.png)

**Shows**: Maximum achieved QV as function of parameter
**Interpret**:
- Steep sections = sensitive regions
- Plateaus = saturation or limited by other factors
- Logarithmic y-axis emphasizes exponential QV metric

### Parameter Correlation Matrix

![Example](plots/parameter_correlations.png)

**Shows**: Correlation between all parameters and max width
**Interpret**:
- Dark blue = strong positive correlation (increase → higher QV)
- Dark red = strong negative correlation (increase → lower QV)
- White = no correlation
- Bottom row = most important (correlations with QV)

---

## Customization

### Custom Parameter Ranges

Edit `campaign_config_generator.py`:

```python
def _get_custom_sweep(base_config, n_points):
    return {
        "F1": np.linspace(0.995, 0.9999, 10),  # Custom range
        "T2": np.logspace(-5, -3, 8),          # Log scale
    }
```

### Custom Sweep Type

Add to `run_parameter_campaign.py`:

```python
elif args.sweep_type == "my_custom_sweep":
    sweep_params = _get_custom_sweep(base_config, n_points)
```

### Reduced Simulation Settings

For faster testing, modify base config:

```yaml
simulation:
  n_circuits: 10     # Fewer circuits (default: 100)
  n_shots: 500       # Fewer shots (default: 5000)
  widths: [2, 3, 4]  # Smaller widths only
```

---

## Interpreting Results

### Question: Which parameter matters most?

**Answer**: Check `analysis/parameter_sensitivity.json`

```json
{
  "F2": {"correlation": 0.89, "qv_range": 4},  ← Strongest impact
  "F1": {"correlation": 0.76, "qv_range": 3},
  "T2": {"correlation": 0.42, "qv_range": 1},  ← Weaker impact
}
```

**Interpretation**: Prioritize optimizing F2, then F1. T2 has smaller effect.

### Question: What QV can I achieve with better fidelities?

**Answer**: Look at `plots/param_F1/F1_vs_qv.png` and `plots/param_F2/F2_vs_qv.png`

Find your target fidelity on x-axis → read QV from curve.

### Question: Is my device limited by coherence or gates?

**Answer**: Compare correlation magnitudes:

- If fidelity correlations >> coherence correlations → **gate-limited**
- If coherence correlations >> fidelity correlations → **coherence-limited**
- If similar → **balanced limitation**

### Question: Where should I focus optimization efforts?

**Answer**: 
1. Check correlation matrix → highest correlations
2. Look at parameter vs QV plots → steep regions (high sensitivity)
3. Consider feasibility → which parameters you can improve experimentally

---

## Performance Tips

### 1. Start Small

Always test with:
```bash
--dry-run  # Generate configs, preview campaign
--n-points 3  # Fewer parameter values
--base-config test_small.yaml  # Reduced sim settings
```

### 2. Parallel Execution (Experimental)

```bash
python run_parameter_campaign.py \
    --parallel \
    ...
```

**Note**: Requires `multiprocessing` support, may have limited benefit due to NumPy threading.

### 3. Incremental Analysis

Campaign saves results after each config. If interrupted:
- Results are preserved in `results/`
- Re-run analyzer separately:

```python
from campaign_analyzer import CampaignAnalyzer

analyzer = CampaignAnalyzer(
    campaign_results=...,
    campaign_configs=...,
    sweep_params=...,
    output_dir="campaigns/my_campaign"
)
analyzer.generate_all_analyses()
```

### 4. Resource Estimates

| Config Type | n_points | Configs | Est. Time (8-core) |
|-------------|----------|---------|-------------------|
| Fidelity Focus | 3 | ~15 | 1-2 hours |
| Fidelity Focus | 5 | ~25 | 3-5 hours |
| Fidelity Focus | 7 | ~35 | 5-8 hours |
| Comprehensive | 5 | ~35 | 6-12 hours |
| Comprehensive | 7 | ~50 | 10-18 hours |

*Based on production config (widths 2-12, 100 circuits, 5000 shots)*

---

## Troubleshooting

### Issue: "Campaign taking too long"

**Solutions**:
1. Reduce `n_points` (e.g., 5 → 3)
2. Use smaller width range in base config
3. Reduce `n_circuits` or `n_shots`
4. Use `--sweep-type fidelity_focus` (fewer parameters)

### Issue: "Out of memory"

**Solutions**:
1. Reduce width range (avoid m > 10)
2. Use `statevector` backend (not `density_matrix`)
3. Run fewer configs at once
4. Close other applications

### Issue: "Some configs failed"

**Check**:
- `campaign_results.json` shows `n_failed`
- Individual result files in `results/`
- Error messages in terminal output

**Common causes**:
- Invalid parameter combinations (e.g., T2 > 2*T1)
- Numerical instabilities
- Resource limits

### Issue: "Plots missing for some parameters"

**Cause**: Not enough variation in parameter sweep

**Solution**: Increase `n_points` or widen parameter range in `campaign_config_generator.py`

---

## Advanced Usage

### Combining Multiple Campaigns

```python
# Load results from multiple campaigns
campaign1_results = load_campaign("campaigns/run1")
campaign2_results = load_campaign("campaigns/run2")

# Merge
all_results = {**campaign1_results, **campaign2_results}

# Analyze combined
from campaign_analyzer import CampaignAnalyzer
analyzer = CampaignAnalyzer(all_results, ...)
analyzer.generate_all_analyses()
```

### Custom Analysis

```python
import json

# Load results
with open("campaigns/my_campaign/campaign_results.json") as f:
    data = json.load(f)

# Custom processing
for config_name, results in data["results"].items():
    for width, stats in results.items():
        hop = stats["mean_hop"]
        # Your analysis here
```

### Export to Other Formats

```python
import pandas as pd
import json

# Load campaign results
with open("campaign_results.json") as f:
    data = json.load(f)

# Convert to DataFrame
rows = []
for config, results in data["results"].items():
    for width, stats in results.items():
        rows.append({
            "config": config,
            "width": width,
            "mean_hop": stats["mean_hop"],
            "pass_qv": stats["pass_qv"],
        })

df = pd.DataFrame(rows)
df.to_csv("campaign_results.csv", index=False)
```

---

## Best Practices

### 1. Document Your Campaign

Add to base config metadata:
```yaml
metadata:
  experiment_name: "production_fidelity_sweep"
  description: "Exploring F1/F2 impact on QV with Si/SiGe parameters"
  experimenter: "Your Name"
  date: "2025-10-17"
  notes: |
    Goal: Determine minimum F1/F2 for QV=64
    Hypothesis: F2 is limiting factor
```

### 2. Version Control Configs

```bash
git add examples/configs/my_campaign_base.yaml
git commit -m "Add campaign config for fidelity sweep"
```

### 3. Archive Results

```bash
# Compress campaign results
tar -czf campaign_$(date +%Y%m%d).tar.gz campaigns/my_campaign/

# Or zip on Windows
powershell Compress-Archive campaigns/my_campaign campaign_20251017.zip
```

### 4. Share Results

The HTML report is self-contained and can be shared:
```bash
# Email or upload campaign_report.html
# Recipients can view in any browser
```

---

## Examples

### Example 1: Finding Minimum Fidelities for QV=32

```bash
# Goal: What F1/F2 needed for QV=32?

# Step 1: Run fidelity sweep
python run_parameter_campaign.py \
    --sweep-type fidelity_focus \
    --n-points 10 \
    --output campaigns/qv32_fidelity

# Step 2: Check results
# Open: campaigns/qv32_fidelity/plots/param_F1/F1_vs_qv.png
# Find where curve crosses QV=32 line

# Step 3: Quantify
python -c "
import json
with open('campaigns/qv32_fidelity/analysis/F1_sweep_summary.json') as f:
    data = json.load(f)
    
configs_with_qv32 = [
    d for d in data if d['achieved_qv'] >= 32
]

min_f1 = min(c['param_value'] for c in configs_with_qv32)
print(f'Minimum F1 for QV=32: {min_f1:.6f}')
"
```

### Example 2: Coherence Requirements Study

```bash
# Goal: How much can T2 degrade before QV drops?

python run_parameter_campaign.py \
    --sweep-type coherence_focus \
    --n-points 12 \
    --output campaigns/coherence_study

# Check: plots/param_T2/T2_vs_qv.png
# Identify "cliff" where QV suddenly drops
```

### Example 3: Speed vs Accuracy Tradeoff

```bash
# Goal: Can faster gates maintain QV?

python run_parameter_campaign.py \
    --sweep-type timing_focus \
    --n-points 8 \
    --output campaigns/speed_tradeoff

# Analyze: Do faster gates hurt QV?
# Check correlation: negative → speed hurts, positive → speed helps
```

---

## FAQ

**Q: How long does a typical campaign take?**  
A: With production settings (widths 2-12, 100 circuits), 4-12 hours depending on sweep type and n_points.

**Q: Can I run multiple campaigns simultaneously?**  
A: Yes, use different `--output` directories. Be mindful of CPU/memory.

**Q: What if I need to stop mid-campaign?**  
A: Results are saved incrementally. Re-run analyzer on partial results, or restart campaign (already-completed configs will be skipped if you implement checkpointing).

**Q: Can I add my own parameter?**  
A: Yes! Edit `campaign_config_generator.py` to add new parameters to sweep ranges.

**Q: How do I compare to experimental data?**  
A: Load experimental QV results, add to `campaign_results` dict with key "experimental", run analyzer.

**Q: Can I use this with real devices?**  
A: Yes! Replace simulator calls in `campaign_executor.py` with your device interface.

---

## Citation

If you use this campaign system in research, please cite:

```
spinq_qv_sim: Quantum Volume Simulator for Si/SiGe Spin Qubits
Parameter Campaign System v1.0
https://github.com/yourusername/spinq_qv_sim
```

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Authors**: spinq_qv_sim development team
