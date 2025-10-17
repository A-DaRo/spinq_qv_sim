# Examples Directory - Campaign System

## Quick Start

### 1. Test the System

```bash
# Quick demo (no execution)
python test_campaign_system.py
```

### 2. Run a Small Test Campaign

```bash
# Dry run first (generates configs, no simulation)
python run_parameter_campaign.py \
    --base-config configs/test_small.yaml \
    --sweep-type fidelity_focus \
    --n-points 3 \
    --dry-run \
    --output ../campaigns/test

# If satisfied, run without --dry-run
python run_parameter_campaign.py \
    --base-config configs/test_small.yaml \
    --sweep-type fidelity_focus \
    --n-points 3 \
    --output ../campaigns/test
```

### 3. Run Production Campaign

```bash
python run_parameter_campaign.py \
    --base-config configs/production.yaml \
    --sweep-type comprehensive \
    --n-points 5 \
    --output ../campaigns/production_$(date +%Y%m%d)
```

---

## File Overview

### Campaign System (Main Scripts)

| File | Purpose |
|------|---------|
| `run_parameter_campaign.py` | **Main entry point** - Run full campaign |
| `campaign_config_generator.py` | Generate parameter sweep configurations |
| `campaign_executor.py` | Execute QV simulations for all configs |
| `campaign_plotter.py` | **Generate width-grouped plots** |
| `campaign_analyzer.py` | Analyze results and generate plots |
| `test_campaign_system.py` | Quick test/demo of system |

### Plotting Demos

| File | Purpose |
|------|---------|
| `demo_campaign_plots.py` | Demo of campaign-level plotting functions |

---

## Campaign Types

### Comprehensive
- **Parameters**: F1, F2, T1, T2, t_single_gate, t_two_gate
- **Use**: Full parameter space exploration
- **Runtime**: 6-12 hours (production config)

### Fidelity Focus
- **Parameters**: F1, F2, F_readout, F_init (fine-grained)
- **Use**: Gate calibration optimization
- **Runtime**: 4-8 hours (production config)

### Coherence Focus
- **Parameters**: T1, T2, T2* (wide range, log scale)
- **Use**: Decoherence studies, materials comparison
- **Runtime**: 4-8 hours (production config)

### Timing Focus
- **Parameters**: t_single_gate, t_two_gate, t_readout
- **Use**: Speed vs accuracy tradeoffs
- **Runtime**: 3-6 hours (production config)

---

## Command-Line Options

```
python run_parameter_campaign.py [OPTIONS]

Required:
  --base-config PATH    Base configuration file

Optional:
  --output DIR          Output directory (default: campaigns/param_sweep_TIMESTAMP)
  --sweep-type TYPE     comprehensive|fidelity_focus|coherence_focus|timing_focus
  --n-points N          Number of values per parameter (default: 5)
  --dry-run             Generate configs without running
  --parallel            Enable parallel execution (experimental)
```

---

## Output Structure

```
campaigns/my_campaign/
├── configs/                  # Generated YAML configs
├── results/                  # JSON results per config
├── plots/                    # All visualizations
│   ├── by_width/            # Per-width HOP & QV comparisons ← NEW
│   │   ├── hop_width_2.png
│   │   ├── hop_width_3.png
│   │   ├── qv_width_2.png
│   │   └── ...
│   ├── global/              # Global comparison plots ← NEW
│   │   ├── global_width_average.png
│   │   ├── hop_trajectories.png
│   │   └── qv_heatmap.png
│   ├── overview/            # Campaign-level plots
│   └── param_*/             # Parameter-specific plots
├── analysis/                 # Sensitivity analysis
├── campaign_manifest.json    # Campaign metadata
├── campaign_results.json     # Aggregated results
└── campaign_report.html      # Interactive report (OPEN THIS!)
```

---

## Key Outputs

### 1. campaign_report.html
**Interactive HTML dashboard** with all plots and analysis.  
Open in browser after campaign completes.

### 2. Width-Grouped Plots (NEW)
**Per-Width Analysis**: `plots/by_width/hop_width_{m}.png`  
Shows HOP comparison across all configs at specific width.

**Global Comparisons**: `plots/global/`
- `global_width_average.png` - Overall performance summary
- `hop_trajectories.png` - HOP degradation patterns
- `qv_heatmap.png` - Pass/fail matrix overview

### 3. plots/overview/campaign_comparison.png
**Main result**: All configurations overlaid showing HOP vs width.

### 4. analysis/parameter_sensitivity.json
**Quantitative impact**: Which parameters correlate strongest with QV.

---

## Typical Workflow

```bash
# 1. Create/modify base config
vim configs/my_experiment.yaml

# 2. Test with dry run
python run_parameter_campaign.py \
    --base-config configs/my_experiment.yaml \
    --sweep-type fidelity_focus \
    --n-points 5 \
    --dry-run \
    --output campaigns/my_run

# 3. Review generated configs
ls campaigns/my_run/configs/

# 4. Run campaign
python run_parameter_campaign.py \
    --base-config configs/my_experiment.yaml \
    --sweep-type fidelity_focus \
    --n-points 5 \
    --output campaigns/my_run

# 5. Open report
# Windows: start campaigns/my_run/campaign_report.html
# Mac: open campaigns/my_run/campaign_report.html
# Linux: xdg-open campaigns/my_run/campaign_report.html
```

---

## Performance Tips

### For Quick Testing
```yaml
# Use test_small.yaml or create custom config:
simulation:
  n_circuits: 10      # Instead of 100
  n_shots: 500        # Instead of 5000
  widths: [2, 3, 4]   # Instead of [2..12]
```

### For Production
```yaml
# Use production.yaml settings:
simulation:
  n_circuits: 100
  n_shots: 5000
  widths: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

---

## Examples

### Example 1: Quick Fidelity Sweep
```bash
python run_parameter_campaign.py \
    --base-config configs/baseline.yaml \
    --sweep-type fidelity_focus \
    --n-points 7 \
    --output campaigns/fidelity_sweep
```

### Example 2: Coherence Study
```bash
python run_parameter_campaign.py \
    --base-config configs/production.yaml \
    --sweep-type coherence_focus \
    --n-points 10 \
    --output campaigns/coherence_study
```

### Example 3: Full Parameter Space
```bash
python run_parameter_campaign.py \
    --base-config configs/production.yaml \
    --sweep-type comprehensive \
    --n-points 5 \
    --output campaigns/full_sweep_$(date +%Y%m%d)
```

---

## Troubleshooting

### Campaign too slow?
- Reduce `n_points` (5 → 3)
- Use smaller width range
- Reduce n_circuits and n_shots in base config

### Out of memory?
- Avoid widths > 10
- Use `statevector` backend (not `density_matrix`)
- Close other applications

### Some configs failed?
- Check `campaign_results.json` for error messages
- Look at individual result files in `results/`
- Verify parameter combinations are valid (e.g., T2 ≤ 2*T1)

---

## Documentation

- **Full Guide**: `../docs/campaign_system_guide.md`
- **Plotting Guide**: `../docs/campaign_plotting_guide.md`
- **Width-Grouped Plots**: `../docs/campaign_plots_by_width_guide.md` ← NEW
- **Quick Reference**: `../docs/campaign_plots_quick_reference.md`

---

## Support

For questions or issues:
1. Check documentation in `docs/`
2. Run `python test_campaign_system.py` to verify setup
3. Try with `--dry-run` first to preview campaign

---

**Last Updated**: October 2025
