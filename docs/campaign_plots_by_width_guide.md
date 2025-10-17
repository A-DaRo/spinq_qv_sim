# Campaign Plotting System - Quick Reference

## Overview

The campaign plotting system creates **width-grouped visualizations** to compare QV results across different device configurations. It generates two types of plots:

1. **Per-Width Plots** - Detailed comparisons at each specific circuit width
2. **Global Plots** - Overall performance summaries across all widths

---

## Generated Plots

### 📊 Per-Width Plots (`plots/by_width/`)

#### 1. HOP Comparison by Width
**File**: `hop_width_{m}.png`

- **X-axis**: Configuration name
- **Y-axis**: Heavy-Output Probability (HOP)
- **Features**:
  - Bar chart with error bars (95% CI)
  - Color coding: Blue = passed QV, Purple = failed QV
  - Red dashed line shows QV threshold (2/3)
  - Shows which configs pass/fail at specific width

**Use Case**: *"At width m=4, which device configurations achieve QV?"*

#### 2. Achieved QV by Width
**File**: `qv_width_{m}.png`

- **X-axis**: Configuration name
- **Y-axis**: Quantum Volume (2^m if passed, 0 if failed)
- **Features**:
  - Bar chart showing QV achievement
  - Log₂ scale for large QV values
  - Color coding: Blue = passed, Purple = failed

**Use Case**: *"What QV value does each configuration achieve at width m=5?"*

---

### 🌍 Global Plots (`plots/global/`)

#### 3. Global Width-Average Performance
**File**: `global_width_average.png`

Two-panel figure:

**Left Panel**: Average HOP across all widths
- Shows mean ± std dev of HOP values
- Indicates overall configuration robustness
- Threshold line for reference

**Right Panel**: Maximum width passed
- Shows highest width where QV criterion met
- Dual y-axis shows corresponding QV value
- Labels show QV=2^m on each bar

**Use Case**: *"Which configuration has best overall performance across all widths?"*

#### 4. HOP Trajectories Across Widths
**File**: `hop_trajectories.png`

- **X-axis**: Circuit width (m)
- **Y-axis**: Heavy-Output Probability
- **Features**:
  - Line plot for each configuration
  - Shaded confidence intervals
  - Shows degradation pattern as width increases
  - Color-coded by configuration

**Use Case**: *"How does HOP degrade as circuits get larger for different configs?"*

#### 5. QV Pass/Fail Heatmap
**File**: `qv_heatmap.png`

- **Rows**: Configurations
- **Columns**: Widths
- **Colors**:
  - Green (✓) = Passed QV
  - Red (✗) = Failed QV
  - Gray (—) = Not tested
- **Purpose**: Quick visual overview of pass/fail patterns

**Use Case**: *"At a glance, which configs pass at which widths?"*

---

## Plot Grouping Logic

### Width-Level Grouping

Each **per-width plot** shows:
- All configurations tested at that width
- Direct comparison of HOP values
- Clear pass/fail indication

**Example**: At `width = 3`:
```
F1_0.9990: HOP = 0.82 ± 0.03 → PASS ✓
F1_0.9985: HOP = 0.75 ± 0.04 → PASS ✓
F1_0.9980: HOP = 0.68 ± 0.05 → PASS ✓
F1_0.9975: HOP = 0.61 ± 0.06 → FAIL ✗
```

### Global Averaging

The **global average plot** computes:
```python
avg_hop = mean(HOP_width2, HOP_width3, HOP_width4, ...)
std_hop = std(HOP_width2, HOP_width3, HOP_width4, ...)
```

This gives a single performance metric per configuration.

---

## Integration with Campaign Runner

### Automatic Generation

When you run a campaign:

```bash
python run_parameter_campaign.py --sweep-type fidelity_focus
```

The system automatically:

1. **Executes** all QV simulations
2. **Generates width-grouped plots** (new feature)
3. **Generates campaign analysis plots** (existing feature)
4. **Creates HTML report**

### Output Structure

```
campaigns/my_campaign/
├── plots/
│   ├── by_width/              # ← NEW: Width-specific plots
│   │   ├── hop_width_2.png
│   │   ├── hop_width_3.png
│   │   ├── qv_width_2.png
│   │   ├── qv_width_3.png
│   │   └── ...
│   ├── global/                # ← NEW: Overall comparisons
│   │   ├── global_width_average.png
│   │   ├── hop_trajectories.png
│   │   └── qv_heatmap.png
│   └── overview/              # Existing campaign plots
│       ├── campaign_comparison.png
│       ├── dashboard.png
│       └── ...
```

---

## Interpreting the Plots

### Example: Fidelity Sweep Campaign

**Scenario**: Testing F1 = {0.999, 0.998, 0.997, 0.996, 0.995}

#### Per-Width Analysis (hop_width_3.png)

```
At width m=3:
- F1=0.999 → HOP=0.85 → PASS ✓
- F1=0.998 → HOP=0.78 → PASS ✓
- F1=0.997 → HOP=0.70 → PASS ✓
- F1=0.996 → HOP=0.65 → FAIL ✗
- F1=0.995 → HOP=0.58 → FAIL ✗

**Finding**: Need F1 ≥ 0.997 for QV=8 (2^3)
```

#### Global Analysis (global_width_average.png)

```
Average HOP across widths 2-5:
- F1=0.999 → Avg=0.75, Max_Width=4 → QV=16
- F1=0.998 → Avg=0.70, Max_Width=3 → QV=8
- F1=0.997 → Avg=0.65, Max_Width=3 → QV=8
- F1=0.996 → Avg=0.58, Max_Width=2 → QV=4
- F1=0.995 → Avg=0.52, Max_Width=2 → QV=4

**Finding**: F1=0.999 achieves best overall QV=16
```

#### Trajectory Analysis (hop_trajectories.png)

```
All configs show similar degradation slope
→ Indicates coherent error dominance (not stochastic)

Configs separate early (at low widths)
→ Fidelity difference is critical even for small circuits
```

---

## Key Questions Answered

### 1. Width-Specific Questions

**Q**: *"At width m=X, which configs pass QV?"*  
**A**: Check `hop_width_{X}.png` - blue bars show passing configs

**Q**: *"What's the minimum F1 needed for QV=8?"*  
**A**: Find width m=3, look at lowest F1 with blue bar

### 2. Global Questions

**Q**: *"Which config achieves highest QV overall?"*  
**A**: Check `global_width_average.png` (right panel) - tallest bar

**Q**: *"Which config is most robust across widths?"*  
**A**: Check `global_width_average.png` (left panel) - smallest error bars

**Q**: *"How does performance degrade with width?"*  
**A**: Check `hop_trajectories.png` - steeper lines = faster degradation

### 3. Comparison Questions

**Q**: *"Are there clear tipping points in parameter space?"*  
**A**: Check `qv_heatmap.png` - sharp color transitions indicate thresholds

**Q**: *"Does one parameter dominate performance?"*  
**A**: Compare campaigns - if fidelity sweep shows sharp drops but coherence sweep is gradual, fidelity dominates

---

## Advanced Usage

### Custom Threshold

By default, QV threshold = 2/3. To change:

```python
from campaign_plotter import create_all_campaign_plots

create_all_campaign_plots(
    campaign_results=results,
    output_dir=Path("output"),
    threshold=0.7,  # Custom threshold
)
```

### Standalone Plotting

Generate plots from saved results without re-running campaign:

```python
import json
from pathlib import Path
from campaign_plotter import create_all_campaign_plots

# Load saved results
with open("campaign_results.json") as f:
    campaign_results = json.load(f)

# Generate plots
create_all_campaign_plots(campaign_results, Path("new_plots"))
```

### Testing Plot Generation

Test with mock data:

```bash
cd examples
python campaign_plotter.py
# Generates test plots in test_campaign_plots/
```

---

## Performance Tips

### Large Campaigns (>50 configs)

- **Per-width plots**: May have many bars - consider filtering to top N configs
- **Trajectory plot**: Legend may overflow - automatically moves outside if >8 configs
- **Heatmap**: Auto-scales figure size based on matrix dimensions

### High-Resolution Exports

All plots saved at **300 DPI** by default (publication quality).

For presentations, you can regenerate at lower DPI:
```python
# In campaign_plotter.py, change:
fig.savefig(output_path, bbox_inches='tight', dpi=150)  # Faster, smaller files
```

---

## Troubleshooting

### Issue: Empty plots
**Cause**: No configurations passed at any width  
**Solution**: Check campaign_results.json - verify simulations ran successfully

### Issue: Missing per-width plots
**Cause**: Some widths not tested for all configs  
**Solution**: Expected behavior - only generates plots for tested widths

### Issue: Overlapping labels
**Cause**: Too many configurations or long names  
**Solution**: Config names auto-rotate 45° and truncate at 30 chars

---

## Summary

| Plot Type | Location | Answers |
|-----------|----------|---------|
| HOP by Width | `by_width/hop_width_{m}.png` | Which configs pass at specific width? |
| QV by Width | `by_width/qv_width_{m}.png` | What QV is achieved at specific width? |
| Global Average | `global/global_width_average.png` | Best overall performer? |
| Trajectories | `global/hop_trajectories.png` | How does HOP degrade? |
| Heatmap | `global/qv_heatmap.png` | Pass/fail pattern overview? |

**All plots are automatically generated** when running a parameter campaign.

---

**Created**: October 17, 2025  
**Version**: 1.1  
**Module**: `campaign_plotter.py`
