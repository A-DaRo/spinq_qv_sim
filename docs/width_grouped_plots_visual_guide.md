# Width-Grouped Campaign Plots - Visual Guide

## Quick Visual Reference

### Plot Types Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAMPAIGN PLOTTING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PER-WIDTH PLOTS (by_width/)                                         │
│  ════════════════════════════                                        │
│                                                                       │
│  1. HOP by Width (hop_width_m.png)                                   │
│     ┌────────────────────────────────────┐                          │
│     │ HOP │                                │                          │
│     │ 1.0 │    ║                           │                          │
│     │ 0.8 │  ╔═║══╗                        │  ← Bar chart            │
│     │ 0.6 │  ║ ║  ║ ─ ─ ─ Threshold      │  ← Error bars           │
│     │ 0.4 │  ║ ║  ║  ╔══╗                 │  ← Color: pass/fail     │
│     │ 0.2 │  ║ ║  ║  ║  ║                 │                          │
│     │ 0.0 └──┴─┴──┴──┴──┴─────────        │                          │
│     │      C1 C2 C3 C4                     │                          │
│     └────────────────────────────────────┘                          │
│     Answers: "Which configs pass at width m=X?"                      │
│                                                                       │
│  2. QV by Width (qv_width_m.png)                                     │
│     ┌────────────────────────────────────┐                          │
│     │ QV  │                                │                          │
│     │ 32  │  ║                             │  ← Shows QV=2^m         │
│     │ 16  │  ║  ║                          │  ← 0 if failed          │
│     │  8  │  ║  ║  ║                       │  ← Color coded          │
│     │  4  │  ║  ║  ║  ░                    │                          │
│     │  0  └──┴──┴──┴──┴─────────          │                          │
│     │      C1 C2 C3 C4                     │                          │
│     └────────────────────────────────────┘                          │
│     Answers: "What QV achieved at width m=X?"                        │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  GLOBAL PLOTS (global/)                                              │
│  ══════════════════════                                              │
│                                                                       │
│  3. Global Width Average (global_width_average.png)                  │
│     ┌──────────────────┬──────────────────┐                          │
│     │ Avg HOP          │ Max Width        │                          │
│     │ ╔══╗   ╔══╗      │    5│  ║          │                          │
│     │ ║  ║   ║  ║      │    4│  ║  ║       │                          │
│     │ ║  ║   ║  ║ ─ ─  │    3│  ║  ║  ║    │  ← QV labels on bars   │
│     │ ║  ║   ║  ║ ╔══╗ │    2│  ║  ║  ║    │                          │
│     └─┴──┴───┴──┴─┴──┴─┴─────┴──┴──┴──┴───┘                          │
│       C1   C2   C3          C1 C2 C3        │                          │
│     Answers: "Best overall performer?"                                │
│                                                                       │
│  4. HOP Trajectories (hop_trajectories.png)                          │
│     ┌────────────────────────────────────┐                          │
│     │ HOP │     C1 ╲                       │                          │
│     │ 1.0 │          ╲                     │  ← Line per config      │
│     │ 0.8 │      C2 ─╲─╲                   │  ← Shaded CI            │
│     │ 0.6 │     ─ ─ ─ ─╲─ Threshold        │  ← Shows degradation    │
│     │ 0.4 │        C3 ──╲╲                 │                          │
│     │ 0.2 │               ╲╲               │                          │
│     │ 0.0 └────────────────╲╲─────        │                          │
│     │      2  3  4  5  6  Width            │                          │
│     └────────────────────────────────────┘                          │
│     Answers: "How does HOP degrade?"                                 │
│                                                                       │
│  5. QV Heatmap (qv_heatmap.png)                                      │
│     ┌────────────────────────────────────┐                          │
│     │        Width                        │                          │
│     │        2  3  4  5  6                │                          │
│     │ C1   │ ✓  ✓  ✓  ✗  ✗ │             │  ← Green = pass          │
│     │ C2   │ ✓  ✓  ✗  ✗  ✗ │             │  ← Red = fail            │
│     │ C3   │ ✓  ✗  ✗  ✗  ✗ │             │  ← Gray = not tested     │
│     │ C4   │ ✗  ✗  ✗  ✗  ✗ │             │                          │
│     └────────────────────────────────────┘                          │
│     Answers: "Quick pass/fail overview?"                             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Typical Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RUN CAMPAIGN                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
           python run_parameter_campaign.py
                   --sweep-type fidelity_focus
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               AUTOMATIC PLOT GENERATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Execute QV Simulations                                 │
│  ════════════════════════════                                   │
│  [████████████████████████] 100% | 23/23 configs                │
│                                                                  │
│  Step 2: Generate Width-Grouped Plots  ← NEW                    │
│  ════════════════════════════════════                           │
│  [1/5] Per-width HOP comparisons                                │
│    ✓ hop_width_2.png                                            │
│    ✓ hop_width_3.png                                            │
│    ✓ hop_width_4.png                                            │
│    ...                                                           │
│  [2/5] Per-width QV achievement                                 │
│    ✓ qv_width_2.png                                             │
│    ✓ qv_width_3.png                                             │
│    ...                                                           │
│  [3/5] Global width average                                     │
│    ✓ global_width_average.png                                   │
│  [4/5] Configuration trajectories                               │
│    ✓ hop_trajectories.png                                       │
│  [5/5] QV heatmap                                               │
│    ✓ qv_heatmap.png                                             │
│                                                                  │
│  Step 3: Generate Campaign Analysis Plots                       │
│  ════════════════════════════════════════                       │
│  ✓ campaign_comparison.png                                      │
│  ✓ dashboard.png                                                │
│  ✓ correlation_matrix.png                                       │
│  ...                                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT STRUCTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  campaigns/my_campaign/                                          │
│  ├── plots/                                                      │
│  │   ├── by_width/          ← NEW: Per-width comparisons        │
│  │   │   ├── hop_width_2.png                                    │
│  │   │   ├── hop_width_3.png                                    │
│  │   │   ├── qv_width_2.png                                     │
│  │   │   └── ...                                                │
│  │   ├── global/            ← NEW: Overall summaries            │
│  │   │   ├── global_width_average.png                           │
│  │   │   ├── hop_trajectories.png                               │
│  │   │   └── qv_heatmap.png                                     │
│  │   └── overview/          ← Existing campaign plots           │
│  │       └── ...                                                │
│  ├── campaign_report.html                                       │
│  └── ...                                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Question → Plot Mapping

```
╔═══════════════════════════════════════════════════════════════════╗
║  QUICK DECISION TREE: WHICH PLOT SHOULD I CHECK?                  ║
╚═══════════════════════════════════════════════════════════════════╝

Start: What's your question?
│
├─ "Which config is BEST OVERALL?"
│  └─→ global_width_average.png (right panel)
│      Look at: Tallest bar = highest max width
│
├─ "Which config is MOST ROBUST?"
│  └─→ global_width_average.png (left panel)
│      Look at: Smallest error bars = consistent across widths
│
├─ "How does performance DEGRADE with width?"
│  └─→ hop_trajectories.png
│      Look at: Line slopes = degradation rate
│
├─ "What's the MINIMUM F1 for QV=X?"
│  └─→ hop_width_m.png (where QV=X means width m=log₂(X))
│      Look at: Lowest F1 with blue (passed) bar
│
├─ "Which configs PASS at width m=X?"
│  └─→ hop_width_X.png
│      Look at: Blue bars = passed configurations
│
├─ "Quick OVERVIEW of all results?"
│  └─→ qv_heatmap.png
│      Look at: Color patterns = pass/fail matrix
│
└─ "How do configs COMPARE at width m=X?"
   └─→ hop_width_X.png AND qv_width_X.png
       Look at: Direct comparison of bars
```

---

## Example Analysis Workflow

### Scenario: Fidelity Sweep (F1 from 0.999 to 0.995)

```
Step 1: Quick Overview
═══════════════════════
Open: qv_heatmap.png

Observation:
        W2  W3  W4  W5  W6
F1_999  ✓   ✓   ✓   ✓   ✗
F1_998  ✓   ✓   ✓   ✗   ✗
F1_997  ✓   ✓   ✗   ✗   ✗
F1_996  ✓   ✗   ✗   ✗   ✗
F1_995  ✗   ✗   ✗   ✗   ✗

Finding: Sharp cutoffs at each width
→ Proceed to detailed analysis


Step 2: Identify Best Performer
════════════════════════════════
Open: global_width_average.png (right panel)

Observation:
F1_999: Max width = 5 → QV = 32
F1_998: Max width = 4 → QV = 16
...

Finding: F1=0.999 achieves QV=32
→ Check if this is consistent


Step 3: Analyze Degradation
════════════════════════════
Open: hop_trajectories.png

Observation:
- All lines start high (~0.9 at width 2)
- F1_999 degrades slowest
- F1_995 crosses threshold at width 2

Finding: Degradation is consistent with fidelity
→ No unexpected interactions


Step 4: Determine Minimum Requirements
═══════════════════════════════════════
Goal: What F1 needed for QV=16? (width m=4)

Open: hop_width_4.png

Observation:
F1_999: HOP = 0.75 ± 0.03 → PASS
F1_998: HOP = 0.68 ± 0.04 → PASS
F1_997: HOP = 0.61 ± 0.05 → FAIL

Finding: Minimum F1 ≈ 0.998 for QV=16
→ Tolerance: ~0.001


Step 5: Verify at Adjacent Widths
═══════════════════════════════════
Open: hop_width_3.png and hop_width_5.png

Observation:
Width 3: F1_997 still passes (HOP=0.71)
Width 5: F1_998 fails (HOP=0.61)

Finding: Requirements scale predictably
→ Each additional width needs ~ΔF1 ≈ 0.001


Conclusion
══════════
For this device:
- QV=32 requires F1 ≥ 0.999
- QV=16 requires F1 ≥ 0.998
- Rule of thumb: ΔF1 ≈ 0.001 per width level
```

---

## Color Coding Reference

```
┌────────────────────────────────────────────────────────────┐
│  COLOR SCHEME (Consistent Across All Plots)                │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  PERFORMANCE STATUS:                                        │
│  ══════════════════                                         │
│  🔵 Blue (#2E86AB)    →  Passed QV threshold               │
│  🟣 Purple (#A23B72)  →  Failed QV threshold               │
│  ⬜ Gray (#CCCCCC)    →  Not tested / N/A                  │
│                                                             │
│  REFERENCE LINES:                                           │
│  ════════════════                                           │
│  🔴 Red (dashed)      →  QV threshold (2/3)                │
│                                                             │
│  UNCERTAINTY:                                               │
│  ═══════════                                                │
│  Light shading        →  95% confidence interval           │
│  Error bars           →  Statistical uncertainty           │
│                                                             │
│  TRAJECTORY COLORS:                                         │
│  ══════════════════                                         │
│  Viridis colormap     →  Distinguishes multiple configs    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## File Naming Convention

```
Per-Width Plots:
  hop_width_{m}.png      →  HOP comparison at circuit width m
  qv_width_{m}.png       →  QV achievement at circuit width m

Global Plots:
  global_width_average.png  →  Overall performance summary
  hop_trajectories.png      →  HOP vs width for all configs
  qv_heatmap.png            →  Pass/fail matrix

Example:
  hop_width_2.png   = HOP comparison at width m=2
  hop_width_5.png   = HOP comparison at width m=5
  qv_width_3.png    = QV achievement at width m=3
```

---

## Quick Commands

```bash
# Test plotting system
python examples/campaign_plotter.py

# View test output
cd test_campaign_plots/plots
ls by_width/   # Per-width plots
ls global/     # Global plots

# Run small test campaign
python run_parameter_campaign.py \
    --base-config configs/test_small.yaml \
    --sweep-type fidelity_focus \
    --n-points 3 \
    --output test_campaign

# View results
cd test_campaign/plots
# Open plots in your image viewer
```

---

## Summary Card

```
╔═══════════════════════════════════════════════════════════╗
║           WIDTH-GROUPED CAMPAIGN PLOTS                     ║
╠═══════════════════════════════════════════════════════════╣
║                                                            ║
║  📊 5 PLOT TYPES                                           ║
║     • 2 per-width types (N plots each)                    ║
║     • 3 global summary types                              ║
║                                                            ║
║  🎯 KEY FEATURES                                           ║
║     • Width-level grouping                                ║
║     • Global averaging                                    ║
║     • Pass/fail visualization                             ║
║     • Publication quality (300 DPI)                       ║
║                                                            ║
║  📁 OUTPUT LOCATIONS                                       ║
║     • plots/by_width/    (per-width comparisons)          ║
║     • plots/global/      (overall summaries)              ║
║                                                            ║
║  🚀 INTEGRATION                                            ║
║     • Automatic in run_parameter_campaign.py              ║
║     • Standalone: python campaign_plotter.py              ║
║                                                            ║
║  📖 DOCUMENTATION                                          ║
║     • campaign_plots_by_width_guide.md                    ║
║     • width_grouped_plots_implementation.md               ║
║                                                            ║
╚═══════════════════════════════════════════════════════════╝
```

---

**Quick Start**: `python examples/campaign_plotter.py`  
**Full Campaign**: `python run_parameter_campaign.py --sweep-type fidelity_focus`  
**View Results**: Open `campaigns/*/plots/global/` and `by_width/`

