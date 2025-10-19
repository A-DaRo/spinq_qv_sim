# Executive Summary: spinq_qv_sim

**Quantum Volume Simulator for Silicon Spin Qubits**

---

## Project Purpose

The **spinq_qv_sim** project is a research-oriented, physics-first simulator designed to estimate **Quantum Volume (QV)** for silicon/silicon-germanium (Si/SiGe) spin-qubit devices. The simulator implements IBM's QV benchmarking protocol under **physically grounded noise models** derived from experimental parameters.

### Core Objectives

1. **Predict achievable Quantum Volume** for Si/SiGe spin-qubit devices before fabrication
2. **Guide hardware optimization** by identifying limiting factors (gate fidelity, coherence, readout errors)
3. **Enable sensitivity analysis** to prioritize experimental improvements
4. **Validate noise models** by reproducing measured gate fidelities through randomized benchmarking
5. **Support research** with publication-quality analysis and visualization tools

### Approach

The simulator implements a **noise-model-first** methodology that composes:
- **Stochastic channels** (depolarizing, amplitude/phase damping)
- **Coherent errors** (systematic over/under-rotations, ZZ coupling)
- **Non-Markovian noise** (quasi-static dephasing from 1/f-like spectra)
- **SPAM errors** (state preparation and readout imperfections)
- **Device constraints** (connectivity, gate durations)

The goal is to approximate experimentally relevant behavior within a controlled mathematical framework.

---

## Implementation Architecture

### System Design

The simulator follows a modular, pipeline-based architecture with strict backend abstraction for CPU/GPU portability:

```
Configuration (YAML) → Noise Model Builder → Circuit Generator
                            ↓                        ↓
                   Device Parameters    QV Random Circuits (width m)
                            ↓                        ↓
                    Kraus Operators        Transpilation (to device topology)
                            ↓                        ↓
                   Composite Noise         Gate Scheduling (with idles)
                            ↓                        ↓
                    Simulator Backend ← Apply gates + noise channels
                            ↓
                   Measurement Sampling (n_shots per circuit)
                            ↓
            Heavy-Output Probability (HOP) Computation
                            ↓
         Statistical Analysis (Bootstrap CI, QV Decision Rule)
                            ↓
           Results Storage (HDF5) + Visualization (Plots) + Reports (PDF)
```

### Core Components

#### 1. Configuration System (`config/`)
- **Pydantic-validated YAML configs** ensure type-safe device parameters
- **Defaults from experiment**: F₁=99.926%, F₂=99.8%, T₁=1s, T₂=99µs, T₂*=20µs
- **Metadata tracking**: Git hash, package versions, random seeds for reproducibility

#### 2. Noise Model Construction (`noise/`)

**Mathematical Conversions** (validated formulas):
- **Gate fidelity → Depolarizing probability**:
  - Single-qubit: `p₁ = 2(1 - F₁)` 
  - Two-qubit: `p₂ = (4/3)(1 - F₂)`
  
- **Coherence times → Decoherence probabilities**:
  - Amplitude damping: `p_amp = 1 - exp(-τ/T₁)`
  - Phase damping: `p_phi = 1 - exp(-τ/T_phi)` where `1/T_phi = 1/T₂ - 1/(2T₁)`
  
- **T₂* → Quasi-static noise amplitude**:
  - Gaussian detuning: `σ = √2/T₂*`

**Kraus Operator Implementations**:
- Depolarizing channels (Pauli error averaging)
- Amplitude damping (energy relaxation to ground state)
- Phase damping (pure dephasing without relaxation)
- Coherent unitary errors (systematic ZZ coupling, over-rotation)
- Readout POVM (asymmetric measurement errors)

**Unified Constrained Error Model (v2):**
- Compute decoherence-induced fidelity from T₁/T₂ during gate duration τ: `F_decoh(τ) = AvgF[ E_amp ∘ E_phi ]` with `p_amp = 1 - exp(-τ/T₁)`, `p_phi = 1 - exp(-τ/T_phi)`, `1/T_phi = 1/T₂ - 1/(2T₁)`.
- Include systematic coherent error `U_err` (e.g., `R_axis(ε)` or `exp(-i θ ZZ)`) with fidelity `F_coh = |Tr(U_err)|²/d²` relative to identity in the gate frame.
- Combine: `F_physical = F_decoh × F_coh`. To match a target experimental average fidelity `F_exp`, add only the residual depolarizing noise:
  - Single qubit: `p_residual = (3/2) × (1 - F_exp / F_physical)`
  - Two qubits: `p_residual = (5/4) × (1 - F_exp / F_physical)`
- Enforce `p_residual ≥ 0` (set to zero if `F_physical ≥ F_exp`). This prevents double counting and preserves time-dependence via τ.

#### 3. Circuit Generation & Compilation (`circuits/`)
- **QV Circuit Generator**: IBM-style random square circuits (width = depth = m)
  - Random two-qubit unitaries (Haar-distributed SU(4))
  - Deterministic from seed for reproducibility
  
- **Transpiler**: Maps logical circuits to physical device topology
  - Linear chain (nearest-neighbor connectivity)
  - All-to-all (full connectivity)
  - SWAP insertion for non-adjacent operations
  
- **Scheduler**: Time-aware gate parallelization
  - Maximize parallel two-qubit gates where topology allows
  - Insert idle operations for accurate coherence modeling

#### 4. Simulation Backends (`sim/`)

**Three backend implementations** with unified interface:

| Backend | Method | Memory | Max Qubits | Use Case |
|---------|--------|--------|------------|----------|
| **Statevector** | Pure state (CPU/GPU) | 2ⁿ × 16 bytes | context-dependent | General studies |
| **Density Matrix** | Exact Kraus propagation | 2²ⁿ × 16 bytes | context-dependent | Exact non-unitary channels |
| **MCWF** | Monte Carlo trajectories | 2ⁿ × 16 bytes/traj | context-dependent | Stochastic sampling |

- GPU usage is optional; backends are abstracted for portability.
- Deterministic RNG seeding is supported for reproducibility.

#### 5. Statistical Analysis (`analysis/`)

**Heavy-Output Probability (HOP)**:
- Heavy set = bitstrings with `ideal_probability > median(ideal_probs)`
- HOP = fraction of measurements in heavy set
- Expected values: Ideal=1.0, Random=0.5, QV threshold>2/3

**IBM QV Decision Rule** (both criteria must hold):
1. Mean HOP > 2/3 across all circuits
2. Lower 95% confidence interval > 2/3

**Statistical Methods**:
- Bootstrap confidence intervals (10,000 resamples)
- Clopper-Pearson binomial CI (shot-level uncertainty)
- One-sided hypothesis testing (H₀: HOP ≤ 2/3)

**QV Determination**:
```
QV = 2^m* where m* = max width satisfying both criteria
```

#### 6. Sensitivity & Ablation (`experiments/`)

**Parameter Sweeps**:
- **1D sweeps**: Single parameter variation (e.g., F₁ from 99.4% to 99.99%)
- **2D grids**: Cartesian product (e.g., F₁ × F₂ heatmap)
- Logarithmic spacing for time scales (T₁, T₂)

**Error Budget Analysis**:
- Ablation studies: Turn off individual noise sources
- Contribution metric: `% of gap = (Ideal_HOP - Source_Off_HOP) / (Ideal_HOP - Baseline_HOP)`
- Identifies limiting factors (e.g., "T₂ contributes 65% of QV loss")

**Campaign System**:
- Multi-configuration experiments (5-50 configs)
- Automated analysis and reporting
- Resume capability for interrupted runs
- Four preset campaign types:
  - **Comprehensive**: All parameters (F₁, F₂, T₁, T₂, gate times)
  - **Fidelity Focus**: Gate fidelities and SPAM errors
  - **Coherence Focus**: Wide T₁/T₂ range (log scale)
  - **Timing Focus**: Gate duration trade-offs

#### 7. Data Management (`io/`)

**HDF5 Structured Storage**:
```
/metadata                           # Config, git hash, versions, seeds
/circuits/{m}/{circuit_id}/
    spec                            # Circuit description (JSON)
    ideal_probs                     # Noiseless probability distribution
    measured_counts                 # Sampled bitstrings (compressed)
/aggregated/{m}/
    mean_hop, ci_lower, ci_upper    # Statistical summaries
    pass_qv                         # Boolean decision
```

**Output Formats**:
- **HDF5**: Primary storage (compressed, queryable)
- **JSON**: Campaign results, sensitivity data
- **CSV**: Tabular exports for analysis
- **PNG/SVG**: High-resolution plots (300 DPI)
- **PDF**: Multi-page reports with LaTeX-quality formatting

---

## Key Capabilities

### 1. End-to-End QV Prediction

**Input**: Device parameters (fidelities, coherence times, gate durations)  
**Output**: Estimated QV with statistical confidence

**Workflow**:
```bash
python -m spinq_qv.experiments.run_qv \
    --config examples/configs/baseline.yaml \
    --widths 2,3,4,5,6,7,8 \
    --output results/my_experiment/
```

**Results**:
- HOP vs width plot with error bars
- QV determination: "Achieves QV=64 (m=6)"
- Confidence intervals for each width
- Pass/fail indicators per IBM criteria

### 2. Noise Model Validation (Methodology)

Randomized benchmarking (RB) sequences can be simulated to estimate average gate fidelities under the constructed noise model. Agreement with configured target fidelities is used as a self-consistency check prior to QV studies.

### 3. Sensitivity Analysis

**Question**: "Which parameter improvements give biggest QV boost?"  
**Method**: Sweep parameters, compute correlation with achieved QV

**Example Results**:
| Parameter | Correlation | QV Range | Interpretation |
|-----------|-------------|----------|----------------|
| F₂ | +0.89 | 4 widths | **Strongest impact** - prioritize two-qubit gates |
| F₁ | +0.76 | 3 widths | Important secondary factor |
| T₂ | +0.42 | 1 width | Moderate - not currently limiting |
| T₁ | +0.08 | 0 widths | Negligible (T₁=1s already excellent) |

**Visualizations**:
- Parameter vs HOP curves (one line per width)
- Parameter vs QV (exponential y-axis)
- Correlation matrix heatmaps
- 2D sensitivity surfaces

### 4. Error Budget Decomposition

**Ablation Study**: Simulate with individual noise sources turned off

**Example Budget (illustrative)**:
```
Total QV gap (baseline → ideal): 4 widths
  - T₂ dephasing:        65% of gap  ← Primary bottleneck
  - F₂ depolarizing:     22% of gap
  - Readout errors:      8% of gap
  - F₁ depolarizing:     3% of gap
  - T₁ relaxation:       2% of gap
```

Interpretation of such budgets indicates which mechanisms dominate HOP degradation for given settings.

### 5. Campaign Management

**Multi-Configuration Experiments**:
- Generate 10-50 configs automatically
- Execute with parallelization (multi-core)
- Resume interrupted campaigns
- Aggregate results across configs

**Interactive HTML Report**:
- Embedded plots (all visualizations)
- Parameter sensitivity rankings
- Statistical tables
- Key findings summary
- Shareable (single-file, no dependencies)

Typical campaigns consist of tens of configurations across widths and shot counts; outputs include structured HDF5 data and plots.

### 6. Reproducibility and Configuration

The system records configuration details (e.g., YAML specs, seeds) alongside results for reproducibility. Backends expose deterministic RNG controls.

---

## Mathematical Foundations

### Noise Conversions (Exact Formulas)

All conversions validated against quantum information theory:

**1. Average Fidelity → Depolarizing Probability**

For d-dimensional system: `F_avg = 1 - p(d-1)/d`

Inverting:
- Single-qubit (d=2): `p₁ = 2(1 - F₁)`
- Two-qubit (d=4): `p₂ = (4/3)(1 - F₂)`

Example: F₁=99.926% → p₁≈0.148% per gate

**2. Coherence Times → Kraus Probabilities**

During gate duration τ:
- Amplitude damping: `p_amp = 1 - exp(-τ/T₁)`
- Phase damping: `p_phi = 1 - exp(-τ/T_phi)`
  where `T_phi = 1 / (1/T₂ - 1/(2T₁))`

Example: Single-qubit τ=60 ns, T₂=99 µs → p_phi≈0.061%

**3. Quasi-Static Noise (Non-Markovian)**

Ramsey T₂* decay from Gaussian detuning:
```
<exp(iΔt)> = exp(-σ²t²/2)  →  T₂* = √2/σ
```
Therefore: `σ = √2/T₂*` (rad/s)

**Sample once per circuit** to model slow (1/f) drift.

### Kraus Operators (Exact Representations)

**Amplitude Damping** (probability γ):
```
K₀ = [1      0    ]    K₁ = [0  √γ]
     [0  √(1-γ)]         [0   0]
```

**Phase Damping** (probability λ):
```
K₀ = [1      0    ]    K₁ = [0    0  ]
     [0  √(1-λ)]         [0  √λ]
```

**Depolarizing** (probability p):
```
ε(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
```

### Statistical Analysis

**Bootstrap Confidence Interval**:
1. Resample circuits with replacement (n_bootstrap=10,000)
2. Compute mean HOP for each resample
3. Percentiles → 95% CI = [Q₂.₅%, Q₉₇.₅%]

**IBM QV Pass Criteria** (AND condition):
```
mean(HOP) > 2/3  AND  CI_lower(HOP) > 2/3
```

**Heavy-Output Set Definition**:
```
H = {bitstring x : P_ideal(x) > median(P_ideal)}
HOP = (1/n_shots) × Σ_{x∈H} counts(x)
```

---

## Implementation Notes

- Numerical precision: default to complex128 (state) and float64 (probabilities).
- Channel composition order (typical): gate unitary → depolarizing (if residual) → amplitude damping → phase damping.
- For non-commuting effects or time-dependent Hamiltonians, piecewise-constant slicing or MCWF can approximate continuous evolution.

---

## Validation Methodology (Optional)

Prior to QV estimation, one may simulate randomized benchmarking sequences to confirm that the constructed noise model reproduces target average gate fidelities within a chosen tolerance. This is recommended for internal consistency but is not a substitute for experimental validation.

---

## Practical Usage Scenarios

### Scenario 1: Pre-Fabrication QV Estimation

**Context**: Lab has characterized single-qubit gates (F₁=99.9%, T₂=100µs), planning two-qubit implementation.

**Question**: "What QV can we achieve with F₂=99.5% vs 99.8%?"

**Workflow**:
```bash
# Create two configs (F2_995.yaml, F2_998.yaml)
# Run both
python -m spinq_qv.experiments.run_qv --config F2_995.yaml --output results/f2_995/
python -m spinq_qv.experiments.run_qv --config F2_998.yaml --output results/f2_998/

# Compare results
# F₂=99.5% → QV=16 (m=4)
# F₂=99.8% → QV=32 (m=5)
```

**Recommendation**: "Improving F₂ by 0.3% doubles QV. Prioritize two-qubit calibration."

### Scenario 2: Identifying Performance Bottleneck

**Context**: Device achieves QV=32 but simulations predicted QV=64.

**Question**: "Which parameter is underperforming?"

**Workflow**:
```bash
# Run ablation study
python -m spinq_qv.experiments.run_qv \
    --mode ablation \
    --config measured_params.yaml \
    --output results/ablation/

# Check error budget
# Output: "T₂ dephasing: 72% of gap"
```

**Recommendation**: "Measured T₂ may be lower than spec. Re-characterize T₂ with Hahn echo."

### Scenario 3: Hardware Improvement Roadmap

**Context**: Planning second-generation device with improved materials.

**Question**: "What QV improvement from T₂: 99µs→300µs?"

**Workflow**:
```bash
# Sensitivity sweep
python -m spinq_qv.experiments.run_qv \
    --mode sensitivity-1d \
    --param T2 \
    --values 99e-6,150e-6,200e-6,300e-6 \
    --config current_device.yaml \
    --output results/t2_sweep/

# View plot: T2_vs_qv.png
# T₂=99µs → QV=32
# T₂=300µs → QV=128
```

**Business Case**: "4× QV improvement justifies materials R&D investment."

### Scenario 4: Publication-Quality Analysis

**Context**: Writing paper on Si/SiGe qubit scalability.

**Requirement**: Rigorous QV analysis with statistical confidence.

**Workflow**:
```bash
# Production campaign (100 circuits/width, 5000 shots)
python -m spinq_qv.experiments.run_qv \
    --config production.yaml \
    --widths 2,3,4,5,6,7,8,9,10 \
    --output results/publication/

# Generate PDF report
python -m spinq_qv.analysis.report_generator \
    results/publication/qv_run_*.h5 \
    --output paper_figures/
```

**Output**:
- High-res plots (300 DPI PNG + vector SVG)
- LaTeX-ready statistical tables
- Reproducible analysis (seed, config, git hash in metadata)

---

## Documentation & Learning Resources

### Core Documentation

1. **README.md** - Project overview, quick start, architecture
2. **Technicalities.md** - Original design document (physics formulas)
3. **docs/math_foundations.md** - Complete mathematical derivations
4. **installation.md** - Detailed setup (CPU and GPU paths)
5. **CHANGELOG.md** - Version history (v0.1 → v1.0)

### Specialized Guides

6. **docs/campaign_system_guide.md** - Campaign workflows and best practices
7. **docs/campaign_plotting_guide.md** - Visualization reference
8. **examples/CAMPAIGN_README.md** - Campaign quick start

### Interactive Tutorials (Jupyter Notebooks)

1. **01_quickstart_qv_experiment.ipynb** - Your first QV run (15 min)
2. **02_interactive_campaign.ipynb** - Build custom campaigns
3. **03_noise_model_exploration.ipynb** - Understand noise channels
4. **04_sensitivity_analysis.ipynb** - Parameter sweep examples
5. **05_campaign_monitor.ipynb** - Real-time campaign tracking

### API Reference

- Inline docstrings (all public functions/classes)
- Type hints throughout (mypy-validated)
- Example usage in docstrings

---

## Technical Requirements

### Minimum (CPU-Only Development)

- **OS**: Windows 10+, Linux (Ubuntu 20.04+), macOS 11+
- **Python**: 3.10 or higher
- **RAM**: 8 GB (for m≤8 widths)
- **Storage**: 1 GB (package + results)
- **CPU**: Multi-core recommended (parallelization)

**Dependencies**:
```
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
h5py >= 3.8
matplotlib >= 3.7
pydantic >= 1.10
pytest >= 7.2
```

### Recommended (Production)

- **RAM**: 16-32 GB (for m≤12 widths with campaigns)
- **CPU**: 8+ cores (parallelization efficiency)
- **Storage**: 50-100 GB (large campaigns with HDF5)

### Optional (GPU Acceleration)

- **GPU**: NVIDIA with CUDA 11.0+ (Compute Capability 7.0+)
- **VRAM**: 8+ GB (for m≤12)
- **Driver**: Latest NVIDIA driver

**Additional Dependencies**:
```
cupy-cuda12x >= 12.0
cuquantum >= 23.08  (optional)
jax[cuda12] (optional alternative)
```

**Installation**:
```bash
# CPU-only (development)
pip install -r requirements_cpu.txt

# GPU-accelerated (production)
pip install -r requirements_gpu.txt
```

---

## Scope and Roadmap

### Current Scope

- Configuration-driven experiments using YAML inputs and Pydantic validation
- QV circuit generation, transpilation to device connectivity, and scheduling
- Noise model construction with Kraus channels, coherent errors, and quasi-static noise
- Backends: statevector, density matrix, and MCWF abstractions
- Analysis: heavy-output probability computation and bootstrap statistics

### Possible Extensions

- Gate-dependent noise and leakage models
- Tensor-network backends for larger widths
- Advanced error mitigation and readout calibration models
- Distributed execution for large campaign studies

### Notes

Performance characteristics and empirical timing vary by hardware and configuration and are intentionally omitted here; focus is on the mathematical and algorithmic structure.

---

<!-- Comparative discussion with other simulators is intentionally omitted per academic focus. -->

---

<!-- Success metrics and comparative validation targets are omitted to maintain an academic, technical emphasis. -->

---

## Getting Started

### 5-Minute Quick Start

```bash
# 1. Clone repository
git clone https://github.com/A-DaRo/spinq_qv_sim.git
cd spinq_qv_sim

# 2. Install dependencies (CPU-only)
pip install -r requirements_cpu.txt

# 3. Run validation (test noise model)
python -m spinq_qv.experiments.validate

# 4. Run small QV experiment (m=2,3,4)
python -m spinq_qv.experiments.run_qv \
    --config examples/configs/test_small.yaml \
    --output results/first_run/

# 5. View results
# Open: results/first_run/plots_*/hop_vs_width.png
```

Expected output: QV determination, HOP plot, statistical summary

### Next Steps

1. **Explore Tutorials**: Open `notebooks/01_quickstart_qv_experiment.ipynb`
2. **Customize Config**: Edit `examples/configs/baseline.yaml` with your device params
3. **Run Campaign**: Try `python examples/run_parameter_campaign.py --help`
4. **Read Math Foundations**: `docs/math_foundations.md` for deep understanding
5. **Run Production**: Use `production.yaml` for publication-quality results

---

## Contact & Support

**Issues & Bug Reports**: [GitHub Issues](https://github.com/A-DaRo/spinq_qv_sim/issues)

**Discussions**: [GitHub Discussions](https://github.com/A-DaRo/spinq_qv_sim/discussions)

**Author Contact**: Alessandro Da Ros (a.da.ros@student.tue.nl)

---

## Summary

**spinq_qv_sim** is a research-oriented simulator linking experimental device parameters to achievable Quantum Volume under mathematically specified noise models. It provides:

1. **Accurate Predictions**: Physics-based noise models validated against fidelities
2. **Actionable Insights**: Sensitivity analysis and error budgets guide optimization
3. **Research-Grade**: Statistical rigor, reproducibility, publication-quality outputs
4. **Modular Architecture**: Clear separation of configuration, circuits, noise, simulation backends, and analysis

Intended audience: researchers and students studying Si/SiGe spin-qubit performance under realistic noise assumptions.

Primary utility: structured exploration of how specific physical parameters (F₁, F₂, T₁, T₂, T₂*, gate times, SPAM) influence HOP and the resulting QV criterion.

---

**Last Updated**: October 19, 2025  
**Version**: 1.0.0  
**License**: MIT
