# spinq_qv_sim

**Physics-first Quantum Volume simulator for Si/SiGe spin qubits**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-167%20passing-brightgreen.svg)](tests/)

A production-ready simulator for estimating **Quantum Volume (QV)** of Si/SiGe spin-qubit devices using physically realistic noise models. Built from experimental parameters (gate fidelities, coherence times, SPAM errors) with full support for sensitivity analysis, error budgeting, and campaign-based parameter sweeps.

---

## 🎯 Key Features

### ✅ **Implemented & Production-Ready**

- **📊 Full QV Pipeline**: Generate → Transpile → Schedule → Simulate → Analyze
- **🔬 Physics-Based Noise Models**: Depolarizing, amplitude damping, phase damping, coherent errors, quasi-static noise, readout errors
- **🚀 Multiple Backends**: Statevector (CPU/GPU), Density Matrix (exact Kraus), Monte Carlo Wavefunction
- **📈 Statistical Analysis**: Bootstrap confidence intervals, IBM QV decision rule (HOP > 2/3)
- **🔍 Sensitivity & Ablation**: 1D/2D parameter sweeps, error budget decomposition, limiting factor identification
- **📝 Campaign System**: Multi-configuration experiments with state persistence, resume capability, automated reporting
- **📉 Publication-Quality Plots**: HOP vs width, error budgets, sensitivity heatmaps, per-width comparisons
- **🔄 Reproducibility**: Deterministic RNG, git hash tracking, metadata logging, HDF5 structured storage
- **⚡ Parallelization**: Multi-core circuit execution with performance profiling
- **📖 Complete Documentation**: Mathematical foundations, API reference, tutorials (5 Jupyter notebooks)

---

## 🏗️ Architecture

### Core Components

```
spinq_qv/
├── config/          # YAML configs with Pydantic validation
├── circuits/        # QV generation, transpilation, scheduling
├── noise/           # Physical noise model construction
│   ├── channels.py  # Kraus operators (depolarizing, damping)
│   ├── coherent.py  # Systematic errors (ZZ, over-rotation)
│   ├── stochastic.py# Quasi-static noise (1/f, T2*)
│   └── builder.py   # Device params → composite model
├── sim/             # Backend abstraction (CPU/GPU)
│   ├── statevector.py    # Pure state (NumPy/CuPy)
│   ├── density_matrix.py # Kraus propagation
│   └── mcwf.py           # Monte Carlo wavefunction
├── analysis/        # HOP computation, statistics, ablation
├── io/              # HDF5 storage, data formats
├── experiments/     # CLI runners (QV, campaigns, sensitivity)
└── utils/           # RNG, logging, performance profiling
```

### Data Flow

```
Config YAML → NoiseModelBuilder → Circuit Generator
                     ↓                    ↓
          Simulator Backend ← Transpiler + Scheduler
                     ↓
       Measurement Sampling → HOP Calculation → Statistics
                                                     ↓
                                         Plots + HDF5 + Reports
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/A-DaRo/spinq_qv_sim.git
cd spinq_qv_sim

# CPU-only install (recommended for development)
pip install -r requirements_cpu.txt

# GPU install (optional, for large widths)
# See installation.md for CUDA-specific instructions
pip install -r requirements_gpu.txt
```

### Run Your First QV Experiment

```bash
# Single QV run (widths 2-5, fast test)
python -m spinq_qv.experiments.run_qv \
    --config examples/configs/baseline.yaml \
    --widths 2,3,4,5 \
    --output results/my_first_run/

# View results
# HDF5 data: results/my_first_run/qv_run_*.h5
# Plots: results/my_first_run/plots_*/
```

**Expected output:**
- `qv_run_TIMESTAMP.h5` - Structured HDF5 with all data
- `plots_TIMESTAMP/` - HOP vs width, distributions, summary
- Console: Real-time HOP per width, QV determination

---

## 📚 Usage Examples

### 1. Basic QV Experiment

```python
from spinq_qv.config import Config
from spinq_qv.experiments.run_qv import run_qv_experiment

# Load config
config = Config.from_yaml("examples/configs/baseline.yaml")

# Run experiment
results = run_qv_experiment(
    config=config,
    widths=[2, 3, 4, 5],
    output_dir="results/my_experiment/"
)

# Results dict contains:
# - results[width]['mean_hop']
# - results[width]['ci_lower']
# - results[width]['qv_passed']
```

### 2. Sensitivity Analysis

```bash
# 1D parameter sweep (vary F1)
python -m spinq_qv.experiments.run_qv \
    --config examples/configs/baseline.yaml \
    --mode sensitivity-1d \
    --param device.single_qubit_fidelity \
    --values 0.995,0.997,0.999,0.99926 \
    --widths 2,3,4,5,6

# 2D parameter grid (F1 vs F2)
python -m spinq_qv.experiments.run_qv \
    --config examples/configs/sensitivity_grid.yaml \
    --mode sensitivity-2d \
    --param device.single_qubit_fidelity \
    --param2 device.two_qubit_fidelity \
    --values 0.995,0.997,0.999 \
    --values2 0.990,0.995,0.998
```

### 3. Error Budget Analysis

```bash
# Run ablation study
python -m spinq_qv.experiments.run_qv \
    --config examples/configs/test_ablation.yaml \
    --mode ablation \
    --widths 2,3,4,5

# Output: error_budget.json + pie/bar charts
# Shows contribution of each error source
```

### 4. Production Campaign

```bash
# Multi-parameter campaign with resume capability
python examples/run_parameter_campaign.py \
    --base-config examples/configs/production.yaml \
    --sweep-type comprehensive \
    --n-points 5 \
    --output campaigns/my_campaign/

# Resume if interrupted
python examples/run_parameter_campaign.py \
    --resume campaigns/my_campaign/

# Output: campaign_report.html (interactive dashboard)
```

---

## 🧪 Noise Model

### Device Parameters (Baseline Config)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Single-qubit fidelity** (F1) | 99.926% | Experimental |
| **Two-qubit fidelity** (F2) | 99.8% | Experimental |
| **T1** (relaxation) | 1 s | Donor spins |
| **T2** (Hahn echo) | 99 µs | Experimental |
| **T2*** (Ramsey) | 20 µs | Quasi-static dephasing |
| **Single-qubit gate time** | 60 ns | EDSR π/2 pulse |
| **Two-qubit gate time** | 40 ns | CZ gate |
| **Readout fidelity** | 99.97% | Best reported |
| **Initialization fidelity** | 99.4% | Experimental |

### Noise Conversion Formulas

**Critical formulas** (implemented in `noise/builder.py`):

#### Fidelity → Depolarizing Probability
```
p_1qubit = 2 * (1 - F1)              # Example: F1=0.99926 → p1≈0.00148
p_2qubit = (4/3) * (1 - F2)          # Example: F2=0.998  → p2≈0.00267
```

#### Coherence → Decoherence Probability
```
p_amp = 1 - exp(-τ / T1)             # Amplitude damping
p_phi = 1 - exp(-τ / T_phi)          # Phase damping
where 1/T_phi = 1/T2 - 1/(2*T1)
```

#### T2* → Quasi-Static Noise
```
σ_detuning = sqrt(2) / T2*           # Gaussian sampling per circuit
```

**See `docs/math_foundations.md` for complete derivations.**

---

## 📊 Output Formats

### HDF5 Structure

```
qv_run_TIMESTAMP.h5
├── /metadata                    # Config, git hash, versions
├── /circuits/{m}/{id}/
│   ├── spec                     # Circuit description
│   ├── ideal_probs              # Noiseless distribution
│   └── measured_counts          # Shot outcomes
└── /aggregated/{m}/
    ├── mean_hop                 # Mean HOP across circuits
    ├── ci_lower / ci_upper      # 95% confidence intervals
    └── qv_passed                # Boolean pass/fail
```

### Plots Generated

- `hop_vs_width.png` - Main result with error bars
- `qv_summary.png` - Pass/fail indicators
- `hop_distributions.png` - Violin plots per width
- `error_budget_pie.png` - Error source contributions
- `sensitivity_heatmap.png` - 2D parameter impact

---

## 🧮 Mathematical Foundations

### Heavy-Output Probability (HOP)

**Definition**: Fraction of measurement outcomes in the "heavy" set (ideal probability > median).

```python
# For each circuit:
ideal_probs = compute_ideal_probabilities(circuit)
median_prob = np.median(ideal_probs)
heavy_set = {outcome: ideal_probs[outcome] > median_prob}

# From measurements:
hop = sum(counts[outcome] for outcome in heavy_set) / total_shots
```

### IBM QV Decision Rule

**Pass criterion** (both must hold):
1. Mean HOP > 2/3 across all circuits
2. Lower 95% CI > 2/3 (bootstrap confidence interval)

**QV assignment**: QV = 2^m where m is largest passing width.

### Statistical Methods

- **Bootstrap CI**: 10,000 resamples with replacement
- **Hypothesis test**: One-sided test H₀: HOP ≤ 2/3
- **Shot noise**: Binomial(n_shots, HOP_true)

**See `docs/math_foundations.md` for detailed proofs.**

---

## 🔬 Backends

### Statevector (Default)

- **Memory**: 2^n × 16 bytes (complex128)
- **Scalable to**: n ≤ 20 qubits (CPU), n ≤ 25+ (GPU)
- **Use for**: Most QV experiments

```bash
python -m spinq_qv.experiments.run_qv \
    --config baseline.yaml \
    --backend statevector
```

### Density Matrix (Exact Noise)

- **Memory**: 2^(2n) × 16 bytes
- **Scalable to**: n ≤ 12 qubits (CPU)
- **Use for**: Exact Kraus channel propagation

```bash
python -m spinq_qv.experiments.run_qv \
    --config baseline.yaml \
    --backend density_matrix \
    --widths 2,3,4,5,6  # Limit to small widths
```

### Monte Carlo Wavefunction (MCWF)

- **Memory**: 2^n × 16 bytes per trajectory
- **Scalable to**: n ≤ 20+ qubits
- **Use for**: Large systems with stochastic noise

```bash
python -m spinq_qv.experiments.run_qv \
    --config baseline.yaml \
    --backend mcwf \
    --mcwf-trajectories 100
```

---

## 🎨 Campaign System

### What is a Campaign?

A **campaign** is a multi-configuration experiment exploring parameter space:
- Sweep device parameters (F1, F2, T1, T2, gate times)
- Generate width-grouped comparisons
- Automated sensitivity analysis
- Interactive HTML report

### Campaign Types

| Type | Parameters | Use Case |
|------|-----------|----------|
| **comprehensive** | F1, F2, T1, T2, gate times | Full exploration |
| **fidelity_focus** | F1, F2, F_readout, F_init | Gate optimization |
| **coherence_focus** | T1, T2, T2* | Materials research |
| **timing_focus** | t_single, t_two, t_readout | Speed-accuracy tradeoffs |

### Run a Campaign

```bash
# Quick test (3 configs, fast)
python examples/run_parameter_campaign.py \
    --base-config examples/configs/test_small.yaml \
    --sweep-type fidelity_focus \
    --n-points 3 \
    --output campaigns/test/

# Production (25 configs, overnight)
python examples/run_parameter_campaign.py \
    --base-config examples/configs/production.yaml \
    --sweep-type comprehensive \
    --n-points 5 \
    --output campaigns/prod_$(date +%Y%m%d)/
```

### Campaign Outputs

```
campaigns/my_campaign/
├── campaign_report.html         # 📊 OPEN THIS IN BROWSER
├── campaign_results.json        # Aggregated metrics
├── configs/                     # Generated YAML configs
├── results/                     # Per-config JSON results
├── plots/
│   ├── by_width/               # Per-width comparisons
│   ├── global/                 # Campaign-level summaries
│   └── overview/               # Main comparison plots
└── analysis/                    # Sensitivity analysis
```

**See `examples/CAMPAIGN_README.md` for full guide.**

---

## 📖 Documentation

### Core Docs

- **`Technicalities.md`** - Original project blueprint (physics formulas)
- **`docs/math_foundations.md`** - Mathematical derivations (Kraus, HOP, stats)
- **`installation.md`** - CPU/GPU setup instructions
- **`CHANGELOG.md`** - Complete development history (v0.1 → v1.0)

### Campaign System

- **`examples/CAMPAIGN_README.md`** - Quick start guide
- **`docs/campaign_system_guide.md`** - Full campaign documentation
- **`docs/campaign_plotting_guide.md`** - Visualization details
- **`docs/campaign_plots_by_width_guide.md`** - Per-width analysis

### Notebooks (Interactive Tutorials)

1. **`01_quickstart_qv_experiment.ipynb`** - First QV run
2. **`02_interactive_campaign.ipynb`** - Build custom campaigns
3. **`03_noise_model_exploration.ipynb`** - Understand noise channels
4. **`04_sensitivity_analysis.ipynb`** - Parameter impact studies
5. **`05_campaign_monitor.ipynb`** - Real-time campaign tracking

---

## 🧪 Testing

### Run Tests

```bash
# All tests (167 passing, 1 skipped for GPU)
pytest tests/

# Unit tests only (fast, ~5s)
pytest tests/unit/

# Integration tests (slower, ~30s)
pytest tests/integration/

# Specific test file
pytest tests/unit/test_depolarizing_conversion.py -v
```

### Test Coverage

| Category | Tests | Focus |
|----------|-------|-------|
| **Unit** | 120 | Formulas, conversions, Kraus operators |
| **Integration** | 47 | Pipelines, backends, reproducibility |
| **Statistical** | 12 | RB validation, sampling accuracy |
| **Performance** | 3 | Parallelization, profiling |

---

## ⚡ Performance

### Parallelization

```bash
# Use all CPU cores
python -m spinq_qv.experiments.run_qv \
    --config baseline.yaml \
    --parallel \
    --workers 8

# Enable profiling
python -m spinq_qv.experiments.run_qv \
    --config baseline.yaml \
    --parallel \
    --profile \
    --output results/profiled_run/
```

### Benchmarks (Approximate)

| Width (m) | Circuits | Shots | Backend | Time (CPU) | Time (GPU) |
|-----------|----------|-------|---------|------------|------------|
| 2-5 | 50 | 1000 | Statevector | ~2 min | ~30 s |
| 2-8 | 100 | 5000 | Statevector | ~30 min | ~5 min |
| 2-10 | 100 | 5000 | Statevector | ~4 hours | ~30 min |
| 2-6 | 50 | 1000 | Density Matrix | ~10 min | N/A |

*Intel i7-12700K, NVIDIA RTX 3080 (GPU times)*

---

## 🛠️ Development

### Project Structure

1. **Configuration & metadata** — a single YAML/JSON config describing device parameters, noise-model choices, simulator backend options, and experiment parameters (random seeds, number of circuits, shots per circuit, widths to test).
2. **Circuit generator** — implement IBM QV circuit generator (square circuits with random layers of two-qubit Haar-like or random native-gate sequences) parameterized by width `m`, connectivity graph, native gate set, and transpilation rules.
3. **Transpiler / mapper** — map logical QV circuits to the device native gates and connectivity, schedule gates into time steps (parallelism), add idle operations where necessary.
4. **Noise-model builder** — take device parameters (fidelities, gate times, T1/T2/T2*, SPAM, crosstalk maps, 1/f parameters) and assemble a composite noise model made of:

   * Gate-dependent channels (Kraus or Pauli/d epolarizing approximations).
   * Amplitude-damping (T1) and pure-dephasing (Tφ derived from T2/T2*).
   * Coherent error terms (small systematic unitary over/under rotation, ZZ/ZX couplings).
   * Correlated noise (quasi-static low-frequency 1/f or Ornstein–Uhlenbeck noise).
   * Crosstalk (microwave control crosstalk and readout crosstalk).
   * Leakage channels (optional qutrit-level leakage to valley states).
5. **Simulator backend** — choose simulation method(s):

   * Exact statevector + sampling (fast for pure, up to ~20 qubits, but needs repeated shots per circuit).
   * Density-matrix (for non-unitary channels; feasible up to ~12–14 qubits on desktop/workstation).
   * Kraus-channel propagation or Monte-Carlo wavefunction (stochastic unraveling) — tradeoffs for scaling.
   * Tensor-network path/PEPS or specialized GPU statevector for larger widths if needed.
6. **Execution harness** — run the QV experiments: for each width `m`, for each random circuit instance:

   * transpile, simulate `n_shots` to compute output bitstring frequencies,
   * compute heavy outputs for that circuit (compare to ideal output amplitudes),
   * produce heavy-output probability (HOP) per circuit,
   * aggregate across circuits to estimate mean HOP and error bars.
7. **Statistics & decision rule** — determine for each `m` whether the QV test passes at desired confidence (e.g., mean HOP > 2/3 and lower confidence bound > 2/3). Determine final QV (max `m` that passes).
8. **Sensitivity & ablation studies** — run parameter sweeps and ablations to see the impact of T2, gate fidelity, connectivity, crosstalk, SPAM, etc.
9. **Outputs & plots** — save raw data and produce publication-ready figures.
10. **Validation** — sanity checks to ensure the noise model reproduces reported single- and two-qubit fidelities (RB-style simulations) and readout fidelities before trusting QV results.

---

# Inputs / configuration (concrete defaults from your numbers)

Use these as defaults in the config file (all values can be overridden):

* Qubit counts to test (widths): `m ∈ {2,3,4,5,6,7,8,9,10,11,12}` (extendable).
* Native gates:

  * Single-qubit `Xπ/2` time = `60 ns` (π/2), `Xπ` ~ `120 ns`.
  * Two-qubit CZ/CPHASE time = `40 ns`.
* Fidelities (average gate fidelities):

  * Single-qubit `F1 = 99.926%` (0.99926). Use alternatives (99.6%, etc.) in sensitivity runs.
  * Two-qubit `F2 = 99.8%` (0.998).
* Decoherence:

  * `T1 = 1 s` (donor spins) — for quantum dot electrons, if no precise T1 use a large value (≥1 s) so amplitude damping is negligible on gate times.
  * `T2 = 99 µs` (Hahn echo); use `T2* = 20 µs` for Ramsey quasi-static dephasing modelling; CPMG extended value `3.1 ms` can be used for pulsed-decoupled gates if applicable.
* SPAM / readout:

  * Readout fidelity `F_read = 99.97%` (as one reported best; otherwise 98.1–99.8%).
  * Readout time ~ `10 µs`.
  * Initialization fidelities `F_init` as provided (e.g., 99.4% or 97.5%).
* Crosstalk:

  * Start with small coherent crosstalk angles (e.g., stray microwave pulses cause 1–5% of intended amplitude on neighbor), adjustable.
* 1/f / quasi-static noise:

  * Model with a quasi-static Gaussian-distributed detuning per run (σ set to reproduce measured T2* = 20 µs).
* Leakage:

  * Optional: leakage to valley state modeled as small probability per two-qubit gate (tunable).
* Simulation parameters:

  * `n_circuits_per_m` = 50 (recommend 50–200; see resource estimate).
  * `n_shots` = 1000 per circuit (1000–5000 depending on confidence requirements).
  * Random seeds logged.
* Confidence level: 95% for statistical tests (bootstrap or Clopper-Pearson intervals).

---

# Noise-model details (mathematics & mapping from experimental numbers)

This section gives explicit formulas and mapping rules to convert experimental reported numbers into channels that a simulator can ingest.

## 1) From average gate fidelity to depolarizing probability (Pauli channels)

For a d-dimensional system, the relation (depolarizing channel) is:

* `F_avg = 1 - p * (d-1)/d`
  => `p = d/(d-1) * (1 - F_avg)`

For qubits:

* single-qubit (d=2): `p1 = 2*(1 - F1)`
* two-qubit (d=4): `p2 = (4/3)*(1 - F2)`

**Numeric examples (from your numbers):**

* `F1 = 0.99926` ⇒ `p1 ≈ 2*(1 - 0.99926) = 0.00148` (≈ **0.148%** depolarizing probability per single-qubit gate).
* `F2 = 0.998` ⇒ `p2 ≈ 4/3*(1 - 0.998) = 0.0026667` (≈ **0.267%** per two-qubit gate).

> Use these `p1`/`p2` as starting stochastic error strengths. Note: depolarizing approximations ignore coherent errors — add separate coherent terms below.

## 2) Amplitude damping & dephasing from T1/T2

For a gate of duration `τ`, the amplitude-damping probability and dephasing probability are:

* amplitude damping: `p_amp = 1 - exp(-τ / T1)`
* dephasing (approx): `p_phi = 1 - exp(-τ / Tφ)`

Where `1 / Tφ = 1 / T2 - 1 / (2 T1)` or use T2 directly if T1 ≫ T2. For your numbers:

* For single-qubit τ = 60 ns, `T1 = 1 s`: `p_amp ≈ 6.0e-8` (negligible).
* For dephasing with `T2 = 99 µs`: `p_phi ≈ 0.0006059` (≈ **0.0606%** per single-qubit gate).
* For two-qubit τ = 40 ns: `p_amp ≈ 4.0e-8`, `p_phi ≈ 0.000404` (≈ **0.0404%**).

**Implementation note:** apply amplitude-damping and dephasing channels during gates and idles. For idles, use the same formulas with idle durations.

## 3) Coherent errors (unitary misrotations, residual ZZ coupling)

* Parameterize coherent single-qubit over/under-rotation as a small angle `ε1` (radians) per gate; map a fraction of the depolarizing error to coherent rotation if you want to model coherent behavior. A start value could be ε1 ≈ sqrt(p1)/10 or set by calibration data. Example: if p1=0.00148, set ε1 = 1e-2 rad (tunable).
* Two-qubit coherent residual ZZ (or unwanted exchange) modeled as `U_err = exp(-i θ ZZ/2)` per two-qubit gate, where `θ` is a small phase (e.g., θ ∈ [0.001, 0.02] rad). This is important because coherent ZZ errors strongly affect depth-limited benchmarks.

**Important:** Quantum Volume is sensitive to coherent errors. Also consider including a small, systematic phase in EDSR pulses or CZ entangling rotations.

## 4) Crosstalk

* **Control (microwave) crosstalk:** when applying a pulse to qubit `i`, a fraction `α_ij` of that pulse appears on neighbor `j` as a scaled rotation. Model by composing target rotation on `i` and small simultaneous unitary rotation `R_j(α_ij * θ)`.
* **Readout crosstalk:** correlation matrix `C_read` mapping actual measurement probabilities to observed outcomes; model as a small probability of mis-associating neighbor results.

## 5) Quasi-static / 1/f noise (non-Markovian)

* Model low-frequency charge/hyperfine noise as a slowly varying detuning `δ` per qubit sampled once per circuit instance from Gaussian with variance set to match T2* = 20 µs (i.e., the Ramsey dephasing). This produces run-to-run coherent phase errors.
* Optionally simulate time-dependent noise using an Ornstein–Uhlenbeck process with parameters chosen to reproduce observed spectral density.

## 6) Leakage & valley states (optional)

* For silicon valleys, model a three-level system (`|0⟩`, `|1⟩`, `|v⟩`). Include a per-two-qubit-gate leakage probability `p_leak` and a relaxation `|v⟩ -> |0/1⟩` with a characteristic time.

---

# Circuit generation & transpilation

1. **Generate QV circuits**: For each `m`:

   * Create `n_rand` random circuits of width `m` and depth `m` using random two-qubit layers (random permutation of qubits, random pairings according to mapper), interleaving single-qubit random rotations sampled from Haar or from a suitable approximate distribution.
   * Keep both the **ideal** (noise-free) output amplitudes (for heavy-output determination) and the native-gate decomposed circuits (for simulation under noise).

2. **Mapping to device**:

   * Embed logical qubits onto a specific set of physical qubits (choose mapping strategy: linear chain mapping; for 2D arrays use minimal SWAP routing).
   * Respect device connectivity: use your device’s adjacency graph (for linear 12QD array you described, limited neighbor connectivity) — limited connectivity reduces parallel two-qubit gate layers and lowers effective QV.

3. **Transpile to native gates**:

   * Decompose random two-qubit unitaries into device-native gates (CZ + single-qubit rotations) or whatever native gate set you selected.
   * Insert idles when gates conflict on same physical qubit to simulate time and accumulate decoherence.

4. **Schedule**:

   * Compute gate start/end times, so that durations matter (simultaneous gates on different qubits produce independent decoherence except for explicitly modeled crosstalk).

---

# Simulation engines & tradeoffs

Which simulator to use depends on `m`:

* **Density-matrix (Kraus) simulation**:

  * Pros: can model non-unitary channels (T1/T2, amplitude damping) exactly.
  * Cons: memory scales as 4^n; feasible up to ~12–14 qubits on a beefy workstation (see memory estimates below).
* **Statevector with stochastic unraveling (Monte-Carlo wavefunction)**:

  * Pros: memory 2^n; can include non-unitary effects stochastically; scales farther.
  * Cons: need many trajectories to converge.
* **Tensor-network / contracted simulators**:

  * Use if you need higher `m` but circuits are shallow and connectivity allows efficient TN contraction.
* **Specialized high-performance simulators (qsim, cuStateVec)** for large width but limited depth.

**Memory example (approx):**

* Statevector memory ~ `2^n * 16 bytes` (complex128).
* Density matrix memory ~ `2^(2n) * 16 bytes`.
* Example: density matrix for `n=12` ≈ 256 MB; for `n=14` ≈ 4 GB. (This guides which `m` you can simulate with density matrices on a workstation.)

---

# Execution plan (detailed steps)

This describes the exact step-by-step loop the experiment runner will perform.

1. **Initialize**:

   * Load YAML config, seed RNGs, create output folders (results/, plots/, logs/). Save config and seed.
2. **Noise-model calibration**:

   * Compute `p1`, `p2` from `F1`, `F2` (see formulas above).
   * Compute `p_amp` and `p_phi` for each gate type from `T1`, `T2` and gate durations `τ`.
   * Build channels: combine depolarizing/Pauli channels with amplitude damping and dephasing into a composite Kraus per gate or use Pauli-twirled equivalents for efficiency.
3. **Validation check (small)**:

   * Run short RB or two-qubit benchmarking circuits to verify the implementation of the noise model reproduces expected average fidelities `F1`, `F2` (within tolerance). Save results.
4. **For each width `m` in ascend order**:

   * For `i` from `1..n_circuits_per_m`:

     * Generate random QV circuit `C_i` (width=depth=m), produce ideal outputs (probabilities or amplitudes).
     * Sample per-circuit quasi-static noise variables (detunings) if modeling quasi-static noise.
     * Transpile `C_i` to native gates with scheduling and mapping.
     * Simulate `C_i` under the noise model with `n_shots`:

       * If using density matrix or statevector + measurement sampling, produce sampled bitstrings and counts.
     * Compute heavy outputs for `C_i`:

       * For the ideal circuit, rank outputs by ideal probabilities; define heavy outputs as those with ideal prob > median ideal prob (the IBM QV definition).
       * For noisy sampled outputs, compute the fraction of shots that are heavy outputs = `HOP_i`.
     * Store per-circuit `HOP_i`, circuit seed, per-qubit error diagnostics, and raw shot counts.
   * Aggregate: compute mean `HOP(m) = mean_i HOP_i` and standard error (bootstrap or analytic).
   * Decision test: if mean and lower confidence interval exceed 2/3 (or chosen threshold), mark `m` as passed.
5. **Determine QV**:

   * Find maximum `m` that passes the decision test. QV = `2^m`.
6. **Sensitivity experiments**:

   * Repeat runs varying one parameter at a time (e.g., reduce T2 by factor 2, or increase two-qubit fidelity by 0.1%) to produce sensitivity maps.
7. **Ablation**:

   * Turn off coherent errors, then turn off crosstalk, then remove quasi-static noise, etc., to compute error contribution.

---

# Statistical analysis & decision rule

* **Heavy-output probability (HOP)** per circuit: fraction of observed outcomes that are in the set of heavy outputs (those whose ideal probabilities exceed the median).
### Project Structure

```
spinq_qv_sim/
├── src/spinq_qv/           # Main package
├── examples/               # Campaign scripts + configs
├── notebooks/              # Jupyter tutorials (5 notebooks)
├── tests/                  # Unit + integration tests
├── docs/                   # Documentation
├── campaigns/              # Campaign outputs
├── Technicalities.md       # Physics blueprint
├── CHANGELOG.md            # Development history
└── pyproject.toml          # Package metadata
```

### Code Quality

- **Type hints**: All public APIs (mypy validated)
- **Pydantic schemas**: Configuration validation
- **Docstrings**: NumPy-style documentation
- **Linting**: Black + Flake8 (CI enforced)
- **Git hooks**: Pre-commit checks (optional)

### Contributing

1. **Fork & clone** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Install dev dependencies**: `pip install -r requirements_cpu.txt`
4. **Run tests**: `pytest tests/`
5. **Submit PR** with test coverage

---

## 🎓 Citation

If you use this simulator in research, please cite:

```bibtex
@software{spinq_qv_sim,
  title = {spinq_qv_sim: Quantum Volume Simulator for Si/SiGe Spin Qubits},
  author = {Da Ros, Alessandro},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/A-DaRo/spinq_qv_sim}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🔗 Related Projects

- **Qiskit Aer** - IBM's quantum simulator (inspiration for QV circuits)
- **Cirq** - Google's quantum framework
- **QuTiP** - Quantum Toolbox in Python (master equation solvers)
- **cuQuantum** - NVIDIA's GPU quantum simulators

---

## 🚀 Roadmap

### Current (v1.0.0) ✅
- Full QV pipeline with 3 backends
- Campaign system with resume capability
- Complete noise model (5 error sources)
- Publication-quality analysis

### Future (Optional)
- **GPU Acceleration**: cuQuantum integration, deterministic GPU RNG
- **Error Mitigation**: ZNE, readout correction
- **Advanced Benchmarks**: Randomized benchmarking, gate set tomography
- **Real Device Comparison**: Validation against experimental data
- **Circuit Optimization**: Gate compilation, dynamic decoupling

---

## 💬 Support

- **Documentation**: See `docs/` directory
- **Examples**: Run `python examples/test_campaign_system.py`
- **Issues**: [GitHub Issues](https://github.com/A-DaRo/spinq_qv_sim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/A-DaRo/spinq_qv_sim/discussions)

---

## 🙏 Acknowledgments

- **IBM Quantum** - QV benchmark methodology
- **QuTiP team** - Quantum simulation foundations
- **Si/SiGe community** - Experimental parameter data
- **TU/e IQT Group** - Project supervision

---

## 📊 Quick Reference

### Common Commands

```bash
# Basic QV run
python -m spinq_qv.experiments.run_qv --config baseline.yaml --widths 2-6

# Sensitivity analysis
python -m spinq_qv.experiments.run_qv --mode sensitivity-1d --param device.single_qubit_fidelity

# Campaign
python examples/run_parameter_campaign.py --base-config production.yaml --sweep-type comprehensive

# Validation (RB)
python -m spinq_qv.experiments.validate --config baseline.yaml

# Tests
pytest tests/unit/ -v
```

### Key Files

| File | Purpose |
|------|---------|
| `examples/configs/baseline.yaml` | Default device parameters |
| `examples/configs/production.yaml` | Full-scale QV experiment |
| `examples/run_parameter_campaign.py` | Campaign entry point |
| `notebooks/01_quickstart_qv_experiment.ipynb` | Tutorial |
| `docs/math_foundations.md` | Formula reference |

---

## 📋 Appendix: Original Project Blueprint

<details>
<summary><b>Click to expand:</b> Original implementation plan (historical reference)</summary>

### Metrics & outputs to save

For reproducibility save everything (use HDF5 / parquet + JSON metadata):

1. **Per-run raw files**:

   * `circuit_id`, `m`, `seed`, `mapping`, `transpiled_circuit` (text), `ideal_probs` (vector), `measured_counts` (dict), `HOP_i`, `noise_realization_params`.
2. **Aggregates**:

   * `HOP_mean`, `HOP_std`, `95%_CI_lower`, `95%_CI_upper`, `n_circuits`, `n_shots`.
3. **Validation outputs**:

   * RB/benchmark results that reproduce single and two-qubit fidelities.
4. **Sensitivity grid**:

   * Grid of parameters and corresponding QV or HOPs.
5. **Logs**:

   * Config YAML, commit hash, python package versions, random seeds, wall-clock CPU/GPU usage.

---

# Plots & visualizations (what to print)

1. **HOP vs width `m`**: line plot with error bands (95% CI). Mark threshold 2/3 and highlight passing widths.
2. **Per-circuit HOP distributions**: violin/box plots grouped by `m` showing spread across random circuits.
3. **Heavy-output histogram**: distribution of heavy-output counts for typical circuits.
4. **Error budget pie chart**: fraction of HOP degradation due to each error source (coherent errors, depolarizing, decoherence, SPAM, crosstalk), from ablation runs.
5. **Sensitivity heatmaps**: QV (or HOP) as function of two parameters (e.g., T2 vs two-qubit fidelity).
6. **Time schedule diagram**: example transpiled schedule with gate start times, showing parallelism and idles.
7. **1/f spectral representation**: (if modeled) show noise PSD and sampled quasi-static offsets distribution.
8. **Validation plots**: simulated RB decay curves and extracted fidelities compared to target.

All figures should be saved as PNG + PDF, and the raw data (CSV/HDF5) saved alongside.

---

# Validation & unit/integration tests (must-do)

1. **Unit tests**:

   * For conversion functions (`F -> p_dep`, `T -> p_amp/p_phi`), test against known examples.
   * Kraus channel builder unit tests: trace-preserving and positive.
2. **Integration tests**:

   * Verify that simulated single-qubit RB sequence yields `F1 ± tolerance` given the built noise model (adjust tolerances to account for finite sample effects).
   * Verify two-qubit benchmarking likewise.
3. **Sanity tests for QV harness**:

   * Run a noiseless QV simulation and confirm HOP ~ 1 (should be near 1 because heavy outputs are favored).
   * Run an extremely noisy depolarizing baseline and confirm HOP ≈ 0.5.

---

# Recommended simulation parameter defaults & resource guidance

* `n_circuits_per_m`: 50 (recommended), up to 200 for high confidence.
* `n_shots`: 1000 per circuit (balance between shot noise and compute). 1000 shots yield binomial SD ≈ sqrt(p*(1-p)/1000) ~ 0.016 at p≈0.5.
* If using density matrix: limit `m` ≤ 12–14 on workstation; for `m>12` consider statevector + stochastic unraveling or tensor-network.
* GPU acceleration recommended for statevector simulations of `m>12`.
* Logging: track runtime per circuit for profiling; parallelize circuits across CPU cores/GPUs.

---

# Sensitivity & ablation plan (detailed experiments)

For publication-grade analysis run the following:

1. **Baseline run**: all errors on with reported values — determine QV.
2. **Fidelity sweep**:

   * Increase single- and two-qubit fidelities in small steps (e.g., +0.05%, +0.1%, +0.5%) to find sensitivity.
3. **T2 sweep**:

   * Run runs with T2 = {20 µs, 99 µs (baseline), 500 µs, 3.1 ms} to show effect of decoupling.
4. **Crosstalk toggle**:

   * Set microwave crosstalk α_ij = 0 and compare.
5. **Quasi-static noise toggle**:

   * Turn off 1/f/quasi-static offsets; measure HOP change.
6. **Readout SPAM impact**:

   * Vary readout fidelity from 98% to 99.97% and show effect.
7. **Connectivity experiments**:

   * Compare linear chain mapping vs all-to-all (idealized) to show connectivity effect on parallelism and QV.
8. **Coherent vs stochastic error swap**:

   * Replace coherent errors with equivalent depolarizing strength (pauli-twirled) to show impact on QV (coherent errors often worse at depth).

Produce a small table summarizing for each ablation the change in the maximum passing `m`.

---

# Error-model caveats & interpretation guidance

1. **Depolarizing vs coherent errors**: depolarizing approximations may over- or under-estimate the true depth sensitivity; coherent errors can be far more damaging to QV.
2. **Non-Markovian noise**: quasi-static low-frequency noise can bias the HOP significantly in real devices — include it to be realistic.
3. **Connectivity & transpilation**: poor connectivity (linear arrays) greatly limits parallel two-qubit gates and reduces achievable depth; always simulate with the actual connectivity graph.
4. **Leaked/sticky errors**: leakage to valley states or slow thermalization affects later circuits if you model device resets incorrectly. Simulate reset behavior explicitly (measurement-based reset or thermal reset).
5. **Shot noise & number of circuits**: the statistical pass/fail is sensitive to the number of circuits and shots; use bootstrap to ensure robust decisions.
6. **Simulator accuracy**: some simulator approximations (stabilizer approximations, heavy twirling) can bias results; document approximations.

---

# Reproducibility & best practices

* Save the exact commit hash of your code repository and the config file with every run.
* Fix RNG seeds and log them; but also run multiple independent seeds to ensure stability.
* Use structured data outputs (HDF5/parquet) with metadata fields: sim_backend, sim_version, qubit_layout, noise_key.
* Create a “run report” PDF automatically combining key figures and tables.
* If comparing to experiments, use exactly the same transpilation and gate set as the hardware vendor uses.

---

# Suggested Python libraries & tools (implementation accents)

* Circuit generation / QV harness: Qiskit (has QV circuit recipes and transpiler tools), Cirq (custom QV circuits), or custom generator.
* Noise & channels: Qiskit Aer (noise model + density matrix), QuTiP (Kraus and master equation), pyGSTi (for benchmarking tools).
* High-performance simulation: qsim, cuStateVec, or additional GPU-accelerated libraries if available.
* Data handling: pandas, h5py, pyarrow/parquet.
* Plotting: matplotlib (one plot per figure; do not rely on seaborn).
* Statistical bootstrap: scipy / numpy.
* Reproducibility: pyyaml for configs.
  (Use whichever stack your team uses; the plan is library-agnostic.)

---

# Example deliverables (what you will get when implemented)

1. `results/QV_baseline.h5` — raw per-circuit and per-shot data.
2. `reports/QV_summary.pdf` — summary report with HOP vs m, QV determination, sensitivity figures.
3. `plots/HOP_vs_m.png`, `plots/error_budget.png`, `plots/sensitivity_heatmap.png`.
4. `configs/baseline.yaml` — exact simulation parameters and noise model.
5. `logs/run_YYYYMMDD_HHMMSS.log` with package versions and random seeds.
6. `notebooks/analysis.ipynb` — interactive notebook to reproduce plots.

---

# Quick numeric-check examples (for implementer)

(These are the literal conversion formulas you should code and the numerical baseline values to check against.)

* `p1 = 2*(1 - F1)` → with `F1 = 0.99926` → `p1 ≈ 0.00148` (0.148%).
* `p2 = 4/3*(1 - F2)` → with `F2 = 0.998` → `p2 ≈ 0.0026667` (0.2667%).
* Amplitude damping for τ=60 ns, T1=1 s: `p_amp ≈ 1 - exp(-60e-9 / 1) ≈ 6.0e-8` (negligible).
* Dephasing for τ=60 ns, T2=99 µs: `p_phi ≈ 1 - exp(-60e-9 / 99e-6) ≈ 0.0006059` (≈ 0.0606%).

(Implement these exact computations in the noise-model builder and log the results.)

---

# Recommended first implementation milestones (no time estimates provided)

* Implement config loader and parameter conversions (F→p, T→p).
* Implement QV circuit generator and ideal heavy-output calculation.
* Implement single-qubit and two-qubit Kraus channels, schedule, and a small density-matrix simulator for `m` ≤ 6.
* Validate by reproducing the reported single/two qubit fidelities via RB-like tests.
* Run baseline QV for `m ∈ {2,3,4,5,6}` and inspect HOPs and debug.
* Extend to full planned sweeps and sensitivity maps.

---

# Final notes & interpretation guidance for the user

### Metrics & outputs to save

For reproducibility save everything (use HDF5 / parquet + JSON metadata):

1. **Per-run raw files**:
   * `circuit_id`, `m`, `seed`, `mapping`, `transpiled_circuit` (text), `ideal_probs` (vector), `measured_counts` (dict), `HOP_i`, `noise_realization_params`.
2. **Aggregates**:
   * `HOP_mean`, `HOP_std`, `95%_CI_lower`, `95%_CI_upper`, `n_circuits`, `n_shots`.
3. **Validation outputs**:
   * RB/benchmark results that reproduce single and two-qubit fidelities.
4. **Sensitivity grid**:
   * Grid of parameters and corresponding QV or HOPs.
5. **Logs**:
   * Config YAML, commit hash, python package versions, random seeds, wall-clock CPU/GPU usage.

### Plots & visualizations (what to print)

1. **HOP vs width `m`**: line plot with error bands (95% CI). Mark threshold 2/3 and highlight passing widths.
2. **Per-circuit HOP distributions**: violin/box plots grouped by `m` showing spread across random circuits.
3. **Heavy-output histogram**: distribution of heavy-output counts for typical circuits.
4. **Error budget pie chart**: fraction of HOP degradation due to each error source (coherent errors, depolarizing, decoherence, SPAM, crosstalk), from ablation runs.
5. **Sensitivity heatmaps**: QV (or HOP) as function of two parameters (e.g., T2 vs two-qubit fidelity).
6. **Time schedule diagram**: example transpiled schedule with gate start times, showing parallelism and idles.
7. **1/f spectral representation**: (if modeled) show noise PSD and sampled quasi-static offsets distribution.
8. **Validation plots**: simulated RB decay curves and extracted fidelities compared to target.

All figures should be saved as PNG + PDF, and the raw data (CSV/HDF5) saved alongside.

</details>

---

**Built with ❤️ for the quantum computing community**

*Last updated: October 2025*
