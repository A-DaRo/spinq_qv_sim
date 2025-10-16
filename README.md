below is a complete, **extensive, ready-to-implement plan** (natural language) for estimating a **simulated Quantum Volume (QV)** for a Si/SiGe spin-qubit device using the numbers you provided. It is written as an engineering blueprint: what modules to build, how to model noise (with formulas), what experiments to run, what to save, and what graphs and statistical tests to produce. No code is included — but everything is specific enough that a developer can implement it in Python (Qiskit/Cirq/QuTiP/QuEST/Stim/etc.) immediately.

# Overview & goals

1. Goal: determine the largest square width `m` (number of qubits = circuit width) for which the IBM-style Quantum Volume test (random square circuits of width = depth = m) yields a **heavy-output probability (HOP)** above the required threshold (standard target: > 2/3) with high statistical confidence. This `m` is the simulated QV (QV = 2^m).
2. Use the experimental Si/SiGe numbers you provided to build a physically realistic noise model (coherent + stochastic + non-Markovian components) and then run Monte Carlo simulations of randomized QV circuits, including SPAM, connectivity, and timing.
3. Produce reproducible outputs: CSV/HDF5 of raw result per circuit and shot, aggregate statistics, figures (HOP vs. width, error budget pie charts, sensitivity maps), and a final QV determination with confidence intervals.

---

# High-level pipeline (module list)

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
* **Aggregate HOP for width m**: mean over `n_circuits` of circuit HOPs.
* **Uncertainty estimation**: use bootstrap resampling (resample circuits with replacement, and within each circuit resample shots or use Binomial Clopper-Pearson for shot noise) to derive 95% confidence intervals for `HOP(m)`.
* **Pass criterion**: Classical IBM QV criterion is `HOP > 2/3`. Use two variants:

  * *Strict:* lower 95% confidence interval bound > 2/3.
  * *Practical:* mean HOP > 2/3 and p-value for null hypothesis `HOP ≤ 2/3` < 0.05 (one-sided test).
* **QV assignment**: QV = 2^m where `m` is the largest width that passes. Report both strict and practical results.

---

# Metrics & outputs to save

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

* The final simulated QV is an estimate that depends strongly on how you model coherent errors, quasi-static noise, connectivity and SPAM. You will get a **range** rather than a single exact number unless you have extremely accurate system-characterization data.
* The plan includes ablation and sensitivity steps so that when you report a simulated QV you can also say *which error sources are dominant* and how much improvement (e.g., improving two-qubit fidelity by X) would be required to increase QV by one bit.

---
