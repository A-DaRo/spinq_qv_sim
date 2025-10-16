# Iterative development plan — CPU-first, GPU optional later

Below is a concrete, step-by-step plan to develop the full Quantum Volume simulation pipeline you asked for. Each iteration specifies **what to build**, **which files/modules to touch**, **unit/integration tests to add**, **expected outputs / acceptance criteria**, **risk points**, and a short **next-step**. The plan assumes the repository layout we designed earlier and is intentionally implementation-friendly (file names, function/class targets, data artifacts).

I do **not** include time estimates — instead I give *scope and success criteria* so you can organize sprints, PRs, or tasks to your cadence.

---

# Iteration 0 — Project bootstrap & environment (foundation)

**Goal:** Create repository skeleton, CI, basic docs and reproducible environments. Ensure developers can run unit tests locally and CI runs.

**What to implement**

* Repo skeleton (README, LICENSE).
* `pyproject.toml` or `setup.cfg` and `requirements.txt`.
* `environment_cpu.yml` (CPU dev env).
* Minimal `Dockerfile` (CPU image).
* `.github/workflows/ci.yaml` running: lint, unit tests, type checks.
* Create top-level `src/spinq_qv` package with `__init__.py`.
* Basic logger setup `src/spinq_qv/utils/logging_setup.py`.
* `defaults.yaml` under `src/spinq_qv/config/` with placeholder fields.
* `tests/unit/test_repo_smoke.py` that imports package and checks config loads.

**Tests**

* CI should run and pass lint + a single smoke test.
* `pytest` local run must pass the smoke test.

**Acceptance criteria**

* CI passes on PR that adds this iteration.
* `pip install -e .` and `python -c "import spinq_qv"` work locally.
* `defaults.yaml` validates against `schemas.py` stub (even if trivial).

**Risks & mitigations**

* Environment pin conflicts — keep `environment_cpu.yml` conservative; only core libs (numpy, pytest, pydantic).
* CI failures due to OS differences — run minimal tests in CI.

**Next step**

* Implement core config/schema and conversion utilities.

---

# Iteration 1 — Configs, RNG, and small utilities

**Goal:** Fully implemented configuration schema, seedable RNG manager, logging & structured outputs. Basic config-driven run harness skeleton.

**What to implement**

* `src/spinq_qv/config/schemas.py` (pydantic `Config` classes).
* `src/spinq_qv/config/defaults.yaml` filled with device defaults (use your provided numbers).
* `src/spinq_qv/utils/rng.py` to create reproducible `numpy.random.Generator` instances from a global seed.
* `src/spinq_qv/utils/logging_setup.py` (JSON logs).
* CLI skeleton `src/spinq_qv/experiments/run_qv.py` that loads config and prints a short JSON summary.
* Unit tests:

  * `tests/unit/test_config_validation.py`: invalid values rejected.
  * `tests/unit/test_rng_repeatability.py`: same seed -> same draws.

**Tests**

* Config parsing valid/invalid cases.
* RNG determinism test.

**Acceptance criteria**

* `run_qv.py --config examples/configs/baseline.yaml` prints identical JSON summary for same seed.
* Unit tests pass in CI.

**Risks**

* Overly strict validation causing friction — keep defaults permissive but validated.

**Next step**

* Start implementing circuit generation and ideal (noiseless) heavy-output calculation.

---

# Iteration 2 — QV circuit generator & ideal-output calculator

**Goal:** Produce IBM-style QV circuits and compute exact ideal output probabilities (no noise). Save circuits as serializable specs.

**What to implement**

* `src/spinq_qv/circuits/generator.py`:

  * `generate_qv_circuit(m, seed) -> CircuitSpec`
  * `compute_ideal_probabilities(circuit_spec) -> np.ndarray`
* `src/spinq_qv/io/formats.py` basic circuit JSON schema.
* Small notebook `notebooks/tutorial_qv.ipynb` illustrating generation and ideal-probabilities plot.
* Unit tests:

  * `tests/unit/test_qv_generator.py`: deterministic circuits from seed; probabilities sum to 1; heavy-output set size equals 2^(m-1).
* Integration test:

  * `tests/integration/test_noiseless_qv_hop.py`: for `m=2,3` with a few random circuits, simulated ideal HOP should be exactly 1 (noiseless heavy-output sampling—since ideal samples always produce the heavy outputs in expectation; more precisely: verify median split and that heavy set defined correctly).

**Tests**

* Determinism, correctness of probability computations, heavy-output set computation.

**Acceptance criteria**

* Circuits saved as JSON with canonical fields.
* Notebook demonstrates ideal HOP computation and saves figure `plots/ideal_prob_example.png`.

**Risks**

* The exact Haar-random two-qubit unitary decomposition for ideal circuits can be heavy; implement a simple unitary-sampling method (random single-qubit rotations + random entangling layers) that matches IBM QV spirit.

**Next step**

* Implement statevector simulator backend and verify noiseless sampling matches ideal probabilities.

---

# Iteration 3 — Pure-state statevector simulator (CPU path)

**Goal:** Implement CPU statevector backend able to apply native gates and sample measurement outcomes. Run noiseless QV circuits to generate sampled HOPs.

**What to implement**

* `src/spinq_qv/sim/backend.py`: abstract base class.
* `src/spinq_qv/sim/statevector.py`: implement `StatevectorBackend` with:

  * `init_state(n_qubits)`
  * `apply_unitary(unitary, targets)`
  * `measure(shots, readout_noise=None)`
  * Efficient bitstring sampling
* `src/spinq_qv/experiments/run_qv.py` integration so that it can run a small set of circuits with `statevector` backend.
* Unit tests:

  * `tests/unit/test_statevector_basic.py`: apply X on qubit 0 and measure; statistics match.
* Integration tests:

  * `tests/integration/test_noiseless_sampling.py`: generate 10 circuits m=3, sample 1000 shots each — compare empirical frequencies to ideal probabilities (Kolmogorov-Smirnov or chi-squared within tolerance).

**Tests**

* Basic gate correctness, measurement sampling accuracy.

**Acceptance criteria**

* Statevector backend returns empirical distributions consistent with ideal probabilities within statistical shot noise bounds.
* Noiseless HOPs for circuits show expected high values (sanity check).

**Risks**

* Large-memory for statevector for m>24; but initial plan uses m ≤ 12 in early runs.

**Next step**

* Implement channels primitives (Kraus) and noise model builder.

---

# Iteration 4 — Noise primitives & NoiseModelBuilder (CPU)

**Goal:** Implement Kraus/Pauli/depolarizing/amplitude-damping/phase-damping channels and the `NoiseModelBuilder` that converts config numbers into per-gate channels.

**What to implement**

* `src/spinq_qv/noise/channels.py` with exact Kraus forms (amplitude damping, phase damping, depolarizing).
* `src/spinq_qv/noise/stochastic.py` quasi-static sampling (map T2* to detuning sigma).
* `src/spinq_qv/noise/coherent.py` small coherent rotations and residual ZZ implementation.
* `src/spinq_qv/noise/builder.py`:

  * Methods to compute `p_dep` from fidelities,
  * Compute `p_amp` and `p_phi` from gate durations and T1/T2,
  * Compose channels: e.g., depolarizing ∘ amplitude-damping ∘ coherent unitary (decide ordering and document).
* Unit tests:

  * `tests/unit/test_kraus_trace_preserving.py`: Kraus sums preserve trace (within tol).
  * `tests/unit/test_depolarizing_conversion.py`: F→p conversion returns known numbers (compare to hand-calculated p1, p2).
  * `tests/unit/test_quasi_static_sigma.py`: sample detunings reproduce variance implied by T2* formula.

**Tests**

* Validations ensure channels are physically valid and conversions match formulas.

**Acceptance criteria**

* `NoiseModelBuilder.build()` returns serializable summary with per-gate numeric values and channels.
* Unit tests pass.

**Risks**

* Numerical instability in Kraus ops for tiny gammas — use stable implementations (sqrt(1-gamma) etc).

**Next step**

* Integrate noise channels into simulation: first via Pauli-twirled depolarizing approximations in statevector (stochastic drops), then exact Kraus via density matrix later.

---

# Iteration 5 — Integrate noise into simulation (stochastic unraveling on statevector)

**Goal:** Add noise by stochastic Pauli application or MCWF-style jumps using the statevector backend. Validate by reproducing reported single- and two-qubit fidelities via RB simulations.

**What to implement**

* In `statevector.py`: implement `apply_channel_as_stochastic_moves(kraus, repeats)` or add `apply_pauli_error` for twirled depolarizing.
* Implement `src/spinq_qv/experiments/validate.py`:

  * Small RB sequences for single-qubit and two-qubit gates
  * Extract average gate fidelity from decay and compare with configured F1/F2.
* Tests:

  * `tests/integration/test_rb_reproduction.py`: run RB (short sequences, few seeds) and check extracted fidelities are within tolerance (e.g., ±0.5% absolute) of target F1/F2.
* Logging: output RB fits CSV to `results/validate_rb.csv`.

**Tests**

* RB-style test reproduces configured fidelities within tolerance.

**Acceptance criteria**

* RB extracted fidelities match configured F1/F2 within defined tolerances.
* Unit/integration tests pass in CI.

**Risks**

* Converting average fidelity to exact depolarizing probabilities loses coherent error modelling — document this limitation and ensure later iterations introduce coherent channels.

**Next step**

* Implement density-matrix simulator for exact non-unitary channels (for smaller m).

---

# Iteration 6 — Density-matrix simulator & MCWF fallback (CPU)

**Goal:** Add exact density-matrix propagation (Kraus-based) for small qubit numbers and MCWF for larger numbers as approximate density evolution.

**What to implement**

* `src/spinq_qv/sim/density_matrix.py` implementing:

  * `apply_kraus(kraus, targets)`
  * `propagate_sequence(scheduled_circuit, noise_model)` returning density matrix or measurement probabilities.
* `src/spinq_qv/sim/mcwf.py` implementing quantum trajectories with choice of number of trajectories.
* Integration in `run_qv.py` with flags `backend=density` or `backend=mcwf`.
* Tests:

  * `tests/unit/test_density_kraus_equivalence.py`: for one-qubit channels, density-matrix propagation equals explicit Kraus application.
  * `tests/integration/test_mcwf_vs_density.py`: for small n and enough trajectories, MCWF average approximates density-matrix results within tolerance.
* Acceptance criteria:

  * Density-matrix method yields measurement probabilities consistent with applied Kraus channels.
  * MCWF approximates density matrix as number of trajectories increases.

**Risks**

* Memory blow-up for density-matrix at n > 14 — enforce safety checks and warn users.

**Next step**

* Implement full QV runner that uses noise model and either density or MCWF to compute HOPs.

---

# Iteration 7 — Full QV runner (CPU baseline) + storage & analysis

**Goal:** Implement the full experimental loop on CPU: transpile, schedule, simulate with noise, compute per-circuit HOP, aggregate, and decide QV. Add robust storage (HDF5) and plotting.

**What to implement**

* `src/spinq_qv/circuits/transpiler.py` and `scheduling.py` (basic mapping for linear connectivity).
* `src/spinq_qv/experiments/run_qv.py` fully functional: loops over `m`, generates circuits, transpiles, schedules, simulates, measures, computes HOPs, persists to HDF5 via `io/storage.py`.
* `src/spinq_qv/analysis/hop.py` and `stats.py` for bootstrap CI and decision rule.
* `src/spinq_qv/io/storage.py` for HDF5 output and `load_run` utility.
* Plotting scripts in `src/spinq_qv/analysis/plots.py`.
* Tests:

  * `tests/integration/test_qv_pipeline_small.py`: run pipeline for m∈{2,3,4}, n_circuits=10, shots=500, verify outputs saved and aggregated HOP in plausible range [0.33, 1.0].
* Acceptance criteria:

  * Pipeline produces `results/run_<timestamp>.h5` with expected structure.
  * HOP vs m plot produced and looks reasonable.
  * Bootstrap CI runs and decision rule outputs documented.
* Deliverables:

  * `examples/sample_outputs/demo_results.h5`.
  * `reports/` sample PDF/markdown summary of baseline run.

**Risks**

* Long runtimes for many circuits — provide sample configs with reduced circuits/shots for local dev; production configs for full runs.

**Next step**

* Add ablation/sensitivity module and automation for parameter sweeps.

---

# Iteration 8 — Sensitivity, ablation studies & reporting

**Goal:** Implement automated param sweeps and ablation experiments; produce sensitivity heatmaps and error-budget breakdowns.

**What to implement**

* `src/spinq_qv/analysis/ablation.py` implementing:

  * Single-parameter sweeps (T2, F2, crosstalk magnitude).
  * Pairwise sweeps for 2D heatmaps.
  * Ablation toggles to switch off coherent errors, crosstalk, quasi-static noise, etc.
* `src/spinq_qv/experiments/sensitivity.py`: high-level driver that creates grid runs and aggregates outputs.
* Plotting (sensitivity heatmaps, error budget pie charts).
* Tests:

  * `tests/integration/test_ablation_small.py`: run ablation with small parameter grid and verify outputs exist and are consistent.
* Acceptance criteria:

  * Ability to run a 4×4 param grid and produce CSV + heatmap.
  * Error-budget computed by pairwise subtraction of HOPs for ablation cases.

**Risks**

* Explosion of simulations; ensure orchestration supports checkpointing/resuming.

**Next step**

* Performance optimization and parallelization.

---

# Iteration 9 — Parallelization & performance optimization (CPU)

**Goal:** Make the CPU pipeline faster and scalable: parallelize circuits across cores, provide per-circuit profiling, and tune memory usage.

**What to implement**

* Use `concurrent.futures.ProcessPoolExecutor` or `joblib` to parallelize independent circuit simulations.
* `src/spinq_qv/utils/perf.py` profiling decorators and resource logging (peak memory, time).
* Implement graceful job restart / checkpointing in `run_qv.py`.
* Add CI performance smoke test that measures runtime for a tiny run.
* Acceptance criteria:

  * Parallel runs (e.g., 4 workers) produce consistent results vs serial run.
  * Profiling output shows time per circuit and memory per-worker.
* Risks:

  * Race conditions writing to HDF5 — use per-worker temporary files and merge, or thread-safe queue writer.

**Next step**

* Optional: GPU backend integration (if you want acceleration).

---

# Iteration 10 — GPU backend integration (optional, requirement-based)

**Goal:** Add GPU-accelerated backends for statevector/density ops (CuPy, cuStateVec, qsim etc.) and allow massive-scale runs for larger `m` and deeper sweeps.

**Preconditions**

* Developer has a CUDA-enabled machine and installs `environment_gpu.yml`.
* Decide which GPU library to use (CuPy for generic linear algebra; cuStateVec/cuQuantum for specialized speed).

**What to implement**

* `src/spinq_qv/sim/statevector.py` extended with GPU implementation (detect CuPy availability).
* GPU-enabled density ops if cuQuantum or similar present; else use GPU-accelerated MCWF with CuPy matrix ops.
* GPU selection logic in `sim/backend.py` with explicit `backend=gpu_statevector` or `gpu_density`.
* GPU tests:

  * `tests/integration/test_gpu_smoke.py` run on designated GPU runner only (not in general CI).
  * Determinism tests for GPU RNG.
* Acceptance criteria:

  * GPU backend reproduces CPU outputs within numeric tolerance for the same seed.
  * Performance improvement is measured (e.g., wall-time per circuit for a given `m` shows significant speedup).
* Risks:

  * Non-deterministic GPU BLAS results; mitigate via environment flags and documented deterministic settings.
  * Dependency headaches for cuQuantum; encapsulate optional GPU code behind well-defined interfaces and fall back gracefully.

**Next step**

* Run a final full-scale QV/ablation campaign and produce publication-ready report.

---

# Iteration 11 — Final full campaigns, documentation, and release

**Goal:** Use the pipeline to run production-quality simulations, produce a final report, archive data, and tag a release.

**What to implement / produce**

* Run full baseline QV campaign across m up to chosen max (e.g., m=12 depending on resources).
* Produce `reports/QV_summary.pdf` with:

  * HOP vs m plots and CI,
  * sensitivity heatmaps,
  * RB validation plots,
  * error-budget summary and recommended hardware improvements to increase QV.
* Finalize `docs/math_foundations.md` and code documentation (docstrings, Sphinx docs).
* Tag repo, publish release artifact with `results/` HDF5 and plots.
* Tests:

  * Full integration tests as part of release checklist; validate run metadata and reproducibility.
* Acceptance criteria:

  * Final report generated and artifacts saved in release.
  * Reproducible run: `run_qv.py --config configs/baseline.yaml --resume` reproduces same results from stored metadata.

**Risks & mitigations**

* Large storage: compress HDF5, store only aggregated data if necessary.
* Reproducibility drift: enforce pinned environments (`environment_gpu.yml` and `environment_cpu.yml`) and record package versions in run metadata.

---

# Additional quality steps (applies across iterations)

* **Code review & PRs:** Each iteration implemented as one or more small PRs; require unit tests and CI pass before merge.
* **Branching policy:** `main` protected; feature branches per iteration.
* **Documentation:** For each module implemented, add a paragraph to `docs/` explaining math and implementation details. Keep `math_foundations.md` in sync.
* **Logging & reproducibility:** Always save `git commit hash`, `config.yaml`, `seed`, and `package versions` into the `results/*.h5` metadata.
* **API stability:** Keep `NoiseModelBuilder` and `Backend` interfaces stable once iterations 1–6 are merged.
* **Performance telemetry:** Save `perf/*` logs per run (time per circuit, memory footprint) to guide optimization.

---

# Tests and acceptance summary table

A compact checklist you can use to mark progress for each iteration:

* Iter 0: repo + CI smoke — **Accept** if CI green.
* Iter 1: configs + RNG — **Accept** if deterministic RNG tests pass.
* Iter 2: QV generator + ideal probs — **Accept** if heavy-output set computed correctly.
* Iter 3: statevector backend — **Accept** if noiseless sampling matches ideal probabilities (statistical tolerance).
* Iter 4: noise primitives — **Accept** if Kraus ops preserve trace and fidelity→depolarizing conversions match expected numbers.
* Iter 5: stochastic-noise integration & RB — **Accept** if RB recovers configured fidelities within ±0.5% abs.
* Iter 6: density matrix & MCWF — **Accept** if density/MCWF results are consistent.
* Iter 7: full QV runner CPU — **Accept** if run produces HDF5 with full schema and HOP aggregation; plots exist.
* Iter 8: ablation/sensitivity — **Accept** if heatmaps + error-budget outputs saved and reproducible.
* Iter 9: parallelization — **Accept** if parallel runs yield same results as serial and time per circuit reduced.
* Iter 10: GPU backend (optional) — **Accept** if GPU reproduces CPU within tolerance and shows speedup.
* Iter 11: final reports & release — **Accept** if full run artifacts created and documented.

---

# Example small-run parameters for iterative testing (recommended)

Use these small numbers during dev so CI and developers can run quickly:

* `m` test set: `{2, 3, 4}` for unit & integration tests.
* `n_circuits`: 5–20 (dev), 50–200 (production).
* `n_shots`: 200–1000 (dev), 1000–5000 (production).
* Backend: `statevector` (dev); `density` only for `m ≤ 12` in CI; `mcwf` for larger `m`.

---

# Final notes and handoff

* This plan is intentionally modular: you can stop after any iteration and have a working artifact (e.g., after Iteration 7 you already have a credible CPU-only QV estimator).
* If you’d like, I can now produce either:

  1. A prioritized task list (Jira-style) derived from these iterations, or
  2. The exact function/class signatures and minimal docstrings for the key classes (`NoiseModelBuilder`, `Backend`, `Transpiler`, `CircuitSpec`) so a developer can start coding immediately.

Which of those do you want next?
