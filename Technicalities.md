Great — below is a **concrete, implementation-ready repository layout** plus a full description of every module, the interfaces they expose, exact file formats, refined GPU/CPU dependency recommendations (with optional acceleration paths), and the mathematics you must implement (with precise equations and algorithm choices). This is written so an engineer can start implementing immediately.

I organize this as:

1. repository tree (files and folders)
2. description of each top-level module and file (responsibilities, inputs/outputs, data formats)
3. dependency matrix (CPU path vs GPU-accelerated path; recommended versions)
4. runtime / CI / deployment notes (Docker, envs, GPU runners)
5. mathematical foundations and formulas you must implement (gate-to-channel conversions, decoherence channels, master-equation options, quasi-static noise, heavy-output test and statistics, decision rules)
6. numerical / precision / performance recommendations and testing checklist

---

# 1 — Repository tree (suggested)

```
spinq_qv_sim/                      # root repo
├── README.md
├── pyproject.toml                 # or setup.cfg + requirements.txt
├── environment.yml                # conda env (GPU and CPU variants)
├── Dockerfile                     # CUDA-base image + dependencies (optional)
├── .github/
│   └── workflows/
│       └── ci.yaml                # CI for lint+unit tests (CPU)
├── docs/                          # design docs, math notes, examples
│   └── math_foundations.md
├── src/
│   └── spinq_qv/                  # python package
│       ├── __init__.py
│       ├── config/                # configuration and metadata
│       │   ├── defaults.yaml
│       │   └── schemas.py         # pydantic dataclasses for config validation
│       ├── circuits/
│       │   ├── generator.py       # QV circuit generator
│       │   ├── transpiler.py      # mapping + native decomposition
│       │   └── scheduling.py      # gate scheduling + parallelism
│       ├── noise/
│       │   ├── builder.py         # build composite noise model
│       │   ├── channels.py        # Kraus and channel factory functions
│       │   ├── coherent.py        # coherent error models (ZZ, misrot)
│       │   └── stochastic.py      # 1/f, OU, quasi-static sampling
│       ├── sim/
│       │   ├── backend.py         # abstract simulator interface
│       │   ├── statevector.py     # pure state simulator (CPU/GPU)
│       │   ├── density_matrix.py  # density-matrix simulator (Kraus)
│       │   ├── mcwf.py            # Monte-Carlo wavefunction trajectories
│       │   └── tensornet.py       # optional tensor network wrapper
│       ├── analysis/
│       │   ├── hop.py             # heavy-output computation & HOP stats
│       │   ├── stats.py           # bootstrap, CI, tests
│       │   └── ablation.py        # ablation & sensitivity pipelines
│       ├── io/
│       │   ├── storage.py         # HDF5/parquet writer/reader
│       │   ├── formats.py         # canonical data schemas
│       │   └── serializers.py
│       ├── utils/
│       │   ├── logging_setup.py
│       │   ├── rng.py             # seeded RNG manager
│       │   └── perf.py            # profiling wrappers
│       └── experiments/
│           ├── run_qv.py          # top-level experiment runner CLI
│           ├── validate.py        # RB & fidelity reproduction tests
│           └── sensitivity.py
├── notebooks/
│   ├── tutorial_qv.ipynb
│   └── exploratory.ipynb
├── tests/
│   ├── unit/
│   └── integration/
└── examples/
    ├── configs/
    │   ├── baseline.yaml
    │   └── high_fidelity.yaml
    └── sample_outputs/
        └── demo_results.h5
```

---

# 2 — Module-by-module descriptions (concrete)

Below I expand what each file/module must contain, the function/class-level interfaces, expected inputs and outputs, and file formats.

## `src/spinq_qv/config/`

* **`defaults.yaml`**

  * Human-editable default configuration including device params (F1, F2, gate times, T1, T2, T2*, readout fidelities, connectivity), simulation parameters (m widths to test, n_circuits, n_shots), and noise toggles.
  * Example fields: `device.qubits=12`, `device.connectivity="linear"`, `device.F1=0.99926`, `device.F2=0.998`.

* **`schemas.py`**

  * Pydantic (or dataclasses) definitions for config validation.
  * Expose `Config` class with `.from_yaml(path)` and `.to_dict()`; throw on invalid ranges.

## `src/spinq_qv/circuits/generator.py`

* Responsibility: implement IBM QV random-square circuits generator.
* Exposed functions/classes:

  * `generate_qv_circuit(m: int, seed: int, gate_set: str = "native") -> CircuitSpec`

    * `CircuitSpec` includes: logical qubit count `m`, gate list in high-level form (random single-qubit rotations and random two-qubit unitaries), and the "ideal" unitary/ideal output amplitudes computeable from spec.
  * `compute_ideal_probabilities(circuit_spec) -> np.ndarray` (length `2**m`, complex amplitudes or probabilities)
* Notes: produce deterministic circuits from seed; store circuit description (JSON serializable).

## `src/spinq_qv/circuits/transpiler.py`

* Responsibility: map logical circuits to physical qubits & native gates.
* Exposed:

  * `Mapper` class with strategies `linear`, `greedy`, `optimal` (optional).
  * `transpile(circuit_spec, mapping, native_gates, connectivity) -> TranspiledCircuit`

    * `TranspiledCircuit` contains ordered native gates with durations and target physical qubit indices.
* Important: must include function to insert SWAPs if needed and to report additional two-qubit gates added.

## `src/spinq_qv/circuits/scheduling.py`

* Responsibility: schedule gates to avoid physical conflicts and compute idles.
* Exposed:

  * `schedule(transpiled_circuit, parallelism=True) -> ScheduledCircuit`
  * Schedules assign start_time, end_time to each gate and produce per-qubit timelines.

## `src/spinq_qv/noise/channels.py`

* Responsibility: factory functions for standard quantum channels (Kraus operators and superoperators). Implement exact Kraus forms and also Pauli-twirled approximations.
* Exposed channel builders:

  * `depolarizing_channel(p: float, n_qubits: int = 1) -> Kraus`
  * `amplitude_damping_channel(gamma: float) -> Kraus` (single-qubit)
  * `phase_damping_channel(gamma_phi: float) -> Kraus`
  * `unitary_error(U: np.ndarray) -> Kraus` or `superop`
  * `two_qubit_zz_error(theta: float) -> Kraus`
* Data format for Kraus: standard numpy arrays; also support conversion to sparse superoperator.

## `src/spinq_qv/noise/coherent.py`

* Implement parameterized coherent errors:

  * Over/under rotation: `R(θ+ε)`
  * Systematic axis misalignment
  * Residual ZZ: implement `exp(-i θ ZZ/2)` generator
* Provide helpers to convert a fraction of average gate infidelity into an equivalent coherent rotation magnitude (optionally via Pauli twirling mapping).

## `src/spinq_qv/noise/stochastic.py`

* Implement low-frequency noise models:

  * Quasi-static Gaussian detuning per-qubit per-circuit: sample `Δ_i ∼ N(0, σ^2)` with relation to T2* (see math section).
  * Ornstein–Uhlenbeck process for time-dependent noise: parameters `τ_corr` and `σ`.
  * 1/f noise generation (approximate via sum of OU processes or spectral synthesis).
* API:

  * `sample_quasi_static_offsets(m, n_realizations, sigma) -> ndarray (n_realizations x m)`
  * `generate_ou_noise(...) -> time-series`

## `src/spinq_qv/noise/builder.py`

* Responsibility: compose full per-gate noise model given device config.
* Exposed:

  * `NoiseModelBuilder(config)` with `.build()` returning an object that can produce the appropriate channel for a given gate (type, duration, targets).
  * Provide `to_dict()` summary (for logging) that lists per-gate p_dep, p_amp, p_phi, coherent-angle, crosstalk map, leakage rates.

## `src/spinq_qv/sim/backend.py`

* Abstract base class interface for simulation backends. Methods:

  * `init_state(n_qubits)`
  * `apply_unitary(unitary, targets, time=None)`
  * `apply_kraus(kraus, targets, time=None)`
  * `measure(shots, readout_noise=None) -> counts dict`
  * `copy()` for snapshotting state if needed.
* Backends must implement deterministic seeding for reproducibility.

## `src/spinq_qv/sim/statevector.py` (CPU/GPU)

* Implementation of pure-state simulation:

  * CPU: numpy with complex128
  * GPU: optional cuStateVec / cuQuantum / JAX / CuPy-based implementation
* Features:

  * Fast unitary application, measurement sampling, support for simultaneous small unitaries (for crosstalk).
  * Support for stochastic unraveling: apply non-unitary channels via Monte Carlo jumps (MCWF) by sampling jump probabilities.

## `src/spinq_qv/sim/density_matrix.py`

* Exact density-matrix propagation:

  * Propagate `ρ` under Kraus channels or by integrating Lindblad master equation over gate durations.
  * Methods for composing sequential Kraus channels for each gate and idle.
* Limit: memory scales like `4^n`. Use sparse superoperators where possible.
* Provide option to use GPU linear algebra (CuPy, cuBLAS) for matrix multiply/reshape.

## `src/spinq_qv/sim/mcwf.py`

* Monte Carlo wavefunction (quantum trajectories) implementation:

  * Draw number of trajectories required to achieve target variance.
  * For each trajectory, propagate statevector with non-Hermitian effective Hamiltonian and stochastic jumps.
  * Average measurement outcomes across trajectories to approximate density-matrix results.

## `src/spinq_qv/sim/tensornet.py` (optional)

* Wrapper that uses a tensor-network simulator for larger m but shallow depth. Interface abstracts `simulate(circuit, noise_model)` and returns sampled counts or probabilities.

## `src/spinq_qv/analysis/hop.py`

* Compute heavy-output sets and HOP:

  * `heavy_outputs(ideal_probs) -> set_of_bitstrings` selects those with `prob > median`.
  * `compute_hop(measured_counts, heavy_set) -> float`
  * Save per-circuit HOP to results.

## `src/spinq_qv/analysis/stats.py`

* Statistical tests and CI:

  * Bootstrap circuits: `bootstrap_hop(hop_list, n_boot=2000) -> (mean, lower95, upper95)`
  * Clopper-Pearson CI for binomial shot-level uncertainty.
  * One-sided hypothesis test for `HOP > 2/3` (binomial/normal approximation).
* Exposed functions: `decide_pass(hop_list, alpha=0.05, method="bootstrap") -> (pass_bool, p_value, CI)`

## `src/spinq_qv/io/storage.py`

* HDF5-backed schema writer and reader; also support Parquet for tabular aggregates.
* Primary dataset layout (HDF5):

  * `/config` (JSON or YAML)
  * `/circuits/{m}/{circuit_id}/spec` (JSON)
  * `/circuits/{m}/{circuit_id}/ideal_probs` (float array)
  * `/circuits/{m}/{circuit_id}/measured_counts` (compressed)
  * `/results/aggregates` (table)
* Provide `save_experiment_run(path, metadata, raw_data)` and `load_run(path)` APIs.

## `src/spinq_qv/experiments/run_qv.py`

* CLI entrypoint that:

  * Loads config, builds noise model, chooses backend (auto-select GPU if available and config requests).
  * Validates noise model via RB (optional).
  * Loops over `m` widths generating, transpiling, scheduling, simulating circuits, computing HOPs, and writing results to storage.
  * Produces final pass/fail for each `m` and writes summary report (PDF or Markdown).
* Should include resume capability (checkpointing), and multi-process parallelization across circuits.

## `src/spinq_qv/experiments/validate.py`

* Reproduce single- and two-qubit fidelities with small RB sequences under the noise model to ensure the model reproduces reported fidelities. Save plots.

## `src/spinq_qv/utils/`

* `logging_setup.py`: structured logging (JSON logs), log levels, and file rotation.
* `rng.py`: centralized RNG seeded manager returning numpy RNGs per module.
* `perf.py`: measurement decorators for CPU/GPU time, memory, and optional profiling hooks.

## `notebooks/` and `examples/`

* Provide small runnable examples that load `baseline.yaml` and run a toy `m={2,3,4}` with reduced shots to demonstrate functionality.
* Provide a `tutorial_qv.ipynb` which demonstrates how to change noise parameters and re-run sensitivity tests.

---

# 3 — Dependency requirements (refined; GPU available)

Split into **required** (CPU path) and **optional / accelerated** (GPU path). Use pinned versions where possible.

## Core (required — CPU-only)

* Python >= 3.10
* numpy >= 1.24
* scipy >= 1.10
* pandas >= 2.0
* h5py >= 3.8
* pyarrow >= 12.0 (for parquet)
* matplotlib >= 3.7
* pydantic >= 1.10 (or `dataclasses` if preferred)
* pytest >= 7.2
* black, flake8 for lints
* (optional) qiskit == 0.45+ if you want to reuse QV circuit generator and transpiler components — but keep the core code independant so Qiskit is optional.

## Optional (GPU-accelerated and high-performance)

* CUDA toolkit >= 12.0 (match host GPU drivers)
* CuPy >= 12 (NumPy-compatible on GPU)
* NVIDIA cuQuantum (cuStateVec / cuTensorNet) — if available (specific version matching CUDA)
* qsim (GPU plugin) or qsimcirq for faster circuits (if you have Cirq)
* JAX with CUDA (jaxlib + jax) — if you choose JAX-based statevector ops (JAX supports GPU/TPU)
* numba >= 0.57 (for JIT CPU/GPU kernels)
* pybind11 / prebuilt bindings if using qsim native lib
* optional: pytorch >= 2.0 (can be used for tensor contractions on GPU, not required)

## Recommended packaging strategy

* Provide two environment files:

  * `environment_cpu.yml` — small fast install for CI and unit testing.
  * `environment_gpu.yml` — includes CUDA-compatible CuPy, cuQuantum, JAX GPU wheels.
* Dockerfile base: `nvidia/cuda:12.0-runtime-ubuntu22.04` and install required wheels.

## Notes about GPU usage

* Abstract backend through `sim.backend` so GPU-specific code is isolated. The backend should automatically detect and select GPU libraries (CuPy / cuStateVec / qsim) if present, else fall back to pure NumPy.
* Provide a deterministic seed mapping for GPU RNGs (CuPy supports seeding).
* Provide fallback tests for CI that run on CPU only; GPU tests run on self-hosted runners.

---

# 4 — Runtime, CI, and deployment

* **CI (`.github/workflows/ci.yaml`)**

  * Run on GitHub-hosted runners for linting and unit tests (CPU only).
  * Artifacts: coverage reports and example small run outputs.
* **GPU tests**

  * Provide a separate workflow template for self-hosted GPU runners (`ci-gpu.yaml`) with steps for CUDA and cuQuantum verification; document how to register a self-hosted runner.
* **Docker**

  * `Dockerfile` builds both CPU and GPU variants; GPU variant extends nvidia/cuda image and installs CuPy/cuQuantum wheels.
  * Include `make docker-build` and `make docker-run`.

---

# 5 — Mathematical foundations (explicit formulas & algorithms to implement)

This is the precise mathematical specification to implement in `docs/math_foundations.md` and in the `noise/` modules. I state formulas cleanly so they can be translated directly to code.

## A. Average gate fidelity → depolarizing parameter

For a quantum channel ( \mathcal{E} ) on a ( d )-dimensional Hilbert space, the average gate fidelity relative to unitary ( U ) is:

[
F_{\text{avg}}(\mathcal{E},U) = \int d\psi \langle\psi| U^\dagger \mathcal{E}(|\psi\rangle\langle\psi|) U |\psi\rangle.
]

For a depolarizing channel on qubits with probability ( p ) (acting as ( \rho \mapsto (1-p)\rho + p \frac{I}{d} )), the relation between ( F_{\text{avg}} ) and ( p ) is:

[
F_{\text{avg}} = 1 - p\frac{d-1}{d}
\quad\Longrightarrow\quad
p = \frac{d}{d-1}\left(1 - F_{\text{avg}}\right).
]

* For single qubit ( d=2 ): ( p_1 = 2(1-F_1) ).
* For two qubits ( d=4 ): ( p_2 = \frac{4}{3}(1-F_2) ).

*(Implement these conversions in `noise/builder.py`.)*

## B. Amplitude damping & pure dephasing per gate

For a gate of duration ( \tau ), a T1 relaxation yields amplitude damping probability:

[
p_{\text{amp}} = 1 - e^{-\tau/T_1}.
]

For dephasing, if the pure dephasing time is ( T_\phi ) (where ( 1/T_2 = 1/(2T_1) + 1/T_\phi )), then

[
p_{\phi} = 1 - e^{-\tau/T_\phi}.
]

You may compute ( T_\phi ) from ( T_2 ) and ( T_1 ):

[
\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2T_1}.
]

Apply amplitude-damping and phase-damping Kraus operators during gate durations and idles.

## C. Kraus operators

* **Amplitude damping (single qubit):**

Kraus operators for parameter ( \gamma = p_{\text{amp}} ):

[
K_0 = \begin{pmatrix}1 & 0\ 0 & \sqrt{1-\gamma} \end{pmatrix},
\qquad
K_1 = \begin{pmatrix}0 & \sqrt{\gamma} \ 0 & 0\end{pmatrix}.
]

* **Phase damping (pure dephasing):** use Kraus with parameter ( \lambda = p_\phi ):

[
K_0 = \sqrt{1-\lambda}, I,\quad
K_1 = \sqrt{\lambda}, Z.
]

(Or implement the standard phase-damping Kraus set.)

* **Depolarizing channel** (single qubit) with probability ( p ):

[
\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z).
]

Construct Kraus or superoperator accordingly.

## D. Composition and time-slicing

For non-commuting channels and time-dependent noise, use **Trotter-like slicing** if needed:

* For a gate with duration ( \tau ) where you have a time-dependent Hamiltonian ( H(t) ) and Lindblad operators ( L_j ), integrate Lindblad master equation:

[
\frac{d\rho}{dt} = -\frac{i}{\hbar}[H(t),\rho] + \sum_j \left(L_j \rho L_j^\dagger - \tfrac{1}{2}{L_j^\dagger L_j, \rho}\right)
]

Implement numerical integration via:

* Exact superoperator exponentiation (if small),
* or piecewise-constant Trotter steps using small Δt slices and applying Kraus channels per slice,
* or Monte Carlo wavefunction trajectories.

## E. Coherent systematic error model

A coherent single-qubit misrotation by small angle ( \varepsilon ) about axis ( \hat{n} ) corresponds to unitary

[
U_{\text{err}} = e^{-i \varepsilon (\hat{n}\cdot\vec{\sigma})/2}.
]

Two-qubit residual ZZ coupling modeled as:

[
U_{\text{zz}} = e^{-i \theta , Z\otimes Z/2}.
]

Combine coherent errors multiplicatively with intended gates: apply (U_{\text{err}}) either before or after native gate depending on calibration model.

## F. Quasi-static noise → relation to (T_2^*)

For quasi-static Gaussian-distributed frequency detuning ( \Delta ) with zero mean and variance ( \sigma^2 ), the ensemble-averaged dephasing envelope for Ramsey experiments is:

[
\langle e^{-i \Delta t} \rangle = e^{-\frac{1}{2}\sigma^2 t^2}.
]

Compare with Ramsey envelope ( \exp\left(-(t/T_2^*)^2\right) ) (Gaussian decay), so identify:

[
\frac{1}{2}\sigma^2 t^2 = \left(\frac{t}{T_2^*}\right)^2
\quad\Rightarrow\quad
\sigma = \sqrt{2}/T_2^*.
]

Thus to reproduce a measured ( T_2^* ) with quasi-static Gaussian detuning, sample ( \Delta \sim \mathcal{N}(0,\sigma^2) ) with ( \sigma = \sqrt{2}/T_2^* ) (radians/second units; be careful with time units).

**Implementation note:** store detuning in angular frequency units (rad/s). When computing a phase accumulated during gate of duration ( \tau ), add phase ( \phi = \Delta \tau ).

## G. Monte-Carlo wavefunction (MCWF) / Quantum trajectories

MCWF propagates a pure state ( |\psi(t)\rangle ) under an effective non-Hermitian Hamiltonian ( H_{\text{eff}} = H - \tfrac{i}{2}\sum_j L_j^\dagger L_j ) and implements stochastic jumps corresponding to Lindblad operators (L_j).

Algorithm sketch:

1. For each trajectory:

   * propagate ( |\psi\rangle ) under ( H_{\text{eff}} ) for small dt (via expm or Krylov).
   * compute cumulative jump probability ( p_j = \Delta t \langle\psi|L_j^\dagger L_j |\psi\rangle ).
   * sample jumps; if jump occurs apply ( L_j |\psi\rangle ) and renormalize.
2. Repeat until end of circuit; average measurement results across trajectories.

MCWF scales with (2^n) per trajectory but avoids density-matrix (4^n). Choose number of trajectories to match target statistical variance.

## H. Heavy-Output Probability (HOP) and QV decision rule

* For a generated ideal circuit, compute the vector of ideal output probabilities ( {p_x} ) for all (x \in {0,1}^m).
* Define the heavy output set ( H ) as the set of bitstrings whose ideal probability exceed the median ideal probability ( p_{\text{med}} ).
* For noisy run with measured counts ( {c_x} ) and (N) shots, compute empirical heavy-output probability:

[
\text{HOP} = \frac{1}{N}\sum_{x\in H} c_x.
]

* For a set of (N_c) random circuits at width (m), compute mean HOP ( \overline{\text{HOP}} ) and confidence intervals (bootstrap or analytic).
* IBM QV pass criterion: ( \overline{\text{HOP}} > 2/3 ) with appropriate confidence (commonly require lower bound of 95% CI > 2/3 or p-value < 0.05 for null ( \overline{\text{HOP}} \le 2/3)).

## I. Statistics — bootstrap & Clopper-Pearson

* **Bootstrap over circuits**: resample circuits (with replacement) and within each circuit resample shots (binomial) or use recorded counts to account for shot noise. For each bootstrap sample compute mean HOP; collect distribution to compute percentiles for CI.
* **Clopper-Pearson**: for single-circuit binomial counts (k) successes out of (N), the exact alpha-level two-sided CI is given by Beta quantiles:

[
\text{CI}*{\text{lower}} = \mathrm{Beta}\left(\alpha/2; k, N-k+1\right),
\quad
\text{CI}*{\text{upper}} = \mathrm{Beta}\left(1-\alpha/2; k+1, N-k\right).
]

Combine per-circuit CIs conservatively in aggregation or apply bootstrap which implicitly handles per-circuit shot noise.

---

# 6 — Numerical, precision and performance recommendations

* Use `complex128` for statevectors and `float64` for probability/CI math to avoid accumulated errors.
* Batched operations: apply gates in parallel on separate state chunks where possible to reduce Python overhead.
* For GPU: use CuPy arrays as drop-in replacements for NumPy. Avoid Python loops over amplitudes; prefer batched linear algebra (matrix-vector products, sparse matmul).
* Memory management:

  * Density matrix for (n) qubits: matrix size (2^n \times 2^n). Use sparse representations for low-rank channels or convert to Kraus propagation.
  * For MCWF, run trajectories sequentially but reuse buffers.
* Threading: use multiprocessing or joblib to parallelize over independent circuits; keep GPU usage balanced (one GPU per worker or multiplex small batches).
* Reproducibility: pin BLAS libs and set environment variables that impact nondeterminism (e.g., `CUBLAS_WORKSPACE_CONFIG` for deterministic cuBLAS kernels where available).

---

# 7 — Unit / integration tests and benchmark cases

**Unit tests (must pass in CI CPU):**

* `tests/unit/test_config.py` — validation of `defaults.yaml`, boundary values.
* `tests/unit/test_channel_construction.py` — check trace-preserving and positivity for Kraus operators (numerical).
* `tests/unit/test_fidelity_conversion.py` — conversions `F->p` produce expected numbers for sample inputs.
* `tests/unit/test_quasi_static_sampling.py` — statistics: sampled detunings reproduce variance.

**Integration tests (fast small runs):**

* `tests/integration/test_qv_small.py` — run QV for `m=2,3` with 5 circuits and 100 shots, assert HOP in [0.4,1.0] (sanity).
* `tests/integration/test_rb_reproduction.py` — run RB under constructed noise model and confirm recovered F1 within tolerance.

**Performance benchmarks (not in CI but useful):**

* `benchmarks/` scripts that measure time per circuit and memory for density-matrix, statevector, and MCWF for `m={8,10,12}`.

---

# 8 — Outputs, file formats and metadata

* Save runs to HDF5 with structure described earlier; include `metadata/config_hash`, `git_commit`, `python_version`, `package_versions` and `random_seed`.
* CSV/Parquet summary table `aggregate_results.parquet` with columns: `m`, `n_circuits`, `n_shots`, `mean_hop`, `ci_lower`, `ci_upper`, `passed_bool`.
* Plot files saved as SVG and PNG, and a summary `report.md` or `report.pdf` assembled by a small renderer (Matplotlib + reportlab or pandoc).

---

# Final practical notes & suggested next actions

* Implement in the order: config/schema → channel primitives → circuit generator → simple CPU statevector backend → run minimal QV (`m<=4`) → validate RB reproduction → add density-matrix/MCWF → add GPU backend.
* Maintain one canonical `NoiseModel` object that can be serialized (JSON) for reproducibility.
* Keep the GPU-specific code isolated in `sim` backends so the rest of the stack is backend-agnostic.
* Include a "hardware profile" JSON file format for each device variant (SiMOS, Si/SiGe enriched, donor-based) so experiments can be reproduced with different parameter sets.
