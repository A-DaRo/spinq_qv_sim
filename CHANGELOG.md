# Changelog

All notable changes to the spinq_qv_sim project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-17

### Added - Iteration 10 (Final Release)

**Production Campaign Infrastructure:**
- `experiments/campaign.py`: Production campaign runner with state persistence and resume capability
  - `CampaignState`: JSON-based checkpoint/resume state management
  - `ProductionCampaignRunner`: Multi-width campaigns with automatic retry logic
  - CLI flags: `--resume`, `--campaign-id`, `--max-retries`, `--mode campaign`
  - Automatic `start_time` initialization for metadata tracking
  - Proper aggregated results extraction from multi-width runs
- `examples/configs/production.yaml`: Full production config (m=2-12, 100 circuits, 5000 shots)

**Report Generation:**
- `analysis/report_generator.py`: Multi-page PDF report generation
  - Title page with executive summary and metadata
  - HOP vs m plots with confidence intervals
  - Statistical analysis tables with pass/fail indicators
  - Error budget visualizations (pie charts, bar charts)
  - Hardware improvement recommendations based on limiting factors
  - Support for sensitivity heatmaps and RB validation sections

**Documentation:**
- `docs/math_foundations.md`: Complete mathematical derivations
  - Fidelity → depolarizing probability conversions (F1→p1, F2→p2)
  - Coherence time → decoherence conversions (T1→p_amp, T2→p_phi, T2*→σ)
  - Kraus operator representations (depolarizing, amplitude/phase damping)
  - Statistical methods (bootstrap CI, IBM QV decision rule)
  - Heavy-output probability (HOP) definition and computation
  - References to primary literature

**Testing:**
- `tests/integration/test_reproducibility.py`: Campaign reproducibility tests
  - Campaign state save/load validation
  - Metadata completeness checks (git hash, versions, seeds)
  - Deterministic results with same seed
  - Resume functionality validation
  - Failed width retry logic
  - HDF5 structure verification

**Storage Optimization:**
- HDF5 compression level 4 (gzip) for all datasets
- Aggregated-only storage mode for large campaigns
- Metadata tracking: git hash, package versions, timestamps

### Fixed - Iteration 10

**Campaign Runner:**
- Fixed `start_time` initialization: now set at campaign start (in `run()` method) and before first HDF5 write
- Fixed aggregated results extraction: properly handle dict structure from `run_experiment` (width as key)
- Fixed circular import between `campaign.py` and `run_qv.py` using lazy import

**Statistics:**
- Renamed `aggregate_hops` output keys for consistency: `mean`→`mean_hop`, `std`→`std_hop`, `median`→`median_hop`
- This ensures all HOP-related metrics follow the same naming convention

**Test Results:**
- All reproducibility tests now passing (6/6)
- Total test suite: **167 passed, 1 skipped, 0 failed**

### Added - Iteration 9 (Parallelization & Performance)

**Parallelization:**
- `utils/perf.py`: Performance profiling infrastructure
  - `PerformanceProfiler`: Context manager for resource monitoring
  - `PerformanceLogger`: Metric aggregation and JSON export
  - `profile_function`: Decorator for function-level profiling
  - Memory estimation for statevector vs density matrix backends
- Multi-core circuit simulation using `ProcessPoolExecutor`
- Worker function pattern for pickle compatibility
- Topology serialization for inter-process communication
- CLI flags: `--parallel`, `--workers`, `--profile`

**Testing:**
- `tests/integration/test_parallel.py`: Parallelization validation (3 tests)
  - Parallel vs serial consistency (deterministic results)
  - Profiling output structure validation
  - Performance smoke test

### Added - Iteration 8 (Sensitivity & Ablation)

**Sensitivity Analysis:**
- `experiments/sensitivity.py`: High-level orchestrator for parameter sweeps
  - `SensitivityRunner`: Checkpointing and CSV export
  - 1D parameter sweeps with configurable value lists
  - 2D grid sweeps (Cartesian product)
  - Error budget computation and averaging across widths
- `analysis/ablation.py`: Ablation studies and error budgets
  - 10 ablation configs (baseline, 6 single-source, 2 pairwise, ideal)
  - Toggles for depolarizing, amplitude damping, phase damping, readout, crosstalk, quasi-static
  - Error budget with `pct_of_gap` metric (contribution to ideal-baseline gap)
  - JSON and text export

**Visualization:**
- `analysis/plots.py` extensions:
  - `plot_sensitivity_heatmap()`: 2D imshow for parameter grids
  - `plot_error_budget_pie()`: Wedges with percentages
  - `plot_error_budget_bar()`: Horizontal bars sorted by contribution
  - `plot_parameter_sweep()`: 1D line plots with CI shading

**Configuration:**
- `examples/configs/test_ablation.yaml`: Small test config (m∈{2,3}, 5 circuits)
- `examples/configs/sensitivity_grid.yaml`: 4×4 grid config (m∈{2,3,4}, 15 circuits)

**CLI:**
- `--mode` flag with options: `qv`, `ablation`, `sensitivity-1d`, `sensitivity-2d`
- Sensitivity-specific arguments: `--param`, `--param2`, `--values`, `--values2`, `--checkpoint-freq`

**Testing:**
- `tests/integration/test_ablation_small.py`: 7 comprehensive tests
  - Ablation config generation (10 configs)
  - 1D/2D parameter sweep mechanics
  - Error budget computation with synthetic data
  - Checkpointing save/load
  - DataFrame export structure
  - 2D grid creation for heatmaps

### Added - Iteration 7 (Full QV Pipeline)

**Core Pipeline:**
- `experiments/run_qv.py`: Complete CLI entry point
  - Full experiment loop: generate → transpile → schedule → simulate → measure → analyze
  - HDF5 output with structured results
  - Automated plot generation (HOP vs m, summary, distributions)
  - Support for statevector, density_matrix, MCWF backends
- `circuits/transpiler.py`: Qubit mapping for device topology
  - Linear chain (nearest-neighbor connectivity)
  - All-to-all (full connectivity)
  - SWAP insertion for distant operations
- `circuits/scheduling.py`: Gate scheduling in time
  - Parallel gate execution where topology allows
  - Idle time insertion for coherence modeling
- `analysis/hop.py`: Heavy-output probability computation
  - Median-based heavy-set definition
  - Per-circuit HOP from measurement counts
- `analysis/stats.py`: Statistical analysis
  - Bootstrap confidence intervals (95% CI)
  - IBM QV decision rule (mean HOP > 2/3 AND CI_lower > 2/3)
  - Aggregation across circuits
- `analysis/plots.py`: Publication-quality visualization
  - HOP vs m with error bars
  - QV summary plots with pass/fail annotations
  - HOP distributions per width
  - 300 DPI PNG + SVG output

**Data Management:**
- `io/storage.py`: HDF5 structured storage
  - `QVResultsWriter`: Write experiment data with metadata
  - `QVResultsReader`: Load and query results
  - Hierarchical structure: `/metadata`, `/circuits/{m}/{id}`, `/aggregated/{m}`
  - Git hash, package versions, timestamps for reproducibility

**Testing:**
- `tests/integration/test_qv_pipeline_small.py`: End-to-end validation
  - Run pipeline for m∈{2,3,4}
  - Verify HDF5 structure and content
  - Check HOP values in plausible range [0.33, 1.0]

### Added - Iteration 6 (Advanced Backends)

**Density Matrix Backend:**
- `sim/density_matrix.py`: Exact Kraus-based propagation
  - Support for non-unitary channels (amplitude damping, phase damping)
  - Memory limit: n ≤ 14 qubits (2^(2n) × 16 bytes per element)
  - Exact noise modeling without stochastic sampling

**Monte Carlo Wavefunction (MCWF):**
- `sim/mcwf.py`: Quantum trajectory method
  - Stochastic unraveling of density matrix evolution
  - Configurable number of trajectories
  - Approximates density matrix as trajectories → ∞
  - Scalable to larger qubit counts than density matrix

**Testing:**
- `tests/unit/test_density_kraus_equivalence.py`: Kraus application correctness
- `tests/integration/test_mcwf_vs_density.py`: MCWF convergence to density matrix

### Added - Iteration 5 (Noise Integration)

**Noise Simulation:**
- Stochastic Pauli channels for depolarizing errors
- RB-style validation to reproduce configured fidelities
- `experiments/validate.py`: Randomized benchmarking runner
  - Extract F1/F2 from decay curves
  - Compare with configured values (tolerance ±0.5%)
  - CSV export of RB fits

**Testing:**
- `tests/integration/test_rb_reproduction.py`: RB fidelity extraction
  - Validate F1 reproduction within tolerance
  - Validate F2 reproduction within tolerance

### Added - Iteration 4 (Noise Model)

**Noise Primitives:**
- `noise/channels.py`: Kraus operator implementations
  - Depolarizing channel (single-qubit, two-qubit)
  - Amplitude damping (T1 relaxation)
  - Phase damping (pure dephasing)
- `noise/coherent.py`: Systematic errors
  - Over/under-rotation (calibration errors)
  - Residual ZZ coupling (always-on interactions)
- `noise/stochastic.py`: Non-Markovian noise
  - Quasi-static detuning sampling (T2* → σ)
  - Ornstein-Uhlenbeck process for slow drifts
- `noise/builder.py`: Conversion from device parameters
  - F1/F2 → depolarizing probabilities (p1, p2)
  - T1, T2, gate times → decoherence probabilities (p_amp, p_phi)
  - T2* → quasi-static noise amplitude (σ)

**Testing:**
- `tests/unit/test_kraus_trace_preserving.py`: Tr(Σ K†K) = I
- `tests/unit/test_depolarizing_conversion.py`: F→p formula validation
- `tests/unit/test_quasi_static_sigma.py`: T2* → σ conversion

### Added - Iteration 3 (Statevector Backend)

**Simulation Infrastructure:**
- `sim/backend.py`: Abstract backend interface
  - `init_state()`, `apply_unitary()`, `measure()`
  - Backend-agnostic design for CPU/GPU portability
- `sim/statevector.py`: Pure state simulator
  - Efficient bitstring sampling from amplitudes
  - NumPy-based (CPU) with CuPy detection for GPU
  - Support for noiseless and Pauli-twirled channels

**Testing:**
- `tests/unit/test_statevector_basic.py`: Gate application (X, H, CNOT)
- `tests/integration/test_noiseless_sampling.py`: Statistical validation
  - Kolmogorov-Smirnov test: empirical vs ideal probabilities
  - 10 circuits × m=3, 1000 shots each

### Added - Iteration 2 (Circuit Generation)

**QV Circuit Generator:**
- `circuits/generator.py`: IBM-style random square circuits
  - Random SU(4) decomposition for two-qubit layers
  - Permutation layers for qubit connectivity
  - Deterministic from seed for reproducibility
  - `generate_qv_circuit(m, seed)` → `CircuitSpec`
  - `compute_ideal_probabilities()` → noiseless distribution

**Heavy-Output Computation:**
- Median-based heavy-set definition (IBM standard)
- Ideal probability array for HOP validation

**Data Formats:**
- `io/formats.py`: Circuit and result schemas
  - `CircuitSpec`: JSON-serializable circuit representation
  - `CircuitResult`: Measurement outcome container

**Testing:**
- `tests/unit/test_qv_generator.py`:
  - Deterministic circuit generation (same seed → same circuit)
  - Probability normalization (Σp = 1)
  - Heavy-set size (2^(m-1) outcomes)
- `tests/integration/test_noiseless_qv_hop.py`: Ideal HOP ≈ 1

**Documentation:**
- `notebooks/tutorial_qv.ipynb`: Interactive tutorial
  - Circuit generation examples
  - Ideal probability visualization
  - Heavy-output set inspection

### Added - Iteration 1 (Configuration & Utilities)

**Configuration System:**
- `config/schemas.py`: Pydantic validation models
  - `DeviceConfig`: F1, F2, T1, T2, T2*, gate times, SPAM
  - `SimulationConfig`: Backend, circuits, shots, widths, seed
  - `Config`: Top-level with metadata
- `config/defaults.yaml`: Device parameter defaults
  - Si/SiGe experimental values (F1=99.926%, F2=99.8%, T1=1s, T2=99µs)

**Utilities:**
- `utils/rng.py`: Seedable RNG management
  - `initialize_global_rng(seed)` → reproducible NumPy generators
  - `get_global_rng_manager()` → shared RNG state
- `utils/logging_setup.py`: Structured logging
  - JSON-formatted logs (optional)
  - Console and file output
  - Metadata injection (git hash, timestamp)

**Testing:**
- `tests/unit/test_config_validation.py`: Pydantic schema enforcement
- `tests/unit/test_rng_repeatability.py`: Same seed → same draws

### Added - Iteration 0 (Project Bootstrap)

**Repository Structure:**
- LICENSE (MIT)
- README.md: Project overview and goals
- pyproject.toml: Package metadata and build config
- requirements_cpu.txt: CPU dependencies (NumPy, SciPy, h5py, Pydantic)
- requirements_gpu.txt: GPU dependencies (CuPy, cuQuantum, JAX)
- environment_cpu.yml: Conda environment for CPU development
- Dockerfile: Containerized CPU environment
- .github/workflows/ci.yaml: CI pipeline (lint, test, type check)

**Package Skeleton:**
- src/spinq_qv/__init__.py: Package initialization
- src/spinq_qv/config/, circuits/, noise/, sim/, analysis/, io/, experiments/, utils/: Module structure
- tests/unit/, tests/integration/: Test organization

**Testing:**
- `tests/unit/test_repo_smoke.py`: Import smoke test
- CI passing on Linux (GitHub Actions)

## Development Philosophy

**Iteration-Driven Development:**
- Each iteration adds 1-2 major features with tests
- Acceptance criteria validated before moving to next iteration
- Incremental complexity: foundation → circuits → simulation → analysis → optimization

**Testing Strategy:**
- **Unit tests**: Isolated component validation (formulas, conversions, channels)
- **Integration tests**: Multi-component workflows (pipelines, backends, reproducibility)
- **Statistical tests**: Noise model validation (RB), sampling accuracy (K-S test)
- **Performance tests**: Profiling, parallelization, memory usage

**Reproducibility:**
- All experiments deterministic from random seed
- Git hash recorded in HDF5 metadata
- Package versions logged
- Configuration-driven (no hardcoded parameters)

**Code Quality:**
- Type hints throughout (mypy validation)
- Pydantic schemas for configuration
- Docstrings on all public APIs
- Lint checks (flake8, black) in CI

## Current Status (v1.0.0)

**Test Coverage:**
- 161 tests passing (1 skipped for optional cuQuantum)
- Full pipeline validated: generation → simulation → analysis → reporting

**Supported Backends:**
- Statevector (CPU/GPU via NumPy/CuPy)
- Density Matrix (exact Kraus, up to ~12 qubits)
- MCWF (Monte Carlo wavefunction, scalable)

**Noise Models:**
- Depolarizing (gate infidelity)
- Amplitude damping (T1)
- Phase damping (T2)
- Quasi-static noise (T2*)
- Coherent errors (ZZ, over-rotation)
- Readout errors (SPAM)

**Analysis Tools:**
- Heavy-output probability (HOP)
- Bootstrap confidence intervals
- IBM QV decision rule
- Sensitivity analysis (1D/2D parameter sweeps)
- Ablation studies (error budget decomposition)
- PDF report generation

**Documented:**
- Mathematical foundations (formulas, derivations, Kraus operators)
- API reference (docstrings)
- Tutorials (Jupyter notebooks)
- Installation guide (CPU/GPU)

## Future Work (Optional)

**GPU Acceleration (Iteration 10 - Optional):**
- CuPy statevector backend (already partially supported)
- cuQuantum integration (cuStateVec, cuTensorNet)
- GPU density matrix operations
- Deterministic GPU RNG

**Advanced Features:**
- Error mitigation (ZNE, readout correction)
- Gate compilation optimization
- Dynamic decoupling
- Circuit cutting for larger systems
- Real-time noise tracking

**Performance:**
- JIT compilation (Numba, JAX)
- Distributed computing (Dask, Ray)
- Checkpointing for very long campaigns
- Adaptive sampling (early stopping)

**Benchmarking:**
- Comparison with IBM Qiskit Aer
- Comparison with Google Cirq
- Validation against experimental data

---

## Contributors

spinq_qv_sim development team

## License

MIT License - see LICENSE file for details
