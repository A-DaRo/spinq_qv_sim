# Copilot Instructions for spinq_qv_sim

## Project Overview
This is a **Quantum Volume (QV) simulator** for Si/SiGe spin-qubit devices. The goal is to estimate simulated QV by running IBM-style QV benchmarks under physically realistic noise models derived from experimental parameters (T1, T2, gate fidelities, crosstalk, SPAM errors).

**Key concept**: Noise-model-first simulation combining coherent errors, stochastic channels, non-Markovian (quasi-static) noise, and device-specific constraints (limited connectivity, gate times).

## Architecture & Data Flow

### Major Components (see `src/spinq_qv/`)
1. **config/** - YAML-based configuration with Pydantic validation
   - `defaults.yaml`: Device params (F1=0.99926, F2=0.998, T1=1s, T2=99µs, gate times)
   - `schemas.py`: Type-safe config classes with validation

2. **circuits/** - QV circuit generation & transpilation
   - `generator.py`: IBM QV random square circuits (width=depth=m)
   - `transpiler.py`: Logical → physical qubit mapping respecting device connectivity
   - `scheduling.py`: Gate scheduling with parallelism and idle insertion

3. **noise/** - Physical noise model construction
   - `builder.py`: Composite noise model from device params
   - `channels.py`: Kraus operators (depolarizing, amplitude damping, phase damping)
   - `coherent.py`: Systematic errors (ZZ coupling, over/under-rotation)
   - `stochastic.py`: Quasi-static 1/f noise, Ornstein-Uhlenbeck processes

4. **sim/** - Simulation backends (CPU/GPU abstractions)
   - `backend.py`: Abstract interface for all simulators
   - `statevector.py`: Pure state (NumPy/CuPy), supports Monte Carlo wavefunction
   - `density_matrix.py`: Kraus channel propagation (exact non-unitary evolution)
   - `mcwf.py`: Monte Carlo wavefunction trajectories
   - `tensornet.py`: Optional tensor network simulator wrapper

5. **analysis/** - QV metric computation & statistics
   - `hop.py`: Heavy-output probability (HOP) calculation
   - `stats.py`: Bootstrap confidence intervals, hypothesis testing
   - `ablation.py`: Sensitivity analysis and error-source ablation

6. **io/** - Data persistence (HDF5/Parquet)
   - `storage.py`: Structured HDF5 writer/reader with circuit metadata
   - `formats.py`: Canonical data schemas

7. **experiments/** - Top-level runners
   - `run_qv.py`: CLI for full QV experiment pipeline
   - `validate.py`: RB-style validation that noise model reproduces target fidelities
   - `sensitivity.py`: Parameter sweeps

### Critical Data Flow
```
Config YAML → NoiseModelBuilder → Simulator Backend
                     ↓                      ↓
          QV Circuit Generator → Transpiler → Scheduler → Noisy Simulation
                                                                 ↓
                                                    Sampled Bitstrings → HOP → Statistics
```

## Project-Specific Conventions

### Noise Model Conversions (Mathematical)
**CRITICAL FORMULAS** (implement exactly as shown):

- **Average fidelity → depolarizing probability:**
  ```
  p_1qubit = 2 * (1 - F1)
  p_2qubit = (4/3) * (1 - F2)
  ```
  Example: F1=0.99926 → p1≈0.00148

- **Decoherence during gates:**
  ```
  p_amp = 1 - exp(-τ / T1)
  p_phi = 1 - exp(-τ / T_phi)
  where 1/T_phi = 1/T2 - 1/(2*T1)
  ```

- **Quasi-static noise vs T2*:**
  ```
  σ = sqrt(2) / T2*
  ```
  Sample detuning Δ ~ N(0, σ²) once per circuit to model run-to-run coherent phase errors

### Coding Patterns

**1. Always use complex128/float64 for numerical precision**
```python
state = np.zeros(2**n, dtype=np.complex128)
probs = np.abs(amplitudes)**2  # float64
```

**2. Backend abstraction for CPU/GPU portability**
```python
# In backend.py
class SimulatorBackend(ABC):
    @abstractmethod
    def apply_kraus(self, kraus_ops, targets): ...
    
# In statevector.py - auto-detect GPU
import numpy as np
try:
    import cupy as cp
    xp = cp  # use CuPy if available
except ImportError:
    xp = np  # fallback to NumPy
```

**3. Configuration-driven experiments (no hardcoded params)**
- All device parameters live in YAML configs
- Use Pydantic for validation and type safety
- Example: `Config.from_yaml("baseline.yaml")`

**4. Deterministic RNG seeding for reproducibility**
- Store and log all random seeds
- Use `utils/rng.py` for centralized seeded RNG management

**5. Structured HDF5 output (see `io/storage.py`)**
```
/config              # JSON/YAML config
/circuits/{m}/{circuit_id}/
    spec             # circuit description
    ideal_probs      # float array
    measured_counts  # compressed dict
/results/aggregates  # pandas DataFrame
```

## Critical Developer Workflows

### Running QV Experiments
```bash
# CPU-only (development)
python -m spinq_qv.experiments.run_qv --config examples/configs/baseline.yaml --widths 2,3,4,5 --output results/

# GPU-accelerated (production)
python -m spinq_qv.experiments.run_qv --config examples/configs/baseline.yaml --widths 2-12 --backend gpu --output results/
```

### Validation Before Production Runs
**Always validate noise model reproduces target fidelities:**
```bash
python -m spinq_qv.experiments.validate --config baseline.yaml
# Should output: Simulated F1 = 0.99926 ± 0.0001 (target: 0.99926)
```

### Testing Strategy
- **Unit tests**: Channel math, conversion formulas, Kraus trace-preservation
- **Integration tests**: Small QV runs (m=2,3 with 5 circuits, 100 shots)
- **Validation tests**: RB sequences must reproduce F1/F2 within tolerance
- Run `pytest tests/unit/` for fast checks (no GPU required)

### Dependency Management
- **CPU path**: `pip install -r requirements_cpu.txt`
- **GPU path**: Match CUDA version! Example for CUDA 12.1:
  ```bash
  pip install cupy-cuda121>=13.0.0
  pip install cuquantum>=23.08.0
  # JAX: check https://github.com/google/jax for exact wheel
  ```
- Check `installation.md` for detailed GPU setup

## Integration Points & External Dependencies

### Optional Qiskit Integration
- Use `qiskit-terra>=0.46.0` for QV circuit helpers/transpiler (optional)
- Keep core simulator independent of Qiskit (backend-agnostic design)

### GPU Libraries (Optional Acceleration)
- **CuPy**: NumPy API on GPU - drop-in replacement
- **cuQuantum**: NVIDIA's statevector/tensornet simulators
- **JAX**: Auto-differentiation + GPU acceleration (alternative to CuPy)
- Abstracted via `sim/backend.py` - code should work on CPU or GPU without changes

### Data Formats
- **Input**: YAML configs (Pydantic-validated)
- **Output**: HDF5 for raw data, Parquet for aggregates, PNG/SVG for plots
- **Logs**: JSON-structured logs with metadata (git hash, package versions, seeds)

## Common Gotchas

1. **Coherent vs stochastic errors**: Coherent systematic errors (ZZ coupling) are often MORE damaging to QV than equivalent depolarizing noise. Don't replace all errors with depolarizing approximations.

2. **Connectivity matters**: Linear qubit chains (limited neighbor connectivity) drastically reduce parallel gates. Always transpile with actual device connectivity graph.

3. **Quasi-static noise is non-Markovian**: Sample detuning once per circuit, not per gate. This models slow drift (1/f noise).

4. **Heavy-output definition**: Heavy outputs are those with ideal probability > median(ideal_probs). Compute median from ideal (noiseless) simulation first.

5. **Statistical requirements**: IBM QV criterion requires mean HOP > 2/3 AND lower 95% CI > 2/3. Use bootstrap with ≥50 circuits per width.

6. **GPU memory**: Density matrix for n qubits needs 2^(2n) × 16 bytes. Example: n=12 → 256 MB; n=14 → 4 GB. Use statevector or MCWF for larger widths.

## Development Priorities

When implementing new features:
1. **Implement exact physics first** - use formulas from `Technicalities.md`
2. **Validate against known results** - reproduce fidelities via RB before trusting QV
3. **Document noise assumptions** - every channel must cite physical origin
4. **Ensure reproducibility** - log seeds, versions, config hashes
5. **Keep backends swappable** - never hardcode `numpy` vs `cupy`

## Reference Files
- **Mathematical foundations**: `Technicalities.md` (conversion formulas, Kraus operators)
- **Architecture blueprint**: `README.md` (high-level plan)
- **Device parameters**: `examples/configs/baseline.yaml` (experimental values)
- **Dependencies**: `requirements_cpu.txt` / `requirements_gpu.txt` (version constraints)

---

**Quick Start for New Contributors:**
1. Read `README.md` for project goals
2. Check `Technicalities.md` for physics/math formulas
3. Install CPU dependencies: `pip install -r requirements_cpu.txt`
4. Run validation: `python -m spinq_qv.experiments.validate`
5. Explore notebooks: `notebooks/tutorial_qv.ipynb`
