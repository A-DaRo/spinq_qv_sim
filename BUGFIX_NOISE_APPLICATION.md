# Critical Bug Fix: Noise Model Not Applied in Simulation

## Date
2024-10-17

## Severity
**CRITICAL** - Results were completely invalid

## Symptom
Campaign with 25 different parameter configurations (varying F1, F2, T1, T2, gate times) ALL produced identical Quantum Volume results (QV=4096, m=12), with nearly identical mean HOP values across all configurations.

**Physically impossible**: F1=0.99426 (poor fidelity) should give MUCH worse results than F1=0.99999 (nearly perfect), but they were identical.

## Root Cause
The `simulate_circuit()` function in `src/spinq_qv/experiments/run_qv.py` was implemented as a placeholder stub:

```python
def simulate_circuit(
    backend,
    circuit,
    n_shots: int,
    noise_model: Dict[str, Any],  # Parameter existed but was IGNORED
    m: int,
):
    # ...
    # Apply each gate (simplified - no noise for now in this iteration)
    # Full noise application will be added in next iteration
    for gate in circuit.gates:
        # Only applies ideal unitaries, no Kraus channels!
        if gate_type == "u3":
            backend.apply_unitary(u3_matrix, [qubits[0]])
        elif gate_type == "su4":
            backend.apply_unitary(unitary, qubits)
    # ...
    counts = backend.measure(shots=n_shots, readout_noise=None)  # No readout noise!
```

**The noise_model parameter was received but completely ignored!**

## Impact
All simulations from project inception until this fix were **completely noiseless**:
- All gates were perfect unitaries
- No depolarizing, amplitude damping, or phase damping channels applied
- No readout noise
- Parameter variations (F1, F2, T1, T2) had ZERO effect on results

This means:
- ❌ All campaign results were invalid
- ❌ No validation that noise model reproduces target fidelities
- ❌ Quantum Volume estimates were for an ideal (noise-free) device, not the realistic device

## Fixes Applied

### 1. Implemented Noise Application in `simulate_circuit()` (run_qv.py, lines 665-755)

**Single-qubit gates (U3):**
```python
# Apply ideal gate
backend.apply_unitary(u3_matrix, [qubits[0]])

# Apply noise channels: amplitude damping → phase damping → depolarizing
if 'amp_kraus' in single_qubit_noise:
    backend.apply_kraus(single_qubit_noise['amp_kraus'], [qubits[0]])

if 'phase_kraus' in single_qubit_noise:
    backend.apply_kraus(single_qubit_noise['phase_kraus'], [qubits[0]])

if 'dep_kraus' in single_qubit_noise:
    backend.apply_kraus(single_qubit_noise['dep_kraus'], [qubits[0]])
```

**Two-qubit gates (SU4, SWAP):**
```python
# Apply ideal gate
backend.apply_unitary(unitary, qubits)

# Apply per-qubit decoherence (amp + phase damping)
if 'amp_kraus_per_qubit' in two_qubit_noise:
    for q in qubits:
        backend.apply_kraus(two_qubit_noise['amp_kraus_per_qubit'], [q])

if 'phase_kraus_per_qubit' in two_qubit_noise:
    for q in qubits:
        backend.apply_kraus(two_qubit_noise['phase_kraus_per_qubit'], [q])

# Apply two-qubit depolarizing
if 'dep_kraus' in two_qubit_noise:
    backend.apply_kraus(two_qubit_noise['dep_kraus'], qubits)
```

### 2. Added `apply_kraus()` Method to `StatevectorBackend` (statevector.py, lines 340-351)

The statevector backend only had `apply_kraus_stochastic()`. Added an alias method for consistency with `DensityMatrixSimulator`:

```python
def apply_kraus(self, kraus_ops: list[np.ndarray], targets: list[int]) -> None:
    """
    Apply Kraus channel (alias for apply_kraus_stochastic).
    
    For statevector simulation, Kraus channels are applied via Monte Carlo
    wavefunction method (stochastic unraveling).
    """
    self.apply_kraus_stochastic(kraus_ops, targets)
```

### 3. Fixed Two-Qubit Depolarizing Channel (builder.py, line 130)

The noise model builder was incorrectly using single-qubit `depolarizing_kraus()` (returns 2×2 Kraus ops) for two-qubit gates. Fixed to use `depolarizing_kraus_2q()` (returns 4×4 Kraus ops):

```python
# BEFORE (WRONG):
dep_kraus = channels.depolarizing_kraus(p_dep)  # Returns 2×2 Kraus ops

# AFTER (CORRECT):
dep_kraus_2q = channels.depolarizing_kraus_2q(p_dep)  # Returns 4×4 Kraus ops
```

## Verification

Created test script `test_noise_fix.py` comparing high fidelity (F1=0.99926, F2=0.998) vs low fidelity (F1=0.95, F2=0.90):

**Before fix:**
- HIGH fidelity: HOP = 0.8040
- LOW fidelity: HOP = 0.8040 (IDENTICAL - bug confirmed)

**After fix:**
- HIGH fidelity: HOP = 0.8040
- LOW fidelity: HOP = 0.7234 (DIFFERENT - bug fixed!)

**Difference: 0.0806** ✅ Physically reasonable!

## Files Modified

1. `src/spinq_qv/experiments/run_qv.py` (lines 665-755)
   - Implemented full noise channel application in `simulate_circuit()`

2. `src/spinq_qv/sim/statevector.py` (lines 340-351)
   - Added `apply_kraus()` alias method

3. `src/spinq_qv/noise/builder.py` (line 130)
   - Fixed two-qubit depolarizing channel to use `depolarizing_kraus_2q()`

## Required Actions

⚠️ **ALL PREVIOUS CAMPAIGN RESULTS ARE INVALID** ⚠️

1. **Delete invalid campaign results:**
   ```bash
   rm -rf campaigns/param_sweep_*
   ```

2. **Re-run all campaigns with fixed noise model:**
   ```bash
   python examples/campaign_executor.py --config examples/campaign_config_generator.py
   ```

3. **Validate noise model reproduces target fidelities:**
   ```bash
   python src/spinq_qv/experiments/validate.py --config examples/configs/baseline.yaml
   ```
   Expected: Simulated F1 ≈ 0.99926 ± 0.0001, Simulated F2 ≈ 0.998 ± 0.0002

4. **Update documentation** to note that results before 2024-10-17 are invalid

## Lessons Learned

1. **Never trust placeholder code in production**: "Full implementation in next iteration" comments are DANGEROUS
2. **Always validate physics early**: The identical results across parameter variations should have been a red flag
3. **Automated testing should include physics sanity checks**: Test that noise actually reduces fidelity
4. **Code review should catch unused parameters**: The `noise_model` parameter was passed but never used

## Test Coverage Added

- `test_noise_fix.py`: Verifies different fidelities produce different HOP values
- Existing integration tests still pass (verified: `test_qv_pipeline_small.py`)

## Status
✅ **FIXED** - Noise model is now correctly applied during simulation.

All new results will be physically meaningful and parameter-dependent.
