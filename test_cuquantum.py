try:
	import cuquantum
except Exception:  # pragma: no cover - optional dependency
	import pytest

	pytest.skip("cuquantum not installed; skipping cuquantum smoke test", allow_module_level=True)

# Check the version of cuQuantum
print("cuQuantum version:", cuquantum.__version__)

# Example of initializing a state vector
from cuquantum import custatevec

# Create a simple state vector
num_qubits = 2
state_vector = custatevec.initialize_state_vector(num_qubits)

print("Initialized state vector:", state_vector)
