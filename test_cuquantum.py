import cuquantum

# Check the version of cuQuantum
print("cuQuantum version:", cuquantum.__version__)

# Example of initializing a state vector
from cuquantum import custatevec

# Create a simple state vector
num_qubits = 2
state_vector = custatevec.initialize_state_vector(num_qubits)

print("Initialized state vector:", state_vector)
