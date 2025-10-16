"""
Data formats and schemas for circuit serialization.

Defines canonical JSON/HDF5 schemas for circuits, results, and metadata.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import json
import numpy as np


@dataclass
class CircuitSpec:
    """
    Canonical circuit specification for QV circuits.
    
    Stores circuit structure in a backend-agnostic format that can be
    serialized to JSON/HDF5 and used to reconstruct the circuit.
    """
    
    width: int  # Number of qubits (m)
    depth: int  # Circuit depth (= m for QV)
    seed: int   # Random seed used to generate this circuit
    
    # Layers of gates (list of dicts describing each gate)
    gates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitSpec":
        """Load from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "CircuitSpec":
        """Load from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_gate(
        self,
        gate_type: str,
        qubits: List[int],
        params: Optional[Dict[str, float]] = None,
        layer: Optional[int] = None,
    ) -> None:
        """
        Add a gate to the circuit.
        
        Args:
            gate_type: Gate name (e.g., 'rx', 'ry', 'rz', 'cz', 'su4')
            qubits: List of qubit indices this gate acts on
            params: Optional gate parameters (angles, etc.)
            layer: Optional layer number for scheduling
        """
        gate = {
            "type": gate_type,
            "qubits": qubits,
        }
        
        if params is not None:
            gate["params"] = params
        
        if layer is not None:
            gate["layer"] = layer
        
        self.gates.append(gate)
    
    def num_gates(self) -> int:
        """Return total number of gates."""
        return len(self.gates)
    
    def num_single_qubit_gates(self) -> int:
        """Count single-qubit gates."""
        return sum(1 for g in self.gates if len(g["qubits"]) == 1)
    
    def num_two_qubit_gates(self) -> int:
        """Count two-qubit gates."""
        return sum(1 for g in self.gates if len(g["qubits"]) == 2)


@dataclass
class CircuitResult:
    """
    Results from simulating a circuit.
    
    Stores both ideal and noisy simulation results, along with
    computed metrics like HOP.
    """
    
    circuit_id: str
    circuit_spec: CircuitSpec
    
    # Ideal (noiseless) results
    ideal_probabilities: Optional[np.ndarray] = None
    ideal_heavy_outputs: Optional[List[str]] = None
    
    # Noisy simulation results
    measured_counts: Optional[Dict[str, int]] = None
    n_shots: Optional[int] = None
    
    # Computed metrics
    hop: Optional[float] = None  # Heavy-output probability
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_arrays: If True, convert numpy arrays to lists for JSON
        """
        data = {
            "circuit_id": self.circuit_id,
            "circuit_spec": self.circuit_spec.to_dict(),
            "n_shots": self.n_shots,
            "hop": self.hop,
            "metadata": self.metadata,
        }
        
        if include_arrays:
            if self.ideal_probabilities is not None:
                data["ideal_probabilities"] = self.ideal_probabilities.tolist()
            if self.ideal_heavy_outputs is not None:
                data["ideal_heavy_outputs"] = self.ideal_heavy_outputs
        
        if self.measured_counts is not None:
            data["measured_counts"] = self.measured_counts
        
        return data
    
    def to_json(self) -> str:
        """Serialize to JSON (without large arrays)."""
        return json.dumps(self.to_dict(include_arrays=False), indent=2)


def bitstring_to_int(bitstring: str) -> int:
    """Convert binary string to integer."""
    return int(bitstring, 2)


def int_to_bitstring(value: int, width: int) -> str:
    """Convert integer to binary string with fixed width."""
    return format(value, f'0{width}b')


def compute_heavy_outputs(probabilities: np.ndarray) -> List[int]:
    """
    Compute heavy outputs: those with probability > median.
    
    Args:
        probabilities: Array of output probabilities (length 2^m)
    
    Returns:
        List of indices of heavy outputs
    """
    median = np.median(probabilities)
    heavy_indices = np.where(probabilities > median)[0]
    return heavy_indices.tolist()


def compute_hop(
    measured_counts: Dict[str, int],
    heavy_outputs: List[str],
) -> float:
    """
    Compute heavy-output probability from measurement counts.
    
    Args:
        measured_counts: Dictionary mapping bitstrings to counts
        heavy_outputs: List of bitstrings that are heavy outputs
    
    Returns:
        Fraction of shots that produced heavy outputs
    """
    total_shots = sum(measured_counts.values())
    heavy_shots = sum(
        measured_counts.get(bitstring, 0)
        for bitstring in heavy_outputs
    )
    
    return heavy_shots / total_shots if total_shots > 0 else 0.0
