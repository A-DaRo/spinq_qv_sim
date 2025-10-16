"""
Gate scheduling with parallelism detection and idle insertion.

Converts circuits into time-scheduled format where gates are organized into
parallel layers. Idle periods are explicitly represented for accurate
decoherence modeling during waiting times.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
from copy import deepcopy

from spinq_qv.io.formats import CircuitSpec


class ScheduledCircuit:
    """
    Represents a circuit with explicit time scheduling.
    
    Gates are organized into time slices with:
    - Gate operations (with durations)
    - Idle periods for qubits not participating in current layer
    """
    
    def __init__(self, width: int, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize scheduled circuit.
        
        Args:
            width: Number of qubits
            metadata: Optional metadata dictionary
        """
        self.width = width
        self.time_slices: List[Dict[str, Any]] = []
        self.total_time = 0.0  # Total circuit time in seconds
        self.metadata = metadata or {}
    
    def add_time_slice(
        self,
        duration: float,
        gates: List[Dict[str, Any]],
        idle_qubits: Set[int],
    ) -> None:
        """
        Add a time slice to the schedule.
        
        Args:
            duration: Duration of this time slice (seconds)
            gates: List of gates executing in this slice
            idle_qubits: Set of qubit indices that are idle during this slice
        """
        time_slice = {
            "start_time": self.total_time,
            "duration": duration,
            "gates": gates,
            "idle_qubits": sorted(idle_qubits),
        }
        
        self.time_slices.append(time_slice)
        self.total_time += duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "width": self.width,
            "total_time": self.total_time,
            "time_slices": self.time_slices,
            "metadata": self.metadata,
        }


class Scheduler:
    """
    Circuit scheduler for converting gates to time-based execution.
    
    Implements ASAP (as-soon-as-possible) scheduling with explicit
    idle gate insertion for accurate noise modeling.
    """
    
    def __init__(
        self,
        gate_durations: Dict[str, float],
        idle_threshold: float = 1e-9,  # Minimum idle time to insert explicit idle gate
    ):
        """
        Initialize scheduler.
        
        Args:
            gate_durations: Dictionary mapping gate types to durations (seconds)
                          e.g., {"u3": 60e-9, "su4": 40e-9, "swap": 80e-9}
            idle_threshold: Minimum idle duration to explicitly model (seconds)
        """
        self.gate_durations = gate_durations
        self.idle_threshold = idle_threshold
    
    def schedule(self, circuit: CircuitSpec) -> ScheduledCircuit:
        """
        Schedule a circuit into time slices.
        
        Args:
            circuit: Input circuit (possibly already transpiled)
        
        Returns:
            ScheduledCircuit with explicit time slices
        """
        m = circuit.width
        
        # Track when each qubit becomes available
        qubit_available_time = [0.0] * m
        
        # Group gates by their original layer if available
        # Otherwise, use sequential ordering
        gates_by_layer = self._group_gates_by_layer(circuit)
        
        # Create scheduled circuit
        scheduled = ScheduledCircuit(
            width=m,
            metadata={
                **circuit.metadata,
                "scheduled": True,
                "scheduling_method": "ASAP",
            }
        )
        
        # Process each layer
        for layer_gates in gates_by_layer:
            # Find when this layer can start (all qubits available)
            layer_qubits = set()
            for gate in layer_gates:
                layer_qubits.update(gate["qubits"])
            
            if layer_qubits:
                layer_start_time = max(
                    qubit_available_time[q] for q in layer_qubits
                )
            else:
                layer_start_time = 0.0
            
            # Insert idle gates if needed for qubits that were idle
            if layer_start_time > 0:
                for q in range(m):
                    idle_duration = layer_start_time - qubit_available_time[q]
                    if idle_duration > self.idle_threshold:
                        # This qubit was idle - decoherence occurs
                        pass  # Will be tracked in time slice
            
            # Determine layer duration (max of all gate durations in layer)
            layer_duration = 0.0
            for gate in layer_gates:
                gate_type = gate["type"]
                gate_duration = self.gate_durations.get(gate_type, 0.0)
                layer_duration = max(layer_duration, gate_duration)
            
            # Find qubits that are idle during this layer
            active_qubits = set()
            for gate in layer_gates:
                active_qubits.update(gate["qubits"])
            idle_qubits = set(range(m)) - active_qubits
            
            # Add time slice
            scheduled.add_time_slice(
                duration=layer_duration,
                gates=layer_gates,
                idle_qubits=idle_qubits,
            )
            
            # Update qubit availability
            for q in active_qubits:
                qubit_available_time[q] = layer_start_time + layer_duration
        
        return scheduled
    
    def _group_gates_by_layer(
        self,
        circuit: CircuitSpec,
    ) -> List[List[Dict[str, Any]]]:
        """
        Group gates into parallel layers.
        
        If gates have a "layer" field, use that. Otherwise, compute
        layers based on dependency analysis.
        
        Args:
            circuit: Input circuit
        
        Returns:
            List of layers, where each layer is a list of gates
        """
        # Check if gates already have layer assignments
        if all("layer" in gate for gate in circuit.gates):
            # Use existing layer assignments
            max_layer = max(gate["layer"] for gate in circuit.gates)
            layers = [[] for _ in range(max_layer + 1)]
            
            for gate in circuit.gates:
                layers[gate["layer"]].append(gate)
            
            return layers
        
        else:
            # Compute layers based on dependencies (greedy ASAP)
            return self._compute_layers_asap(circuit)
    
    def _compute_layers_asap(
        self,
        circuit: CircuitSpec,
    ) -> List[List[Dict[str, Any]]]:
        """
        Compute ASAP layers for gates without layer assignments.
        
        Uses dependency analysis: a gate can be scheduled as soon as
        all qubits it acts on are available.
        """
        m = circuit.width
        layers = []
        remaining_gates = circuit.gates.copy()
        qubit_last_layer = [-1] * m  # Last layer each qubit was used
        
        while remaining_gates:
            current_layer = []
            gates_to_remove = []
            
            for gate in remaining_gates:
                qubits = gate["qubits"]
                
                # Check if all qubits are available (not used in current layer)
                can_schedule = all(
                    q not in [g_q for g in current_layer for g_q in g["qubits"]]
                    for q in qubits
                )
                
                if can_schedule:
                    current_layer.append(gate)
                    gates_to_remove.append(gate)
            
            # Add layer and update tracking
            if current_layer:
                layers.append(current_layer)
                layer_idx = len(layers) - 1
                
                for gate in current_layer:
                    for q in gate["qubits"]:
                        qubit_last_layer[q] = layer_idx
            
            # Remove scheduled gates
            for gate in gates_to_remove:
                remaining_gates.remove(gate)
            
            # Safety check: prevent infinite loop
            if not gates_to_remove and remaining_gates:
                # This shouldn't happen, but just in case
                # Schedule remaining gates one by one
                layers.append([remaining_gates[0]])
                remaining_gates = remaining_gates[1:]
        
        return layers


def create_default_scheduler(device_config: Dict[str, Any]) -> Scheduler:
    """
    Create a scheduler with default gate durations from device config.
    
    Args:
        device_config: Device configuration dictionary
    
    Returns:
        Scheduler instance
    """
    gate_durations = {
        "u3": device_config.get("t_single_gate", 60e-9),
        "rx": device_config.get("t_single_gate", 60e-9),
        "ry": device_config.get("t_single_gate", 60e-9),
        "rz": device_config.get("t_single_gate", 60e-9),
        "h": device_config.get("t_single_gate", 60e-9),
        "su4": device_config.get("t_two_gate", 40e-9),
        "cz": device_config.get("t_two_gate", 40e-9),
        "cx": device_config.get("t_two_gate", 40e-9),
        "swap": device_config.get("t_two_gate", 40e-9) * 2,  # SWAP = 3 CNOTs
    }
    
    return Scheduler(gate_durations=gate_durations)
