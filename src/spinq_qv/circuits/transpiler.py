"""
Transpiler for mapping logical circuits to physical device topology.

Handles qubit mapping for limited connectivity (linear nearest-neighbor chains)
and inserts SWAP gates when needed to route two-qubit gates.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from copy import deepcopy

from spinq_qv.io.formats import CircuitSpec


class DeviceTopology:
    """
    Represents physical qubit connectivity.
    
    For Si/SiGe linear arrays, qubits are arranged in a 1D chain with
    nearest-neighbor connectivity only.
    """
    
    def __init__(self, n_qubits: int, topology_type: str = "linear"):
        """
        Initialize device topology.
        
        Args:
            n_qubits: Number of physical qubits
            topology_type: Type of connectivity ("linear", "all-to-all")
        """
        self.n_qubits = n_qubits
        self.topology_type = topology_type
        
        # Build connectivity graph
        if topology_type == "linear":
            # Linear chain: qubit i connects to i-1 and i+1
            self.edges = set()
            for i in range(n_qubits - 1):
                self.edges.add((i, i + 1))
                self.edges.add((i + 1, i))  # Bidirectional
        
        elif topology_type == "all-to-all":
            # Fully connected (no routing needed)
            self.edges = set()
            for i in range(n_qubits):
                for j in range(n_qubits):
                    if i != j:
                        self.edges.add((i, j))
        
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
    
    def are_connected(self, q1: int, q2: int) -> bool:
        """Check if two qubits are directly connected."""
        return (q1, q2) in self.edges
    
    def distance(self, q1: int, q2: int) -> int:
        """
        Compute shortest path distance between two qubits.
        
        For linear topology, this is simply |q1 - q2|.
        """
        if self.topology_type == "linear":
            return abs(q1 - q2)
        elif self.topology_type == "all-to-all":
            return 1 if q1 != q2 else 0
        else:
            # General graph - use BFS (not implemented for now)
            raise NotImplementedError("General graph distance not implemented")


class Transpiler:
    """
    Circuit transpiler for mapping logical qubits to physical device.
    
    Handles:
    - Qubit mapping (logical → physical)
    - SWAP insertion for routing on limited connectivity
    - Gate decomposition (if needed)
    """
    
    def __init__(self, topology: DeviceTopology):
        """
        Initialize transpiler.
        
        Args:
            topology: Physical device topology
        """
        self.topology = topology
    
    def transpile(
        self,
        circuit: CircuitSpec,
        initial_layout: Optional[List[int]] = None,
        optimization_level: int = 1,
    ) -> CircuitSpec:
        """
        Transpile a logical circuit to physical device.
        
        Args:
            circuit: Input circuit (logical qubits)
            initial_layout: Optional initial qubit mapping (logical → physical)
                           If None, uses identity mapping [0, 1, 2, ...]
            optimization_level: Optimization level (0 = none, 1 = basic, 2 = aggressive)
        
        Returns:
            Transpiled circuit with SWAPs inserted and physical qubit indices
        """
        m = circuit.width
        
        # Initialize qubit layout (logical → physical mapping)
        if initial_layout is None:
            # Identity layout: logical qubit i → physical qubit i
            layout = list(range(m))
        else:
            if len(initial_layout) != m:
                raise ValueError(f"Layout must have {m} qubits, got {len(initial_layout)}")
            layout = initial_layout.copy()
        
        # Create new circuit for output
        transpiled = CircuitSpec(
            width=m,
            depth=0,  # Will update as we add gates
            seed=circuit.seed,
            metadata={
                **circuit.metadata,
                "transpiled": True,
                "topology": self.topology.topology_type,
                "initial_layout": layout.copy(),
            }
        )
        
        # Process each gate
        for gate in circuit.gates:
            gate_type = gate["type"]
            logical_qubits = gate["qubits"]
            params = gate.get("params", {})
            layer = gate.get("layer")
            
            if len(logical_qubits) == 1:
                # Single-qubit gate: just map to physical qubit
                physical_q = layout[logical_qubits[0]]
                transpiled.add_gate(
                    gate_type=gate_type,
                    qubits=[physical_q],
                    params=params,
                    layer=layer,
                )
            
            elif len(logical_qubits) == 2:
                # Two-qubit gate: may need SWAPs for routing
                logical_q1, logical_q2 = logical_qubits
                physical_q1 = layout[logical_q1]
                physical_q2 = layout[logical_q2]
                
                # Check if qubits are connected
                if self.topology.are_connected(physical_q1, physical_q2):
                    # Direct connection: apply gate
                    transpiled.add_gate(
                        gate_type=gate_type,
                        qubits=[physical_q1, physical_q2],
                        params=params,
                        layer=layer,
                    )
                else:
                    # Need routing: insert SWAPs
                    swap_path = self._route_qubits(layout, logical_q1, logical_q2)
                    
                    # Apply SWAPs to bring qubits together
                    for swap_pair in swap_path:
                        transpiled.add_gate(
                            gate_type="swap",
                            qubits=swap_pair,
                            params={},
                            layer=None,  # SWAPs get their own layer
                        )
                        # Update layout
                        self._apply_swap(layout, swap_pair[0], swap_pair[1])
                    
                    # Now apply the two-qubit gate
                    physical_q1 = layout[logical_q1]
                    physical_q2 = layout[logical_q2]
                    transpiled.add_gate(
                        gate_type=gate_type,
                        qubits=[physical_q1, physical_q2],
                        params=params,
                        layer=layer,
                    )
            
            else:
                raise ValueError(f"Gates on {len(logical_qubits)} qubits not supported")
        
        # Update depth
        transpiled.depth = len(transpiled.gates)
        
        # Store final layout
        transpiled.metadata["final_layout"] = layout.copy()
        transpiled.metadata["num_swaps"] = sum(
            1 for g in transpiled.gates if g["type"] == "swap"
        )
        
        return transpiled
    
    def _route_qubits(
        self,
        layout: List[int],
        logical_q1: int,
        logical_q2: int,
    ) -> List[Tuple[int, int]]:
        """
        Find SWAP sequence to bring two logical qubits together.
        
        Uses greedy nearest-neighbor routing for linear topology.
        
        Args:
            layout: Current logical → physical mapping
            logical_q1: First logical qubit
            logical_q2: Second logical qubit
        
        Returns:
            List of (physical_q1, physical_q2) SWAP pairs to apply
        """
        # For linear topology, use simple greedy routing
        if self.topology.topology_type == "linear":
            return self._linear_routing(layout, logical_q1, logical_q2)
        else:
            raise NotImplementedError("Routing only implemented for linear topology")
    
    def _linear_routing(
        self,
        layout: List[int],
        logical_q1: int,
        logical_q2: int,
    ) -> List[Tuple[int, int]]:
        """
        Greedy routing for linear chain topology.
        
        Strategy: Move the qubit on the left towards the one on the right
        until they are adjacent.
        """
        # Work on a copy of the layout
        current_layout = layout.copy()
        swaps = []
        
        # Get current physical positions
        phys_q1 = current_layout[logical_q1]
        phys_q2 = current_layout[logical_q2]
        
        # Move q1 towards q2 until adjacent
        while abs(phys_q1 - phys_q2) > 1:
            # Determine direction to move
            if phys_q1 < phys_q2:
                # Move q1 right
                swap_pair = (phys_q1, phys_q1 + 1)
            else:
                # Move q1 left
                swap_pair = (phys_q1 - 1, phys_q1)
            
            swaps.append(swap_pair)
            
            # Update layout (simulate the swap)
            self._apply_swap(current_layout, swap_pair[0], swap_pair[1])
            
            # Update physical position of logical_q1
            phys_q1 = current_layout[logical_q1]
        
        return swaps
    
    @staticmethod
    def _apply_swap(layout: List[int], phys_q1: int, phys_q2: int) -> None:
        """
        Apply a SWAP to the qubit layout (in-place).
        
        Swaps the logical qubits at physical positions phys_q1 and phys_q2.
        """
        # Find which logical qubits are at these physical positions
        logical_at_phys_q1 = layout.index(phys_q1)
        logical_at_phys_q2 = layout.index(phys_q2)
        
        # Swap their physical positions
        layout[logical_at_phys_q1], layout[logical_at_phys_q2] = (
            layout[logical_at_phys_q2],
            layout[logical_at_phys_q1],
        )


def create_linear_topology(n_qubits: int) -> DeviceTopology:
    """Convenience function to create a linear chain topology."""
    return DeviceTopology(n_qubits, topology_type="linear")


def create_all_to_all_topology(n_qubits: int) -> DeviceTopology:
    """Convenience function to create fully-connected topology."""
    return DeviceTopology(n_qubits, topology_type="all-to-all")
