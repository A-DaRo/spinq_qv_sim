"""
Pydantic schemas for configuration validation.

Type-safe configuration classes with validation for device parameters,
simulation settings, and experiment configurations.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import yaml


class DeviceConfig(BaseModel):
    """Device parameters (gate fidelities, coherence times, etc.)."""
    
    # Gate fidelities
    F1: float = Field(
        default=0.99926,
        ge=0.0,
        le=1.0,
        description="Single-qubit average gate fidelity"
    )
    F2: float = Field(
        default=0.998,
        ge=0.0,
        le=1.0,
        description="Two-qubit average gate fidelity"
    )
    
    # Coherence times (in seconds)
    T1: float = Field(
        default=1.0,
        gt=0.0,
        description="Amplitude damping time T1 (seconds)"
    )
    T2: float = Field(
        default=99e-6,
        gt=0.0,
        description="Hahn echo coherence time T2 (seconds)"
    )
    T2_star: float = Field(
        default=20e-6,
        gt=0.0,
        description="Ramsey dephasing time T2* (seconds)"
    )
    
    # Gate times (in seconds)
    t_single_gate: float = Field(
        default=60e-9,
        gt=0.0,
        description="Single-qubit gate duration (seconds)"
    )
    t_two_gate: float = Field(
        default=40e-9,
        gt=0.0,
        description="Two-qubit gate duration (seconds)"
    )
    
    # Coherent error parameters
    single_qubit_overrotation_rad: float = Field(
        default=0.0,
        description="Single-qubit systematic over-rotation angle (radians)"
    )
    coherent_axis: str = Field(
        default='z',
        description="Axis for single-qubit coherent errors (x, y, or z)"
    )
    residual_zz_phase: float = Field(
        default=0.0,
        description="Two-qubit residual ZZ coupling phase (radians)"
    )
    
    # Crosstalk parameters (NEW - Improvement #2)
    zz_crosstalk_strength: float = Field(
        default=0.0,
        description="Always-on ZZ crosstalk between neighbors (rad/s)"
    )
    control_crosstalk_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Control pulse leakage fraction to spectator qubits (0-1)"
    )
    
    # SPAM parameters (NEW - Improvement #3)
    state_prep_error: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of |1⟩ after reset (state prep error)"
    )
    meas_error_1given0: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="P(measure 1 | true state is |0⟩) - false positive rate"
    )
    meas_error_0given1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="P(measure 0 | true state is |1⟩) - false negative rate"
    )
    
    # Readout time
    t_readout: float = Field(
        default=10e-6,
        gt=0.0,
        description="Readout time (seconds)"
    )
    
    # Time-dependent noise parameters (NEW - Improvement #4)
    coherent_drift_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of coherent error drift (radians)"
    )
    sigma_drift_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relative drift in T2* noise magnitude (0-1)"
    )
    
    @field_validator("T2")
    @classmethod
    def validate_t2(cls, v: float, info) -> float:
        """Ensure T2 <= 2*T1 (physical constraint)."""
        if "T1" in info.data and v > 2 * info.data["T1"]:
            raise ValueError(f"T2 ({v}) cannot exceed 2*T1 ({2*info.data['T1']})")
        return v
    
    @field_validator("T2_star")
    @classmethod
    def validate_t2_star(cls, v: float, info) -> float:
        """Ensure T2* <= T2 (physical constraint)."""
        if "T2" in info.data and v > info.data["T2"]:
            raise ValueError(f"T2* ({v}) cannot exceed T2 ({info.data['T2']})")
        return v
    
    @field_validator("coherent_axis")
    @classmethod
    def validate_coherent_axis(cls, v: str) -> str:
        """Ensure coherent axis is valid."""
        if v.lower() not in ['x', 'y', 'z']:
            raise ValueError("coherent_axis must be 'x', 'y', or 'z'")
        return v.lower()


class SimulationConfig(BaseModel):
    """Simulation parameters (backend, shots, circuits, etc.)."""
    
    backend: str = Field(
        default="statevector",
        description="Simulation backend (statevector, density_matrix, mcwf, tensornet)"
    )
    n_circuits: int = Field(
        default=50,
        ge=1,
        description="Number of random circuits per width"
    )
    n_shots: int = Field(
        default=1000,
        ge=1,
        description="Number of measurement shots per circuit"
    )
    widths: List[int] = Field(
        default=[2, 3, 4, 5, 6],
        description="List of QV circuit widths to test"
    )
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility (None = random)"
    )
    
    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Ensure backend is supported."""
        allowed = {"statevector", "density_matrix", "mcwf", "tensornet"}
        if v not in allowed:
            raise ValueError(f"Backend must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator("widths")
    @classmethod
    def validate_widths(cls, v: List[int]) -> List[int]:
        """Ensure all widths are positive."""
        if any(w < 1 for w in v):
            raise ValueError("All widths must be positive integers")
        return sorted(v)


class Config(BaseModel):
    """Top-level configuration combining device and simulation parameters."""
    
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (experiment name, notes, etc.)"
    )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            # Use model_dump for Pydantic v2 compatibility
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
