"""
Performance profiling and resource monitoring utilities.

Provides decorators and context managers for timing, memory tracking,
and performance logging of circuit simulations.
"""

import time
import logging
import psutil
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function execution."""
    
    function_name: str
    wall_time_seconds: float
    cpu_time_seconds: float
    peak_memory_mb: float
    memory_delta_mb: float
    start_resources: ResourceSnapshot
    end_resources: ResourceSnapshot
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'wall_time_seconds': self.wall_time_seconds,
            'cpu_time_seconds': self.cpu_time_seconds,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'start_resources': self.start_resources.to_dict(),
            'end_resources': self.end_resources.to_dict(),
        }
    
    def format_summary(self) -> str:
        """Format human-readable summary."""
        lines = [
            f"Performance: {self.function_name}",
            f"  Wall time: {self.wall_time_seconds:.3f}s",
            f"  CPU time:  {self.cpu_time_seconds:.3f}s",
            f"  Peak memory: {self.peak_memory_mb:.1f} MB",
            f"  Memory delta: {self.memory_delta_mb:+.1f} MB",
        ]
        return "\n".join(lines)


class PerformanceProfiler:
    """
    Context manager for profiling code blocks.
    
    Example:
        with PerformanceProfiler("circuit_simulation") as prof:
            simulate_circuit(...)
        
        print(prof.metrics.format_summary())
    """
    
    def __init__(self, name: str = "block"):
        """
        Initialize profiler.
        
        Args:
            name: Name for this profiling block
        """
        self.name = name
        self.metrics: Optional[PerformanceMetrics] = None
        
        self._start_time: float = 0.0
        self._start_cpu: float = 0.0
        self._start_snapshot: Optional[ResourceSnapshot] = None
        self._peak_memory: float = 0.0
    
    def __enter__(self) -> "PerformanceProfiler":
        """Start profiling."""
        self._start_time = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_snapshot = self._get_resource_snapshot()
        self._peak_memory = self._start_snapshot.memory_mb
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and compute metrics."""
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        end_snapshot = self._get_resource_snapshot()
        
        # Update peak memory
        self._peak_memory = max(self._peak_memory, end_snapshot.memory_mb)
        
        self.metrics = PerformanceMetrics(
            function_name=self.name,
            wall_time_seconds=end_time - self._start_time,
            cpu_time_seconds=end_cpu - self._start_cpu,
            peak_memory_mb=self._peak_memory,
            memory_delta_mb=end_snapshot.memory_mb - self._start_snapshot.memory_mb,
            start_resources=self._start_snapshot,
            end_resources=end_snapshot,
        )
        
        return False  # Don't suppress exceptions
    
    @staticmethod
    def _get_resource_snapshot() -> ResourceSnapshot:
        """Get current resource usage snapshot."""
        process = psutil.Process()
        
        with process.oneshot():
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
        )


def profile_function(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to profile function performance.
    
    Example:
        @profile_function
        def simulate_circuit(...):
            ...
        
        # Or with custom name:
        @profile_function(name="my_simulation")
        def simulate(...):
            ...
    
    Args:
        func: Function to profile (when used without parentheses)
        name: Optional custom name for profiling block
    
    Returns:
        Decorated function that logs performance metrics
    """
    def decorator(f: Callable) -> Callable:
        prof_name = name if name is not None else f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with PerformanceProfiler(prof_name) as prof:
                result = f(*args, **kwargs)
            
            # Log metrics
            logger.debug(prof.metrics.format_summary())
            
            # Store metrics as function attribute for later access
            if not hasattr(wrapper, '_performance_metrics'):
                wrapper._performance_metrics = []
            wrapper._performance_metrics.append(prof.metrics)
            
            return result
        
        return wrapper
    
    # Handle both @profile_function and @profile_function(...) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


class PerformanceLogger:
    """
    Aggregate performance metrics logger.
    
    Collects metrics from multiple profiling runs and saves to JSON.
    """
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize logger.
        
        Args:
            output_path: Optional path to save metrics JSON
        """
        self.output_path = output_path
        self.metrics: list[PerformanceMetrics] = []
    
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add metrics to collection."""
        self.metrics.append(metrics)
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            path: Path to save (overrides constructor path)
        """
        save_path = path if path is not None else self.output_path
        
        if save_path is None:
            raise ValueError("No output path specified")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metrics': [m.to_dict() for m in self.metrics],
            'summary': self.get_summary(),
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance metrics saved to {save_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all metrics.
        
        Returns:
            Dictionary with summary stats
        """
        if not self.metrics:
            return {}
        
        wall_times = [m.wall_time_seconds for m in self.metrics]
        cpu_times = [m.cpu_time_seconds for m in self.metrics]
        peak_mems = [m.peak_memory_mb for m in self.metrics]
        
        import numpy as np
        
        return {
            'count': len(self.metrics),
            'total_wall_time': sum(wall_times),
            'mean_wall_time': np.mean(wall_times),
            'median_wall_time': np.median(wall_times),
            'std_wall_time': np.std(wall_times),
            'total_cpu_time': sum(cpu_times),
            'mean_cpu_time': np.mean(cpu_times),
            'peak_memory_mb': max(peak_mems),
            'mean_memory_mb': np.mean(peak_mems),
        }
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.get_summary()
        
        if not summary:
            print("No performance metrics collected")
            return
        
        print("=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Total runs: {summary['count']}")
        print(f"Total wall time: {summary['total_wall_time']:.2f}s")
        print(f"Mean wall time: {summary['mean_wall_time']:.3f}s ± {summary['std_wall_time']:.3f}s")
        print(f"Median wall time: {summary['median_wall_time']:.3f}s")
        print(f"Total CPU time: {summary['total_cpu_time']:.2f}s")
        print(f"Peak memory: {summary['peak_memory_mb']:.1f} MB")
        print(f"Mean memory: {summary['mean_memory_mb']:.1f} MB")
        print("=" * 70)


def estimate_memory_requirements(
    n_qubits: int,
    backend_type: str,
    overhead_factor: float = 1.5
) -> Dict[str, float]:
    """
    Estimate memory requirements for simulation.
    
    Args:
        n_qubits: Number of qubits
        backend_type: 'statevector' or 'density_matrix'
        overhead_factor: Multiplicative overhead (default 1.5x for temporaries)
    
    Returns:
        Dictionary with memory estimates in MB
    """
    complex128_bytes = 16  # 8 bytes real + 8 bytes imag
    
    if backend_type == "statevector":
        # 2^n complex amplitudes
        state_size = 2**n_qubits * complex128_bytes
    elif backend_type == "density_matrix":
        # 2^n × 2^n complex matrix
        state_size = (2**n_qubits)**2 * complex128_bytes
    else:
        # Default to statevector
        state_size = 2**n_qubits * complex128_bytes
    
    # Apply overhead for temporaries, gates, etc.
    total_bytes = state_size * overhead_factor
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'state_size_mb': state_size / (1024 * 1024),
        'estimated_total_mb': total_mb,
        'overhead_factor': overhead_factor,
    }
