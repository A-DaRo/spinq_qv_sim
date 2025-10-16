"""
HDF5-based storage for Quantum Volume experiment data.

Provides structured storage for:
- Configuration metadata
- Circuit specifications
- Ideal probabilities
- Measurement results
- Aggregated statistics
- Git commit hash and environment info for reproducibility
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import subprocess
from datetime import datetime
import numpy as np
import h5py

from spinq_qv.io.formats import CircuitSpec, CircuitResult


def _numpy_json_encoder(obj):
    """JSON encoder for numpy types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class QVResultsWriter:
    """
    Writer for Quantum Volume experimental results to HDF5.
    
    Stores data in hierarchical structure:
    /metadata/
        config.json
        git_hash
        timestamp
        package_versions
    /circuits/{width}/{circuit_id}/
        spec
        ideal_probs
        measured_counts
        hop
    /aggregated/{width}/
        mean_hop
        ci_lower
        ci_upper
        ...
    """
    
    def __init__(self, output_path: Path):
        """
        Initialize HDF5 writer.
        
        Args:
            output_path: Path to HDF5 file (will be created/overwritten)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open HDF5 file in write mode
        self.h5file = h5py.File(self.output_path, 'w')
        
        # Create top-level groups
        self.metadata_group = self.h5file.create_group('metadata')
        self.circuits_group = self.h5file.create_group('circuits')
        self.aggregated_group = self.h5file.create_group('aggregated')
    
    def write_metadata(
        self,
        config: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write experiment metadata.
        
        Args:
            config: Configuration dictionary
            additional_metadata: Additional metadata to store
        """
        # Store config as JSON string
        self.metadata_group.attrs['config'] = json.dumps(config, indent=2)
        
        # Git hash for reproducibility
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = 'unknown'
        
        self.metadata_group.attrs['git_hash'] = git_hash
        
        # Timestamp
        self.metadata_group.attrs['timestamp'] = datetime.now().isoformat()
        
        # Python package versions
        try:
            import numpy
            import scipy
            import h5py as h5
            
            versions = {
                'numpy': numpy.__version__,
                'scipy': scipy.__version__,
                'h5py': h5.__version__,
            }
            
            # Try to get cupy version if available
            try:
                import cupy
                versions['cupy'] = cupy.__version__
            except ImportError:
                pass
            
            self.metadata_group.attrs['package_versions'] = json.dumps(versions)
        
        except Exception:
            self.metadata_group.attrs['package_versions'] = '{}'
        
        # Additional metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    self.metadata_group.attrs[key] = value
                else:
                    self.metadata_group.attrs[key] = json.dumps(value)
    
    def write_circuit_result(
        self,
        width: int,
        circuit_id: str,
        circuit_spec: CircuitSpec,
        ideal_probs: np.ndarray,
        measured_counts: Dict[str, int],
        hop: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write results for a single circuit.
        
        Args:
            width: Circuit width (m)
            circuit_id: Unique circuit identifier
            circuit_spec: Circuit specification
            ideal_probs: Ideal probability array
            measured_counts: Measurement count dictionary
            hop: Heavy-output probability
            metadata: Additional circuit metadata
        """
        # Create width group if doesn't exist
        width_key = f"{width}"
        if width_key not in self.circuits_group:
            width_group = self.circuits_group.create_group(width_key)
        else:
            width_group = self.circuits_group[width_key]
        
        # Create circuit group
        circuit_group = width_group.create_group(circuit_id)
        
        # Store circuit spec as JSON
        circuit_group.attrs['spec'] = json.dumps(circuit_spec.to_dict())
        
        # Store ideal probabilities as dataset
        circuit_group.create_dataset(
            'ideal_probs',
            data=ideal_probs,
            compression='gzip',
            compression_opts=9,
        )
        
        # Store measured counts as JSON (sparse representation)
        circuit_group.attrs['measured_counts'] = json.dumps(measured_counts)
        
        # Store HOP
        circuit_group.attrs['hop'] = hop
        
        # Additional metadata
        if metadata:
            circuit_group.attrs['metadata'] = json.dumps(metadata)
    
    def write_aggregated_results(
        self,
        width: int,
        results: Dict[str, Any],
    ) -> None:
        """
        Write aggregated statistics for a width.
        
        Args:
            width: Circuit width
            results: Dictionary of aggregated statistics
        """
        width_key = f"{width}"
        
        if width_key not in self.aggregated_group:
            agg_group = self.aggregated_group.create_group(width_key)
        else:
            agg_group = self.aggregated_group[width_key]
        
        # Store all numeric values as attributes
        for key, value in results.items():
            if isinstance(value, (int, float, bool, str)):
                agg_group.attrs[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                # Convert numpy scalar types to Python types
                agg_group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                # Convert numpy bool to Python bool
                agg_group.attrs[key] = bool(value)
            elif isinstance(value, np.ndarray):
                # Store arrays as datasets
                agg_group.create_dataset(key, data=value, compression='gzip')
            elif isinstance(value, list):
                # Store lists as datasets
                agg_group.create_dataset(key, data=np.array(value), compression='gzip')
            else:
                # JSON encode complex objects (with numpy type handling)
                agg_group.attrs[key] = json.dumps(value, default=_numpy_json_encoder)
    
    def close(self) -> None:
        """Close the HDF5 file."""
        self.h5file.close()
        print(f"[OK] Results saved to {self.output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class QVResultsReader:
    """
    Reader for Quantum Volume results from HDF5.
    """
    
    def __init__(self, input_path: Path):
        """
        Initialize HDF5 reader.
        
        Args:
            input_path: Path to HDF5 file
        """
        self.input_path = Path(input_path)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")
        
        self.h5file = h5py.File(self.input_path, 'r')
    
    def read_metadata(self) -> Dict[str, Any]:
        """Read experiment metadata."""
        metadata = dict(self.h5file['metadata'].attrs)
        
        # Parse JSON fields
        if 'config' in metadata:
            metadata['config'] = json.loads(metadata['config'])
        
        if 'package_versions' in metadata:
            metadata['package_versions'] = json.loads(metadata['package_versions'])
        
        return metadata
    
    def read_circuit_result(
        self,
        width: int,
        circuit_id: str,
    ) -> Dict[str, Any]:
        """
        Read results for a single circuit.
        
        Args:
            width: Circuit width
            circuit_id: Circuit identifier
        
        Returns:
            Dictionary with circuit data
        """
        width_key = f"{width}"
        circuit_path = f"circuits/{width_key}/{circuit_id}"
        
        if circuit_path not in self.h5file:
            raise KeyError(f"Circuit not found: {circuit_path}")
        
        circuit_group = self.h5file[circuit_path]
        
        result = {
            'circuit_id': circuit_id,
            'width': width,
            'spec': json.loads(circuit_group.attrs['spec']),
            'ideal_probs': circuit_group['ideal_probs'][:],
            'measured_counts': json.loads(circuit_group.attrs['measured_counts']),
            'hop': circuit_group.attrs['hop'],
        }
        
        if 'metadata' in circuit_group.attrs:
            result['metadata'] = json.loads(circuit_group.attrs['metadata'])
        
        return result
    
    def read_aggregated_results(self, width: int) -> Dict[str, Any]:
        """
        Read aggregated statistics for a width.
        
        Args:
            width: Circuit width
        
        Returns:
            Dictionary of aggregated statistics
        """
        width_key = f"{width}"
        agg_path = f"aggregated/{width_key}"
        
        if agg_path not in self.h5file:
            raise KeyError(f"Aggregated results not found for width {width}")
        
        agg_group = self.h5file[agg_path]
        
        results = dict(agg_group.attrs)
        
        # Also read datasets
        for key in agg_group.keys():
            results[key] = agg_group[key][:]
        
        # Parse JSON fields if present
        for key, value in results.items():
            if isinstance(value, str) and value.startswith('{'):
                try:
                    results[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        
        return results
    
    def list_widths(self) -> List[int]:
        """List all circuit widths in the results."""
        if 'circuits' not in self.h5file:
            return []
        
        return sorted([int(w) for w in self.h5file['circuits'].keys()])
    
    def list_circuits(self, width: int) -> List[str]:
        """List all circuit IDs for a given width."""
        width_key = f"{width}"
        
        if 'circuits' not in self.h5file or width_key not in self.h5file['circuits']:
            return []
        
        return list(self.h5file['circuits'][width_key].keys())
    
    def close(self) -> None:
        """Close the HDF5 file."""
        self.h5file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_run(input_path: Path) -> Dict[str, Any]:
    """
    Convenience function to load entire QV run.
    
    Args:
        input_path: Path to HDF5 file
    
    Returns:
        Dictionary with all data
    """
    with QVResultsReader(input_path) as reader:
        metadata = reader.read_metadata()
        widths = reader.list_widths()
        
        aggregated = {}
        for width in widths:
            aggregated[width] = reader.read_aggregated_results(width)
        
        return {
            'metadata': metadata,
            'widths': widths,
            'aggregated': aggregated,
        }
