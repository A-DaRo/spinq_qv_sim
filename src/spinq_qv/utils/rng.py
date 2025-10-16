"""
Seedable random number generator management for reproducibility.

Provides centralized RNG management to ensure deterministic behavior across
all stochastic components (circuit generation, noise sampling, etc.).
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RNGState:
    """Container for RNG state and metadata."""
    
    seed: int
    generator: np.random.Generator
    call_count: int = 0
    
    def increment(self) -> None:
        """Increment call counter for tracking."""
        self.call_count += 1


class RNGManager:
    """
    Centralized random number generator manager.
    
    Manages seeded numpy random generators for different components
    (circuits, noise, sampling) to ensure reproducibility while maintaining
    independence between subsystems.
    
    Usage:
        >>> rng_mgr = RNGManager(global_seed=42)
        >>> circuit_rng = rng_mgr.get_rng("circuits")
        >>> noise_rng = rng_mgr.get_rng("noise")
        >>> # Each subsystem has independent but reproducible randomness
    """
    
    def __init__(self, global_seed: Optional[int] = None):
        """
        Initialize RNG manager with global seed.
        
        Args:
            global_seed: Master seed for all RNGs. If None, uses system entropy.
        """
        self.global_seed = global_seed
        self._rngs: Dict[str, RNGState] = {}
        self._seed_generator = np.random.default_rng(global_seed)
        
        logger.info(f"RNGManager initialized with global_seed={global_seed}")
    
    def get_rng(self, name: str) -> np.random.Generator:
        """
        Get or create a seeded RNG for a named subsystem.
        
        Args:
            name: Subsystem identifier (e.g., "circuits", "noise", "sampling")
        
        Returns:
            Numpy random generator with subsystem-specific seed
        """
        if name not in self._rngs:
            # Generate subsystem-specific seed from master seed
            subsystem_seed = self._seed_generator.integers(0, 2**31 - 1)
            generator = np.random.default_rng(subsystem_seed)
            
            self._rngs[name] = RNGState(
                seed=subsystem_seed,
                generator=generator,
                call_count=0
            )
            
            logger.debug(
                f"Created RNG for subsystem '{name}' with seed={subsystem_seed}"
            )
        
        # Increment usage counter
        self._rngs[name].increment()
        
        return self._rngs[name].generator
    
    def reset(self, global_seed: Optional[int] = None) -> None:
        """
        Reset all RNGs with a new global seed.
        
        Args:
            global_seed: New master seed. If None, uses original seed.
        """
        if global_seed is not None:
            self.global_seed = global_seed
        
        self._rngs.clear()
        self._seed_generator = np.random.default_rng(self.global_seed)
        
        logger.info(f"RNGManager reset with global_seed={self.global_seed}")
    
    def get_state_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all RNG states for logging/debugging.
        
        Returns:
            Dictionary mapping subsystem names to their seed and call count
        """
        return {
            name: {
                "seed": state.seed,
                "call_count": state.call_count
            }
            for name, state in self._rngs.items()
        }
    
    def seed_sequence(self, name: str, n: int) -> np.ndarray:
        """
        Generate a sequence of seeds for parallel workers.
        
        Useful for distributing work across processes/threads while
        maintaining reproducibility.
        
        Args:
            name: Subsystem identifier
            n: Number of seeds to generate
        
        Returns:
            Array of n independent seeds
        """
        rng = self.get_rng(name)
        return rng.integers(0, 2**31 - 1, size=n)


# Global RNG manager instance (can be overridden)
_global_rng_manager: Optional[RNGManager] = None


def initialize_global_rng(seed: Optional[int] = None) -> RNGManager:
    """
    Initialize the global RNG manager.
    
    Args:
        seed: Global seed for reproducibility
    
    Returns:
        Global RNG manager instance
    """
    global _global_rng_manager
    _global_rng_manager = RNGManager(global_seed=seed)
    return _global_rng_manager


def get_global_rng_manager() -> RNGManager:
    """
    Get the global RNG manager, creating one if needed.
    
    Returns:
        Global RNG manager instance
    """
    global _global_rng_manager
    if _global_rng_manager is None:
        _global_rng_manager = RNGManager()
    return _global_rng_manager


def get_rng(name: str = "default") -> np.random.Generator:
    """
    Convenience function to get RNG from global manager.
    
    Args:
        name: Subsystem identifier
    
    Returns:
        Numpy random generator
    """
    return get_global_rng_manager().get_rng(name)
