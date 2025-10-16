"""
NoiseModelBuilder: convert device parameters into per-gate channels.

Implements conversions:
  - fidelity -> depolarizing probability
  - gate time, T1/T2 -> amplitude/phase damping probabilities
  - composition of channels: coherent -> amplitude -> phase -> depolarizing

Returns serializable summary with per-gate numeric params and Kraus ops
(when applicable).
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from spinq_qv.noise import channels, stochastic, coherent


class NoiseModelBuilder:
    """
    Build noise models from configuration parameters.
    """
    def __init__(self, device_params: Dict[str, Any]):
        self.params = device_params

    @staticmethod
    def fidelity_to_depolarizing_p(F: float, two_qubit: bool = False) -> float:
        """
        Convert average gate fidelity to depolarizing probability.

        Single-qubit: p1 = 2 * (1 - F1)
        Two-qubit: p2 = (4/3) * (1 - F2)
        """
        if two_qubit:
            return (4.0 / 3.0) * (1.0 - F)
        else:
            return 2.0 * (1.0 - F)

    @staticmethod
    def decoherence_probs(tau: float, T1: float, T2: float) -> Tuple[float, float]:
        """
        Compute amplitude and phase damping probabilities during gate time tau.

        p_amp = 1 - exp(-tau / T1)
        1/T_phi = 1/T2 - 1/(2*T1)
        p_phi = 1 - exp(-tau / T_phi)
        """
        if T1 <= 0 or T2 <= 0:
            raise ValueError("T1 and T2 must be positive")

        p_amp = 1.0 - np.exp(-tau / T1)
        T_phi_inv = 1.0 / T2 - 1.0 / (2.0 * T1)
        if T_phi_inv <= 0:
            # If calculated T_phi_inv <= 0 due to rounding or physical limits,
            # set phase damping to 0.
            p_phi = 0.0
        else:
            T_phi = 1.0 / T_phi_inv
            p_phi = 1.0 - np.exp(-tau / T_phi)

        return float(p_amp), float(p_phi)

    def build_single_qubit_channel(self, gate_time: float) -> Dict[str, Any]:
        """
        Build composite channel for single-qubit gates from device params.

        Composition order (applied left-to-right on state):
          - Coherent (small systematic rotations)
          - Amplitude damping (energy relaxation)
          - Phase damping (dephasing)
          - Depolarizing (random errors)

        Returns:
            Dictionary containing p_amp, p_phi, p_dep and Kraus operators where appropriate.
        """
        F1 = self.params.get('F1', 0.999)
        T1 = self.params.get('T1', 1.0)
        T2 = self.params.get('T2', 0.000099)

        # Convert fidelity to depolarizing probability
        p_dep = self.fidelity_to_depolarizing_p(F1, two_qubit=False)

        # Decoherence during gate
        p_amp, p_phi = self.decoherence_probs(gate_time, T1, T2)

        # Coherent rotation params (small systematic overrotation)
        coherent_angle = self.params.get('single_qubit_overrotation_rad', 0.0)
        coherent_axis = self.params.get('coherent_axis', 'z')

        # Build Kraus operators
        amp_kraus = channels.amplitude_damping_kraus(p_amp)
        phase_kraus = channels.phase_damping_kraus(p_phi)
        dep_kraus = channels.depolarizing_kraus(p_dep)

        return {
            'p_amp': p_amp,
            'p_phi': p_phi,
            'p_dep': p_dep,
            'amp_kraus': amp_kraus,
            'phase_kraus': phase_kraus,
            'dep_kraus': dep_kraus,
            'coherent': {
                'axis': coherent_axis,
                'angle': coherent_angle,
            }
        }

    def build_two_qubit_channel(self, gate_time: float) -> Dict[str, Any]:
        """
        Build composite channel for two-qubit gates using two-qubit fidelity F2.

        For depolarizing conversion use p2 = (4/3) * (1 - F2)
        Decoherence probabilities are computed per-qubit during gate time.
        """
        F2 = self.params.get('F2', 0.99)
        T1 = self.params.get('T1', 1.0)
        T2 = self.params.get('T2', 0.000099)

        p_dep = self.fidelity_to_depolarizing_p(F2, two_qubit=True)

        # Per-qubit amplitude/phase damping during two-qubit gate
        p_amp_q, p_phi_q = self.decoherence_probs(gate_time, T1, T2)

        # For two qubits we can combine by assuming independent per-qubit channels
        amp_kraus_q = channels.amplitude_damping_kraus(p_amp_q)
        phase_kraus_q = channels.phase_damping_kraus(p_phi_q)
        dep_kraus = channels.depolarizing_kraus(p_dep)

        # residual ZZ coherent coupling
        zz_phi = self.params.get('residual_zz_phase', 0.0)

        return {
            'p_amp_per_qubit': p_amp_q,
            'p_phi_per_qubit': p_phi_q,
            'p_dep': p_dep,
            'amp_kraus_per_qubit': amp_kraus_q,
            'phase_kraus_per_qubit': phase_kraus_q,
            'dep_kraus': dep_kraus,
            'coherent': {
                'zz_phi': zz_phi,
            }
        }

    def build(self, gate_durations: Dict[str, float]) -> Dict[str, Any]:
        """
        Build full noise model summary given gate durations.

        gate_durations: dict e.g., {'single': 60e-9, 'two_qubit': 200e-9}

        Returns dictionary summarizing per-gate channels and numeric params.
        """
        single_time = gate_durations.get('single', 1e-7)
        two_time = gate_durations.get('two_qubit', 2e-7)

        single_model = self.build_single_qubit_channel(single_time)
        two_model = self.build_two_qubit_channel(two_time)

        # Quasi-static detuning sigma from T2*
        t2_star = self.params.get('T2_star', None)
        if t2_star is not None:
            sigma = stochastic.t2_star_to_sigma(t2_star)
        else:
            sigma = None

        summary = {
            'single_qubit': single_model,
            'two_qubit': two_model,
            'quasi_static_sigma': sigma,
            'raw_params': self.params,
        }

        return summary


# Small helper: serialize kraus ops to lists for JSON if needed
def serialize_kraus_list(kraus_list: List[np.ndarray]) -> List[List[List[complex]]]:
    return [[[complex(x) for x in row] for row in K.tolist()] for K in kraus_list]
