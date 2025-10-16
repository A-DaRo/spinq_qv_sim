"""
Test fidelity->depolarizing probability conversion.
"""

from spinq_qv.noise.builder import NoiseModelBuilder


def test_fidelity_to_depolarizing():
    # Single-qubit: p1 = 2*(1 - F1)
    F1 = 0.99926
    p1 = NoiseModelBuilder.fidelity_to_depolarizing_p(F1, two_qubit=False)
    assert abs(p1 - (2 * (1 - F1))) < 1e-12

    # Two-qubit: p2 = (4/3)*(1 - F2)
    F2 = 0.998
    p2 = NoiseModelBuilder.fidelity_to_depolarizing_p(F2, two_qubit=True)
    assert abs(p2 - ((4.0/3.0) * (1 - F2))) < 1e-12
