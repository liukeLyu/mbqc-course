"""PennyLane uses the same measurement-angle convention as the NumPy code."""

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
from pennylane.ftqc import diagonalize_mcms, measure_arbitrary_basis  # noqa: E402

from mbqc import J, X, Y, Z, random_state  # noqa: E402


def test_teleportation_convention():
    psi, theta = random_state(1, np.random.default_rng(0)), 0.9
    dev = qml.device("default.qubit")

    @diagonalize_mcms
    @qml.qnode(dev, mcm_method="tree-traversal")
    def circuit(post):
        qml.StatePrep(psi, wires=0)
        qml.H(1)
        qml.CZ([0, 1])
        measure_arbitrary_basis(0, angle=theta, plane="XY", postselect=post)
        return [qml.expval(op(1)) for op in (qml.X, qml.Y, qml.Z)]

    for s in (0, 1):
        out = np.linalg.matrix_power(X, s) @ J(theta) @ psi
        expected = np.real([np.vdot(out, P @ out) for P in (X, Y, Z)])
        assert np.allclose(circuit(s), expected)
