import itertools

import numpy as np
import pytest

from mbqc import (Rx, Rz, compile_jcz, euler_xzx, measurement_depth, random_state, rotation_angles,
                  run_circuit, run_cnot_patch, run_pattern, run_wire, to_jcz, fidelity)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def random_unitary(rng):
    q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return q * (np.diag(r) / abs(np.diag(r)))


@pytest.mark.parametrize("seed", range(5))
def test_euler_xzx(seed):
    U = random_unitary(np.random.default_rng(seed))
    a, b, c = euler_xzx(U)
    assert np.isclose(abs(np.trace(U.conj().T @ Rx(c) @ Rz(b) @ Rx(a))), 2)


def test_euler_edge_cases():
    for U in (Rx(np.pi), Rz(0.3), np.eye(2), Rx(0.2) @ Rz(np.pi)):
        a, b, c = euler_xzx(U)
        assert np.isclose(abs(np.trace(U.conj().T @ Rx(c) @ Rz(b) @ Rx(a))), 2)


@pytest.mark.parametrize("seed", range(3))
def test_wire_rotation_all_branches(seed):
    rng = np.random.default_rng(seed)
    U, psi = random_unitary(rng), random_state(1, rng)
    angles = rotation_angles(U)
    for o in itertools.product((0, 1), repeat=4):
        out, _ = run_wire(psi, angles, outcomes=o)
        assert np.isclose(fidelity(out, U @ psi), 1)


def test_non_adaptive_wire_fails_somewhere():
    rng = np.random.default_rng(4)
    U, psi = random_unitary(rng), random_state(1, rng)
    fids = [fidelity(run_wire(psi, rotation_angles(U), outcomes=o, adaptive=False)[0], U @ psi)
            for o in itertools.product((0, 1), repeat=4)]
    assert min(fids) < 0.99


def test_cnot_patch_all_branches():
    psi = random_state(2, np.random.default_rng(5))
    measured = [(0, 0), (0, 2), (1, 0), (1, 1)]
    for o in itertools.product((0, 1), repeat=4):
        out, _ = run_cnot_patch(psi, outcomes=dict(zip(measured, o)))
        assert np.isclose(fidelity(out, CNOT @ psi), 1)


def test_compiler_matches_circuit():
    rng = np.random.default_rng(6)
    circuit = [("Rx", 0, 0.3), ("Rz", 1, 1.1), ("CNOT", 0, 1), ("Rx", 1, -0.4), ("H", 0),
               ("CNOT", 1, 0), ("Rz", 0, 2.0), ("Rx", 2, 0.8), ("CNOT", 1, 2), ("CZ", 0, 2)]
    pattern = compile_jcz(3, to_jcz(circuit))
    psi = random_state(3, rng)
    for _ in range(10):
        out, _ = run_pattern(pattern, psi, rng=rng)
        assert np.isclose(fidelity(out, run_circuit(3, circuit, psi)), 1)


def test_two_cz_cancel():
    pattern = compile_jcz(2, [("CZ", 0, 1), ("CZ", 0, 1)])
    assert pattern.edges == []


def test_clifford_depth_is_one():
    gates = [("H", 0), ("CNOT", 0, 1), ("Rz", 1, np.pi / 2), ("H", 1), ("CNOT", 1, 0), ("Rx", 0, np.pi)] * 5
    assert measurement_depth(compile_jcz(2, to_jcz(gates))) == 1
    gates = [("Rz", 0, 0.3), ("Rx", 0, 0.4), ("Rz", 0, 0.5)]
    assert measurement_depth(compile_jcz(1, to_jcz(gates))) > 1
