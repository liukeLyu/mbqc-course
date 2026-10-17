import numpy as np

from mbqc import (PLUS, QubitState, X, Z, fidelity, graph_state, grid_edges, line_edges,
                  random_state, stabilizer, xy_basis, J)


def test_graph_state_stabilizers():
    for nodes, edges in [(range(6), line_edges(6)),
                         ([(r, c) for r in range(3) for c in range(3)], grid_edges(3, 3))]:
        g = graph_state(nodes, edges)
        for a in nodes:
            assert np.isclose(g.expval(stabilizer(edges, a)), 1)


def test_cut_entropy_of_line_is_one_bit():
    g = graph_state(range(6), line_edges(6))
    assert all(np.isclose(g.entropy(list(range(k))), 1) for k in range(1, 6))


def test_z_measurement_deletes_vertex():
    g = graph_state(range(5), line_edges(5))
    s = g.measure_z(2, outcome=1)
    assert s == 1 and g.labels == [0, 1, 3, 4]
    for a in [0, 1, 3, 4]:
        assert np.isclose(abs(g.expval(stabilizer([(0, 1), (3, 4)], a))), 1)


def test_one_bit_teleportation_both_outcomes():
    rng = np.random.default_rng(0)
    psi = random_state(1, rng)
    for theta in (0.0, 0.7, np.pi / 2, 2.5):
        for s in (0, 1):
            st = QubitState.product({0: psi, 1: PLUS})
            st.cz(0, 1)
            st.measure_xy(0, theta, outcome=s)
            assert np.isclose(fidelity(st.vector(), np.linalg.matrix_power(X, s) @ J(theta) @ psi), 1)


def test_xy_basis_orthonormal_and_z_flips_it():
    b0, b1 = xy_basis(0.37)
    assert np.isclose(np.vdot(b0, b1), 0) and np.isclose(np.vdot(b0, b0), 1)
    assert np.isclose(fidelity(Z @ b0, b1), 1)


def test_apply2_matches_cz():
    rng = np.random.default_rng(1)
    v = random_state(3, rng)
    a, b = QubitState(v, [0, 1, 2]), QubitState(v, [0, 1, 2])
    a.cz(0, 2)
    b.apply2(np.diag([1, 1, 1, -1]), 0, 2)
    assert np.allclose(a.vector(), b.vector())
