import numpy as np

from mbqc import QubitState, graph_state, random_state, stabilizer
from mbqc.noise import (dephased_cluster, fuse, gkp_error_monte_carlo, gkp_error_probability,
                        grow_chain, negativity, noisy_wire_fidelity, squeezing_to_sigma)


def test_noisy_wire():
    psi = random_state(1, np.random.default_rng(0))
    angles = [0.1, 0.2, 0.3]
    assert np.isclose(noisy_wire_fidelity(psi, angles, 0.0), 1)
    f = [noisy_wire_fidelity(psi, angles, p) for p in (0.01, 0.05, 0.1)]
    assert f[0] > f[1] > f[2] > 0.5


def test_negativity_sudden_death():
    pstar = 1 - 1 / np.sqrt(2)
    for n in (4, 6):
        assert np.isclose(negativity(dephased_cluster(n, 0.0), n // 2), 0.5)
        assert np.isclose(negativity(dephased_cluster(n, 0.1), n // 2), 0.9**2 - 0.5)
        assert negativity(dephased_cluster(n, pstar + 0.01), n // 2) < 1e-10


def test_fusion_merges_vertices():
    edges = [("a0", "a1"), ("a1", "a2"), ("b0", "b1"), ("b1", "b2")]
    st = graph_state(["a0", "a1", "a2", "b0", "b1", "b2"], edges)
    assert np.isclose(fuse(st, "a2", "b0", "m"), 0.5)
    chain = [("a0", "a1"), ("a1", "m"), ("m", "b1"), ("b1", "b2")]
    for q in ["a0", "a1", "m", "b1", "b2"]:
        assert np.isclose(st.expval(stabilizer(chain, q)), 1)


def test_chain_growth_rate():
    rng = np.random.default_rng(1)
    final = np.mean([grow_chain(400, 0.5, rng=rng)[-1] for _ in range(200)])
    assert abs(final - (3 + 200)) < 10


def test_gkp_formula_matches_monte_carlo():
    rng = np.random.default_rng(2)
    for db in (5, 8):
        s = squeezing_to_sigma(db)
        exact, mc = gkp_error_probability(s), gkp_error_monte_carlo(s, 400_000, rng)
        assert abs(exact - mc) < 5 * np.sqrt(exact / 400_000) + 1e-4
    assert gkp_error_probability(squeezing_to_sigma(15)) < 1e-10
