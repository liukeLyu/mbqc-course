"""Lesson 6 helpers: noisy resource states, fusion, and GKP shift errors."""

import itertools
from math import erf, sqrt, pi

import numpy as np

from .state import Z, fidelity, graph_state, line_edges
from .patterns import run_wire


def error_patterns(qubits, p):
    """All subsets of `qubits` with their probability under independent errors of rate p."""
    qubits = list(qubits)
    for bits in itertools.product((0, 1), repeat=len(qubits)):
        k = sum(bits)
        yield [q for q, b in zip(qubits, bits) if b], p**k * (1 - p) ** (len(qubits) - k)


def noisy_wire_fidelity(psi_in, angles, p, rng=None):
    """Exact average output fidelity when every resource qubit gets Z with probability p.

    A Z error on a measured qubit flips its outcome, so the frame undoes the wrong Pauli.
    After correction the output does not depend on the outcomes, so one run per error
    pattern is enough.
    """
    rng = rng or np.random.default_rng(0)
    ideal, _ = run_wire(psi_in, angles, rng=rng)
    resource = range(1, len(angles) + 1)
    total = 0.0
    for errs, w in error_patterns(resource, p):
        out, _ = run_wire(psi_in, angles, rng=rng, z_errors=errs)
        total += w * fidelity(out, ideal)
    return total


def dephased_cluster(n, p):
    """Density matrix of a 1D cluster after each qubit dephases (Z with probability p)."""
    rho = np.zeros((2**n, 2**n), dtype=complex)
    for errs, w in error_patterns(range(n), p):
        state = graph_state(range(n), line_edges(n))
        for q in errs:
            state.apply(Z, q)
        v = state.vector()
        rho += w * np.outer(v, v.conj())
    return rho


def negativity(rho, n_left):
    """Negativity across the cut (first n_left qubits | rest): sum of |negative eigenvalues|."""
    dl = 2**n_left
    dr = rho.shape[0] // dl
    pt = rho.reshape(dl, dr, dl, dr).transpose(0, 3, 2, 1).reshape(rho.shape)
    ev = np.linalg.eigvalsh(pt)
    return float(-ev[ev < 0].sum())


def fuse(state, a, b, new):
    """Successful type-I fusion: project qubits a, b onto |00>, |11> and merge them into `new`.

    On graph states this merges two vertices into one whose neighbours are the union.
    Returns the success probability of this projection (1/2 for two graph-state ends).
    """
    ka, kb = state.axis(a), state.axis(b)
    merged = np.diagonal(state.amps, axis1=ka, axis2=kb)  # amplitude where a = b, new last axis
    prob = float(np.vdot(merged, merged).real)
    state.amps = merged / np.sqrt(prob)
    state.labels = [q for q in state.labels if q not in (a, b)] + [new]
    return prob


def grow_chain(n_attempts, p_success, start=3, piece=3, rng=None):
    """Classical bookkeeping of building a long chain from `piece`-qubit chains.

    Success merges the end of the long chain with the end of a new piece
    (length + piece - 1).  Failure measures both end qubits in Z: the piece is wasted and
    the long chain loses its end qubit.  Returns the length after each attempt.
    """
    rng = rng or np.random.default_rng()
    length, history = start, []
    for _ in range(n_attempts):
        if rng.random() < p_success:
            length += piece - 1
        else:
            length = max(length - 1, 0)
        history.append(length)
    return np.array(history)


# --- GKP: a qubit stored in an oscillator, protected against small shifts ------------

SQRT_PI = sqrt(pi)


def squeezing_to_sigma(db):
    """Std of the shift error for `db` of squeezing (vacuum variance 1/2, hbar = 1)."""
    return sqrt(0.5 * 10 ** (-db / 10))


def gkp_error_probability(sigma, terms=20):
    """Probability that a Gaussian shift of std sigma is decoded to the wrong codeword.

    The decoder rounds the shift to the nearest multiple of sqrt(pi); it fails when the
    shift is closer to an odd multiple, i.e. |shift| in (sqrt(pi)/2, 3 sqrt(pi)/2) mod 2 sqrt(pi).
    """

    def cdf(x):
        return 0.5 * (1 + erf(x / (sigma * sqrt(2))))

    total = 0.0
    for k in range(-terms, terms):
        lo, hi = (2 * k + 0.5) * SQRT_PI, (2 * k + 1.5) * SQRT_PI
        total += cdf(hi) - cdf(lo)
    return total


def gkp_error_monte_carlo(sigma, shots, rng):
    shifts = rng.normal(0, sigma, size=shots)
    nearest = np.round(shifts / SQRT_PI).astype(int)
    return float(np.mean(nearest % 2 == 1))
