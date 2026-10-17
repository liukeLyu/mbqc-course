"""A tiny state-vector simulator with named qubits.

An n-qubit pure state is stored as a NumPy array of shape (2, 2, ..., 2), one axis per
qubit, plus a list of labels saying which qubit each axis is.  Labels can be anything
hashable: 0, "c", (row, col).  Measuring a qubit removes its axis, which is exactly what
happens in MBQC: measured qubits are gone, the computation lives on the rest.
"""

import numpy as np

# Single-qubit gates and states.
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)


def P(phi):
    """Phase gate diag(1, e^{i phi}), equal to Rz(phi) up to a global phase."""
    return np.diag([1, np.exp(1j * phi)])


def Rx(phi):
    return np.cos(phi / 2) * I2 - 1j * np.sin(phi / 2) * X


def Rz(phi):
    return np.cos(phi / 2) * I2 - 1j * np.sin(phi / 2) * Z


def J(theta):
    """The gate one MBQC step applies: J(theta) = H P(-theta)."""
    return H @ P(-theta)


def xy_basis(theta):
    """Measurement basis in the XY plane of the Bloch sphere at angle theta.

    Outcome s in {0, 1} is the state (|0> + (-1)^s e^{i theta} |1>) / sqrt(2).
    theta = 0 is the X basis, theta = pi/2 the Y basis.
    """
    return [np.array([1, (-1) ** s * np.exp(1j * theta)]) / np.sqrt(2) for s in (0, 1)]


Z_BASIS = [ZERO, ONE]


class QubitState:
    """Pure state of a few labelled qubits."""

    def __init__(self, amps, labels):
        self.amps = np.array(amps, dtype=complex).reshape((2,) * len(labels))  # a copy
        self.labels = list(labels)

    @classmethod
    def product(cls, vectors):
        """Product state from a dict {label: single-qubit vector}."""
        amps = np.ones((), dtype=complex)
        for vec in vectors.values():
            amps = np.multiply.outer(amps, vec)
        return cls(amps, list(vectors))

    def copy(self):
        return QubitState(self.amps.copy(), self.labels)

    def axis(self, q):
        return self.labels.index(q)

    def add(self, q, vec=PLUS):
        """Append a fresh qubit q in state vec (|+> by default)."""
        self.amps = np.multiply.outer(self.amps, vec)
        self.labels.append(q)

    def apply(self, U, q):
        """Apply a single-qubit gate U to qubit q."""
        k = self.axis(q)
        self.amps = np.moveaxis(np.tensordot(U, self.amps, axes=([1], [k])), 0, k)

    def apply2(self, U, a, b):
        """Apply a two-qubit gate U (4x4, basis |ab>) to qubits a, b."""
        ka, kb = self.axis(a), self.axis(b)
        U = U.reshape(2, 2, 2, 2)
        out = np.tensordot(U, self.amps, axes=([2, 3], [ka, kb]))
        self.amps = np.moveaxis(out, [0, 1], [ka, kb])

    def cz(self, a, b):
        """Controlled-Z: flip the sign of every amplitude where a = b = 1."""
        index = [slice(None)] * len(self.labels)
        index[self.axis(a)] = 1
        index[self.axis(b)] = 1
        self.amps[tuple(index)] *= -1

    def measure(self, q, basis, outcome=None, rng=None):
        """Measure qubit q in an orthonormal basis [v0, v1] and remove it.

        Returns the outcome s.  Pass `outcome` to force a branch (useful for checking all
        branches), otherwise it is sampled with the Born rule using `rng`.
        """
        k = self.axis(q)
        branches = [np.tensordot(v.conj(), self.amps, axes=([0], [k])) for v in basis]
        probs = [np.vdot(b, b).real for b in branches]
        if outcome is None:
            rng = rng or np.random.default_rng()
            outcome = int(rng.random() < probs[1] / (probs[0] + probs[1]))
        if probs[outcome] < 1e-12:
            raise ValueError(f"outcome {outcome} on qubit {q!r} has zero probability")
        self.amps = branches[outcome] / np.sqrt(probs[outcome])
        self.labels.remove(q)
        return outcome

    def measure_xy(self, q, theta, outcome=None, rng=None):
        return self.measure(q, xy_basis(theta), outcome, rng)

    def measure_z(self, q, outcome=None, rng=None):
        return self.measure(q, Z_BASIS, outcome, rng)

    def vector(self, order=None):
        """State as a flat vector, qubits in `order` (default: current label order)."""
        order = self.labels if order is None else list(order)
        return np.transpose(self.amps, [self.axis(q) for q in order]).reshape(-1)

    def expval(self, paulis):
        """<psi| prod_q paulis[q] |psi> for a dict {label: 2x2 matrix}."""
        other = self.copy()
        for q, U in paulis.items():
            other.apply(U, q)
        return np.vdot(self.amps, other.amps)

    def entropy(self, part):
        """Von Neumann entropy (in bits) of the qubits in `part`."""
        rest = [q for q in self.labels if q not in part]
        m = self.vector(list(part) + rest).reshape(2 ** len(part), -1)
        p = np.linalg.svd(m, compute_uv=False) ** 2
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log2(p)))


def graph_state(nodes, edges):
    """|G> = prod_{(a,b) in edges} CZ_ab |+>^nodes."""
    state = QubitState.product({q: PLUS for q in nodes})
    for a, b in edges:
        state.cz(a, b)
    return state


def stabilizer(edges, a):
    """Stabilizer generator K_a = X_a prod_{b neighbour of a} Z_b, as a dict."""
    K = {a: X}
    for u, v in edges:
        if a in (u, v):
            K[v if u == a else u] = Z
    return K


def line_edges(n):
    return [(k, k + 1) for k in range(n - 1)]


def grid_edges(rows, cols):
    """Edges of a rows x cols square lattice with nodes labelled (r, c)."""
    edges = [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    edges += [((r, c), (r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    return edges


def random_state(n_qubits, rng):
    v = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return v / np.linalg.norm(v)


def fidelity(a, b):
    """|<a|b>|^2 for normalised vectors: 1 means equal up to a global phase."""
    return float(abs(np.vdot(a, b)) ** 2)
