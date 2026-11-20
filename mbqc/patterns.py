"""Measurement patterns: wires, a CNOT on a 2D patch, and a small circuit-to-MBQC compiler.

The bookkeeping idea used everywhere: after each measurement the logical state is the
ideal one times a known Pauli "byproduct" X^x Z^z.  We never undo it on the spot; we just
remember (x, z) for each qubit (the Pauli frame), adapt later measurement angles to it,
and undo it once at the very end.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .state import PLUS, QubitState, X, Z, line_edges, graph_state, Rx, Rz, H


class PauliFrame:
    """Byproduct bookkeeping: actual state = prod_q X_q^x[q] Z_q^z[q] |ideal>.

    The bits can be ints (0/1) or frozensets of measurement labels; XOR (^) works for
    both, and the set version tells you which outcomes each byproduct depends on.
    """

    def __init__(self, zero=0):
        self.x = defaultdict(lambda: zero)
        self.z = defaultdict(lambda: zero)

    def angle(self, q, theta):
        """Angle to measure q at so that, given its X byproduct, it acts as theta."""
        return -theta if self.x[q] else theta

    def teleport(self, src, dst, s):
        """Measuring src (outcome s) moved the logical state to dst, applying J.

        J((-1)^x theta) X^x Z^z = Z^x X^z J(theta) (up to phase), so this holds only when src
        was measured at `angle(src, theta)`; the measurement then adds X^s.  Unflipped, an X
        gives J(theta) X = Z J(-theta): a wrong rotation, not a Pauli.
        """
        x, z = self.x[src], self.z[src]
        del self.x[src], self.z[src]
        self.x[dst] = z ^ s
        self.z[dst] = x ^ self.z[dst]

    def cz(self, a, b):
        """CZ (X_a) = (X_a Z_b) CZ: an X byproduct on one side of a CZ adds Z on the other."""
        self.z[a] ^= self.x[b]
        self.z[b] ^= self.x[a]

    def z_measured(self, s, neighbours):
        """A Z measurement with outcome s leaves Z^s on each of its graph neighbours."""
        for w in neighbours:
            self.z[w] ^= s

    def undo(self, state, q):
        """Remove the byproduct on q: ideal = Z^z X^x actual (up to phase)."""
        if self.x[q]:
            state.apply(X, q)
        if self.z[q]:
            state.apply(Z, q)


def run_wire(psi_in, angles, outcomes=None, rng=None, adaptive=True, z_errors=()):
    """Push a one-qubit state along a 1D cluster of len(angles) + 1 qubits.

    Qubit 0 carries psi_in, qubits 1..n-1 start in |+>, all neighbouring pairs get a CZ,
    then qubits 0, 1, ..., n-2 are measured in the XY plane.  The ideal result on the last
    qubit is J(angles[-1]) ... J(angles[0]) |psi_in>.

    Returns (corrected output vector, list of outcomes).  `z_errors` lists qubits that get
    a Z error after the cluster is made (used in lesson 6).
    """
    n = len(angles) + 1
    state = QubitState.product({0: np.asarray(psi_in, dtype=complex)})
    for k in range(1, n):
        state.add(k)
    for a, b in line_edges(n):
        state.cz(a, b)
    for q in z_errors:
        state.apply(Z, q)
    frame = PauliFrame()
    seen = []
    for k, theta in enumerate(angles):
        angle = frame.angle(k, theta) if adaptive else theta
        forced = None if outcomes is None else outcomes[k]
        s = state.measure_xy(k, angle, outcome=forced, rng=rng)
        frame.teleport(k, k + 1, s)
        seen.append(s)
    frame.undo(state, n - 1)
    return state.vector(), seen


def euler_xzx(U):
    """Angles (a, b, c) with U = Rx(c) Rz(b) Rx(a) up to a global phase."""
    U = np.asarray(U, dtype=complex)
    V = H @ U @ H / np.sqrt(np.linalg.det(U))  # H Rx H = Rz, so V = Rz(c) Rx(b) Rz(a)
    b = 2 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    a_plus_c = 2 * np.angle(V[1, 1])
    a_minus_c = 2 * (np.angle(V[0, 1]) + np.pi / 2)
    return (a_plus_c + a_minus_c) / 2, b, (a_plus_c - a_minus_c) / 2


def rotation_angles(U):
    """Measurement angles for a 5-qubit wire that applies U (lesson 3).

    J(t3) J(t2) J(t1) J(0) = Rx(-t3) Rz(-t2) Rx(-t1) up to phase.
    """
    a, b, c = euler_xzx(U)
    return [0.0, -a, -b, -c]


# --- Lesson 4: a CNOT carved out of a 2 x 3 patch of a 2D cluster -------------------
#
#   (0,0) -- (0,1) -- (0,2)        (0,1) = control (input and output)
#     |        |        |          (1,0) = target input, (1,2) = target output
#   (1,0) -- (1,1) -- (1,2)        (0,0), (0,2) are removed by Z measurements

CNOT_PATCH = {
    "control": (0, 1),
    "target": [(1, 0), (1, 1), (1, 2)],
    "removed": [(0, 0), (0, 2)],
}


def run_cnot_patch(psi_in, outcomes=None, rng=None):
    """CNOT on a 2 x 3 cluster patch.  psi_in is a 4-vector in the order (control, target).

    Returns (corrected output in the order (control, target), outcomes dict).
    """
    from .state import grid_edges

    c = CNOT_PATCH["control"]
    t0, t1, t2 = CNOT_PATCH["target"]
    edges = grid_edges(2, 3)
    state = QubitState(np.asarray(psi_in, dtype=complex), [c, t0])
    for q in [(0, 0), (0, 2), t1, t2]:
        state.add(q, PLUS)
    for a, b in edges:
        state.cz(a, b)

    def neighbours(v, alive):
        return [w for a, b in edges for w in (a, b) if v in (a, b) and w != v and w in alive]

    frame = PauliFrame()
    got = {}
    pick = (lambda q: None) if outcomes is None else (lambda q: outcomes[q])
    # 1. Carve: Z measurements delete (0,0) and (0,2) from the graph.
    for v in CNOT_PATCH["removed"]:
        alive = list(state.labels)
        got[v] = state.measure_z(v, outcome=pick(v), rng=rng)
        frame.z_measured(got[v], neighbours(v, alive))
    # 2. Target: J(0) = H from (1,0) to (1,1), CZ with control, J(0) = H to (1,2).
    got[t0] = state.measure_xy(t0, frame.angle(t0, 0.0), outcome=pick(t0), rng=rng)
    frame.teleport(t0, t1, got[t0])
    frame.cz(c, t1)
    got[t1] = state.measure_xy(t1, frame.angle(t1, 0.0), outcome=pick(t1), rng=rng)
    frame.teleport(t1, t2, got[t1])
    frame.undo(state, c)
    frame.undo(state, t2)
    return state.vector([c, t2]), got


# --- Lesson 5: compile any {J, CZ} circuit into a pattern ---------------------------


@dataclass
class Pattern:
    """A graph, an ordered list of commands, and which nodes hold inputs and outputs.

    commands: ("M", node, next_node, theta)  measure node at theta, state moves on
              ("CZ", node_a, node_b)          frame update for a CZ between two wires
    """

    n_nodes: int
    edges: list
    commands: list
    inputs: list
    outputs: list


def compile_jcz(n, gates):
    """Translate a circuit of ("J", q, theta) and ("CZ", a, b) gates into a Pattern.

    Every J adds one node to wire q; every CZ adds one edge between the current nodes of
    two wires.  That's the whole compiler.
    """
    current = list(range(n))
    n_nodes = n
    edges = set()
    commands = []
    for gate in gates:
        if gate[0] == "J":
            _, q, theta = gate
            new = n_nodes
            n_nodes += 1
            edges ^= {(current[q], new)}
            commands.append(("M", current[q], new, theta))
            current[q] = new
        elif gate[0] == "CZ":
            _, a, b = gate
            edges ^= {tuple(sorted((current[a], current[b])))}  # two CZs cancel
            commands.append(("CZ", current[a], current[b]))
        else:
            raise ValueError(f"unknown gate {gate!r}; use to_jcz first")
    return Pattern(n_nodes, sorted(edges), commands, list(range(n)), current)


def run_pattern(pattern, psi_in, rng=None, outcomes=None):
    """Make the graph state (inputs carry psi_in), measure, return the corrected output."""
    n_in = len(pattern.inputs)
    state = QubitState(np.asarray(psi_in, dtype=complex), pattern.inputs)
    for q in range(n_in, pattern.n_nodes):
        state.add(q)
    for a, b in pattern.edges:
        state.cz(a, b)
    frame = PauliFrame()
    seen = {}
    for cmd in pattern.commands:
        if cmd[0] == "M":
            _, node, nxt, theta = cmd
            forced = None if outcomes is None else outcomes[node]
            seen[node] = state.measure_xy(node, frame.angle(node, theta), forced, rng)
            frame.teleport(node, nxt, seen[node])
        else:
            frame.cz(cmd[1], cmd[2])
    for q in pattern.outputs:
        frame.undo(state, q)
    return state.vector(pattern.outputs), seen


def is_pauli_angle(theta):
    """X or Y measurements: flipping the sign only relabels outcomes, so no waiting."""
    return np.isclose(np.sin(2 * theta), 0)


def measurement_depth(pattern):
    """Number of rounds of measurements needed when each round may use earlier outcomes.

    Runs the frame symbolically (bits are sets of outcome labels).  A non-Pauli
    measurement must wait for every outcome in its X byproduct; Pauli ones never wait.
    """
    frame = PauliFrame(zero=frozenset())
    rounds = {}
    for cmd in pattern.commands:
        if cmd[0] == "M":
            _, node, nxt, theta = cmd
            waits = () if is_pauli_angle(theta) else frame.x[node]
            rounds[node] = 1 + max((rounds[m] for m in waits), default=0)
            frame.teleport(node, nxt, frozenset({node}))
        else:
            frame.cz(cmd[1], cmd[2])
    return max(rounds.values(), default=0)


def to_jcz(gates):
    """Rewrite H, Rz, Rx, CNOT, CZ gates in terms of J(theta) and CZ (time order).

    H = J(0),  Rz(phi) ~ J(0) J(-phi),  Rx(phi) ~ J(-phi) J(0),  CNOT = H_t CZ H_t.
    """
    out = []
    for g in gates:
        name, *args = g
        if name == "H":
            out.append(("J", args[0], 0.0))
        elif name == "Rz":
            q, phi = args
            out += [("J", q, -phi), ("J", q, 0.0)]
        elif name == "Rx":
            q, phi = args
            out += [("J", q, 0.0), ("J", q, -phi)]
        elif name == "CNOT":
            c, t = args
            out += [("J", t, 0.0), ("CZ", c, t), ("J", t, 0.0)]
        elif name in ("CZ", "J"):
            out.append(g)
        else:
            raise ValueError(f"unknown gate {g!r}")
    return out


def run_circuit(n, gates, psi_in):
    """Reference: apply the gates directly to a state vector (qubit 0 first)."""
    from .state import J

    state = QubitState(np.asarray(psi_in, dtype=complex), list(range(n)))
    single = {"H": lambda: H, "Rz": lambda phi: Rz(phi), "Rx": lambda phi: Rx(phi), "J": J}
    for name, *args in gates:
        if name in ("CZ", "CNOT"):
            a, b = args
            if name == "CNOT":
                state.apply(H, b)
            state.cz(a, b)
            if name == "CNOT":
                state.apply(H, b)
        else:
            state.apply(single[name](*args[1:]), args[0])
    return state.vector()


def cluster_wire(n):
    """Convenience: the plain 1D cluster state on qubits 0..n-1."""
    return graph_state(range(n), line_edges(n))
