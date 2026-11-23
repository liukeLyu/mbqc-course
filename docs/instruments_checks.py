"""Numerical checks for every formula and number in docs/instruments-pocket-course.html.

Run:  python docs/instruments_checks.py   (NumPy only; prints each quoted number and asserts each identity).

Conventions: |0> = (1, 0); in a tensor product the leftmost factor is the first qubit (or the
classical register); a classical-quantum (cq) state is stored as one full matrix
sum_k |k><k| (x) rho_k with unnormalized blocks rho_k, so p_k = Tr rho_k.
MBQC conventions follow the MBQC pocket course: |s_theta> = (|0> + (-1)^s e^{i theta}|1>)/sqrt2,
J(theta) = H P(-theta), P(phi) = diag(1, e^{i phi}).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.linalg import eigh
from numpy.typing import NDArray

C = NDArray[np.complex128]

I2: C = np.eye(2, dtype=complex)
X: C = np.array([[0, 1], [1, 0]], dtype=complex)
Y: C = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z: C = np.diag([1, -1]).astype(complex)
H: C = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
ket0: C = np.array([1, 0], dtype=complex)
ket1: C = np.array([0, 1], dtype=complex)
plus: C = (ket0 + ket1) / np.sqrt(2)
minus: C = (ket0 - ket1) / np.sqrt(2)
rng = np.random.default_rng(7)


def kron(*ops: C) -> C:
    """Kronecker product of the arguments, leftmost factor first."""
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def proj(v: C) -> C:
    """|v><v| for a (not necessarily normalized) vector v."""
    return np.outer(v, v.conj())


def basis(k: int, d: int) -> C:
    """Computational basis vector |k> of dimension d."""
    e = np.zeros(d, dtype=complex)
    e[k] = 1
    return e


def cq(blocks: list[C]) -> C:
    """The cq state sum_k |k><k| (x) blocks[k] (register first)."""
    n = len(blocks)
    return sum(kron(proj(basis(k, n)), b) for k, b in enumerate(blocks))


def blocks_of(omega: C, n: int) -> list[C]:
    """Diagonal blocks <k|omega|k> of a state whose first factor is an n-level register."""
    d = omega.shape[0] // n
    return [omega[k * d : (k + 1) * d, k * d : (k + 1) * d] for k in range(n)]


def ptrace(rho: C, dims: list[int], keep: list[int]) -> C:
    """Partial trace of rho over all subsystems not in keep (keep order preserved)."""
    n = len(dims)
    t = rho.reshape(dims + dims)
    idx = list(range(n))
    for j in sorted(set(idx) - set(keep), reverse=True):
        t = np.trace(t, axis1=j, axis2=j + t.ndim // 2)
    d = int(np.prod([dims[j] for j in keep]))
    return t.reshape(d, d)


def random_rho(d: int, pure: bool = False) -> C:
    """A random density matrix of dimension d (pure if asked)."""
    if pure:
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        return proj(v / np.linalg.norm(v))
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    r = g @ g.conj().T
    return r / np.trace(r)


def tnorm(a: C) -> float:
    """Trace norm ||a||_1 of a Hermitian matrix."""
    return float(np.abs(np.linalg.eigvalsh(a)).sum())


def vn_entropy(rho: C) -> float:
    """Von Neumann entropy in bits."""
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-(w * np.log2(w)).sum())


def close(a: C | float, b: C | float, tol: float = 1e-10) -> bool:
    """Entrywise closeness."""
    return bool(np.allclose(a, b, atol=tol))


def equal_up_to_phase(a: C, b: C) -> bool:
    """True if a = e^{i phi} b for some phi."""
    k = np.argmax(np.abs(b))
    if abs(b.flat[k]) < 1e-12:
        return close(a, b)
    ph = a.flat[k] / b.flat[k]
    return abs(abs(ph) - 1) < 1e-10 and close(a, ph * b)


def instrument(kraus: list[list[C]]) -> list:
    """Return the maps E_k(rho) = sum_j K_kj rho K_kj^dagger for a list of Kraus lists."""
    return [lambda r, ks=ks: sum(K @ r @ K.conj().T for K in ks) for ks in kraus]


def P(phi: float) -> C:
    return np.diag([1, np.exp(1j * phi)])


def J(theta: float) -> C:
    return H @ P(-theta)


def s_theta(s: int, theta: float) -> C:
    return (ket0 + (-1) ** s * np.exp(1j * theta) * ket1) / np.sqrt(2)


CZ: C = np.diag([1, 1, 1, -1]).astype(complex)
CNOT: C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def section(title: str) -> None:
    print(f"\n== {title}")


# ---------------------------------------------------------------- Lesson 1-2: keep every branch, reading it
section("L1-2  Measuring qubit A of |psi> = sqrt(1/3)|0>|0> + sqrt(2/3)|1>|+>")
psi = np.sqrt(1 / 3) * kron(ket0[:, None], ket0[:, None]).ravel() + np.sqrt(2 / 3) * np.kron(ket1, plus)
rho_AB = proj(psi)
blocks = [(kron(basis(k, 2)[None, :], I2) @ rho_AB @ kron(basis(k, 2)[:, None], I2)) for k in range(2)]
omega = cq(blocks)
p = [np.trace(b).real for b in blocks]
print("p_k =", np.round(p, 4), " blocks:", [np.round(b.real, 4).tolist() for b in blocks])
assert close(blocks[0], proj(ket0) / 3) and close(blocks[1], 2 / 3 * proj(plus))
assert close(np.trace(omega), 1) and np.all(np.linalg.eigvalsh(omega) > -1e-12)
rho_B_forget = ptrace(omega, [2, 2], [1])
print("Tr_X omega =", np.round(rho_B_forget.real, 4).tolist(), " (= Tr_A |psi><psi|)")
assert close(rho_B_forget, ptrace(rho_AB, [2, 2], [1]))
assert close(rho_B_forget, np.array([[2 / 3, 1 / 3], [1 / 3, 1 / 3]]))
print("postselect k=1: success 2/3, state |+><+|:", close(blocks[1] / p[1], proj(plus)))

# ---------------------------------------------------------------- Lesson 3: the label must be written down
section("L3  Two ensembles of I/2, with and without the record")
omega_Z = cq([proj(ket0) / 2, proj(ket1) / 2])
omega_X = cq([proj(plus) / 2, proj(minus) / 2])
assert close(ptrace(omega_Z, [2, 2], [1]), I2 / 2) and close(ptrace(omega_X, [2, 2], [1]), I2 / 2)
D = 0.5 * tnorm(omega_Z - omega_X)
print(f"trace distance of the cq states = {D:.4f} (1/sqrt2 = {1 / np.sqrt(2):.4f}); best guess {0.5 * (1 + D):.4f}")
assert close(D, 1 / np.sqrt(2))
# HJW: |0>,|1> ensemble and |+>,|-> ensemble related by the Hadamard (a unitary mixing the sqrt-weighted vectors)
V1 = np.column_stack([ket0, ket1]) / np.sqrt(2)
V2 = np.column_stack([plus, minus]) / np.sqrt(2)
assert close(V2, V1 @ H)

# ---------------------------------------------------------------- Lesson 4: a register is a diagonal quantum system
section("L4  Reading or copying a register disturbs nothing")
Delta = lambda r: sum(proj(basis(k, 2)) @ r @ proj(basis(k, 2)) for k in range(2))
DeltaX = lambda om: sum(kron(proj(basis(k, 2)), I2) @ om @ kron(proj(basis(k, 2)), I2) for k in range(2))
assert close(DeltaX(omega), omega)
# copy X onto a fresh bit X': CNOT_{X -> X'} on (X, X', B)
omega_XXpB = np.zeros((8, 8), dtype=complex)
for k in range(2):
    omega_XXpB += kron(proj(basis(k, 2)), proj(ket0), blocks[k])
U = kron(CNOT, I2)
copied = U @ omega_XXpB @ U.conj().T
target = sum(kron(proj(basis(k, 2)), proj(basis(k, 2)), blocks[k]) for k in range(2))
assert close(copied, target)
bell_like = CNOT @ proj(np.kron(plus, ket0)) @ CNOT.T
print(
    "CNOT copies a register exactly; on a coherent |+> it makes a Bell pair, purity of first qubit =",
    round(np.trace(np.linalg.matrix_power(ptrace(bell_like, [2, 2], [0]), 2)).real, 4),
)

# ---------------------------------------------------------------- Lesson 5: instruments
section("L5  A weak Z measurement as an instrument")
phi = np.pi / 8
K0 = np.diag([np.cos(phi), np.sin(phi)]).astype(complex)
K1 = np.diag([np.sin(phi), np.cos(phi)]).astype(complex)
assert close(K0.conj().T @ K0 + K1.conj().T @ K1, I2)
E = instrument([[K0], [K1]])
rho_in = proj(plus)
out = [E[k](rho_in) for k in range(2)]
print(
    "on |+>: p =",
    [round(np.trace(o).real, 4) for o in out],
    " post-state 0 =",
    np.round((out[0] / np.trace(out[0])).real, 4).tolist(),
)
assert close([np.trace(o).real for o in out], [0.5, 0.5])
# Phi = sum_k |k><k| (x) E_k is CPTP: Choi matrix PSD, partial trace over output = I
choi = sum(
    np.kron(
        proj(basis(i, 2)) if i == j else np.outer(basis(i, 2), basis(j, 2)),
        cq([E[k](np.outer(basis(i, 2), basis(j, 2))) for k in range(2)]),
    )
    for i in range(2)
    for j in range(2)
)
assert np.linalg.eigvalsh(choi).min() > -1e-12 and close(ptrace(choi, [2, 4], [0]), I2)
print("instrument channel Q -> XQ is CPTP: Choi min eigenvalue", round(np.linalg.eigvalsh(choi).min(), 12))
pw = [np.cos(phi) ** 2 * 0.5 + np.sin(phi) ** 2 * 0.5] * 2
assert close([np.trace(o).real for o in out], pw)
rho_t = random_rho(2)
pk = [np.trace(E[k](rho_t)).real for k in range(2)]
r0, r1 = rho_t[0, 0].real, rho_t[1, 1].real
assert close(pk[0], np.cos(phi) ** 2 * r0 + np.sin(phi) ** 2 * r1)
print(f"p_0 = cos^2(phi) rho_00 + sin^2(phi) rho_11 checked; phi = pi/8 gives p_0 = {np.cos(phi) ** 2:.4f} on |0>")

# ---------------------------------------------------------------- Lesson 6: same POVM, different instruments
section("L6  Z measurement vs reset: same probabilities, different post-states")
Zmeas = instrument([[proj(ket0)], [proj(ket1)]])
Reset = instrument([[proj(ket0)], [np.outer(ket0, ket1)]])
for _ in range(5):
    r = random_rho(2)
    for k in range(2):
        assert close(np.trace(Zmeas[k](r)), np.trace(Reset[k](r)))
        assert close(Reset[k](r), np.trace(Zmeas[k](r)) * proj(ket0))
    assert close(Reset[0](r) + Reset[1](r), proj(ket0))
print("same p_k for every input; reset leaves |0> in both branches; sum = reset channel")


# Luders: sqrt(E_k) for the weak measurement is K_k itself (diagonal, positive)
def msqrt(A: C) -> C:
    w, v = eigh(A)
    return v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.conj().T


assert close(msqrt(K0.conj().T @ K0), K0)
# Unitary post-processing gives the same statistics
Uk = [J(0.3), J(1.1)]
E2 = instrument([[Uk[0] @ K0], [Uk[1] @ K1]])
assert all(close(np.trace(E2[k](rho_t)), np.trace(E[k](rho_t))) for k in range(2))

# ---------------------------------------------------------------- Lesson 7: everything is an instrument
section("L7  Forgetting the outcome of a Z instrument is dephasing")
r = random_rho(2)
om = cq([Zmeas[k](r) for k in range(2)])
assert close(ptrace(om, [2, 2], [1]), Delta(r))
assert close(ptrace(om, [2, 2], [0]), np.diag(np.diag(r)))  # POVM: Q -> X
print("Tr_X = dephasing Delta, Tr_Q = the outcome distribution diag(rho)")

# ---------------------------------------------------------------- Lesson 8: Ozawa dilation; mid-circuit measurement
section("L8  CNOT onto an ancilla |0>, then read the ancilla = Z instrument")
for _ in range(5):
    r = random_rho(2)
    big = CNOT @ kron(r, proj(ket0)) @ CNOT.T
    for k in range(2):
        Ek = ptrace(kron(I2, proj(basis(k, 2))) @ big @ kron(I2, proj(basis(k, 2))), [2, 2], [0])
        assert close(Ek, Zmeas[k](r))


# weak version: controlled rotation onto the ancilla
def Ry(a: float) -> C:
    return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)


Uw = kron(proj(ket0), Ry(2 * phi)) + kron(proj(ket1), Ry(np.pi - 2 * phi))
for k in range(2):
    Kk = kron(I2, basis(k, 2)[None, :]) @ Uw @ kron(I2, ket0[:, None])
    assert close(Kk, [K0, K1][k])
print("Z instrument and the weak instrument both come from a unitary + ancilla readout")

# ---------------------------------------------------------------- Lesson 9: feedforward
section("L9  Measure Z, then apply X if the outcome is 1")
for _ in range(3):
    r = random_rho(2)
    om = cq([Zmeas[k](r) for k in range(2)])
    Cmap = sum(kron(proj(basis(k, 2)), np.linalg.matrix_power(X, k)) for k in range(2))
    om2 = Cmap @ om @ Cmap.conj().T
    assert close(om2, kron(np.diag(np.diag(r)), proj(ket0)))
print("output = diag(p) (x) |0><0|: the register is decoupled, the qubit is reset")

# ---------------------------------------------------------------- Lesson 10: deferred measurement
section("L10  Classically controlled X = coherent CNOT, then measure the control")
for _ in range(5):
    r = random_rho(4)
    meas_first = sum(kron(proj(basis(k, 2)), I2) @ r @ kron(proj(basis(k, 2)), I2) for k in range(2))
    Cmap = sum(kron(proj(basis(k, 2)), np.linalg.matrix_power(X, k)) for k in range(2))
    a = Cmap @ meas_first @ Cmap.conj().T
    b = CNOT @ r @ CNOT.conj().T
    b = sum(kron(proj(basis(k, 2)), I2) @ b @ kron(proj(basis(k, 2)), I2) for k in range(2))
    assert close(a, b)
print("equal as cq states for random two-qubit inputs")

# ---------------------------------------------------------------- Lesson 11: adaptive tree
section("L11  Measure Z; on 0 measure X, on 1 measure Z again")
v = np.cos(np.pi / 8) * ket0 + np.sin(np.pi / 8) * ket1
r = proj(v)
Xmeas = instrument([[proj(plus)], [proj(minus)]])
second = {0: Xmeas, 1: Zmeas}
joint = {(k, l): np.trace(second[k][l](Zmeas[k](r))).real for k in range(2) for l in range(2)}
print({kl: round(pv, 4) for kl, pv in joint.items()})
assert close(sum(joint.values()), 1)
assert close(joint[(0, 0)], np.cos(np.pi / 8) ** 2 / 2) and close(joint[(1, 0)], 0)

# ---------------------------------------------------------------- Lessons 12-13: teleportation
section("L12-13  Teleportation as maps on cq states")
phi_plus = (np.kron(ket0, ket0) + np.kron(ket1, ket1)) / np.sqrt(2)
sig = {(a, b): np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b) for a in range(2) for b in range(2)}
bell = {ab: kron(I2, sig[ab]) @ phi_plus for ab in sig}
# identity (<beta_ab| (x) I)|psi>|Phi+> = 1/2 sigma_ab |psi>
for _ in range(3):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    for ab in sig:
        lhs = kron(bell[ab].conj()[None, :], I2) @ np.kron(v, phi_plus)
        assert close(lhs, 0.5 * sig[ab] @ v)
for _ in range(3):
    r = random_rho(2)
    w0 = kron(r, proj(phi_plus))
    keys = sorted(sig)
    blocks_tel = [ptrace(kron(proj(bell[ab]), I2) @ w0 @ kron(proj(bell[ab]), I2), [2, 2, 2], [2]) for ab in keys]
    for ab, bl in zip(keys, blocks_tel, strict=True):
        assert close(bl, 0.25 * sig[ab] @ r @ sig[ab].conj().T)
    om = cq(blocks_tel)
    assert close(ptrace(om, [4, 2], [1]), I2 / 2)  # Bob alone
    assert close(ptrace(om, [4, 2], [0]), np.eye(4) / 4)  # record alone
    corr = [sig[ab].conj().T @ bl @ sig[ab] for ab, bl in zip(keys, blocks_tel, strict=True)]
    assert close(cq(corr), kron(np.eye(4) / 4, r))
    # Bob corrects only X^a (he never receives b): output is Z-dephased rho
    onlyX = sum(
        np.linalg.matrix_power(X, ab[0]) @ bl @ np.linalg.matrix_power(X, ab[0])
        for ab, bl in zip(keys, blocks_tel, strict=True)
    )
    assert close(onlyX, 0.5 * (r + Z @ r @ Z))
    # bit a flipped in transit with probability q: output (1-q) rho + q X rho X
    q = 0.1
    noisy = 0
    for ab, bl in zip(keys, blocks_tel, strict=True):
        for flip in range(2):
            a2 = (ab[0] + flip) % 2
            w = q if flip else 1 - q
            S = sig[(a2, ab[1])]
            noisy = noisy + w * S.conj().T @ bl @ S
    assert close(noisy, (1 - q) * r + q * X @ r @ X)
    noisyb = 0
    for ab, bl in zip(keys, blocks_tel, strict=True):
        for flip in range(2):
            S = sig[(ab[0], (ab[1] + flip) % 2)]
            noisyb = noisyb + (q if flip else 1 - q) * S.conj().T @ bl @ S
    assert close(noisyb, (1 - q) * r + q * Z @ r @ Z)
print("blocks = sigma_ab rho sigma_ab^dag / 4; Bob alone I/2; record alone I/4; corrected I/4 (x) rho")
print("only X corrected -> (rho + Z rho Z)/2; a flipped with prob q -> (1-q) rho + q X rho X")

# ---------------------------------------------------------------- Lessons 14-15: bit-flip code
section("L14-15  Three-qubit bit-flip code: syndrome register and decoding")
pe = 0.1
a, b = 0.6, 0.8j
psiL = (
    a * kron(ket0[:, None], ket0[:, None], ket0[:, None]).ravel()
    + b * kron(ket1[:, None], ket1[:, None], ket1[:, None]).ravel()
)
rhoL = proj(psiL)
Xs = [kron(X, I2, I2), kron(I2, X, I2), kron(I2, I2, X)]
noisy = 0
for pattern in itertools.product(range(2), repeat=3):
    Op = np.eye(8, dtype=complex)
    for j, e in enumerate(pattern):
        if e:
            Op = Xs[j] @ Op
    w = np.prod([pe if e else 1 - pe for e in pattern])
    noisy = noisy + w * Op @ rhoL @ Op.conj().T
Z1Z2 = kron(Z, Z, I2)
Z2Z3 = kron(I2, Z, Z)
Psyn = {}
for s1, s2 in itertools.product(range(2), repeat=2):
    Psyn[(s1, s2)] = (np.eye(8) + (-1) ** s1 * Z1Z2) @ (np.eye(8) + (-1) ** s2 * Z2Z3) / 4
keys = sorted(Psyn)
syn_blocks = [Psyn[s] @ noisy @ Psyn[s] for s in keys]
ps = {s: np.trace(bl).real for s, bl in zip(keys, syn_blocks, strict=True)}
print("syndrome probabilities:", {s: round(v, 4) for s, v in ps.items()})
assert close(ps[(0, 0)], (1 - pe) ** 3 + pe**3)
for s in [(1, 0), (1, 1), (0, 1)]:
    assert close(ps[s], pe * (1 - pe) ** 2 + pe**2 * (1 - pe))


# ancilla circuit realizes the same instrument: two ancillas, CNOTs from data, read ancillas
def cnot(n: int, c: int, t: int) -> C:
    ops0 = [I2] * n
    ops1 = [I2] * n
    ops0[c] = proj(ket0)
    ops1[c] = proj(ket1)
    ops1[t] = X
    return kron(*ops0) + kron(*ops1)


Ucirc = cnot(5, 1, 3) @ cnot(5, 0, 3) @ cnot(5, 2, 4) @ cnot(5, 1, 4)
big = Ucirc @ kron(noisy, proj(ket0), proj(ket0)) @ Ucirc.conj().T
for s, bl in zip(keys, syn_blocks, strict=True):
    Pa = kron(np.eye(8), proj(basis(s[0], 2)), proj(basis(s[1], 2)))
    assert close(ptrace(Pa @ big @ Pa, [8, 4], [0]), bl)
print("ancilla circuit (4 CNOTs, read 2 ancillas) gives the same syndrome blocks")
fix = {(0, 0): np.eye(8), (1, 0): Xs[0], (1, 1): Xs[1], (0, 1): Xs[2]}
corrected = [fix[s] @ bl @ fix[s].conj().T for s, bl in zip(keys, syn_blocks, strict=True)]
XL = kron(X, X, X)
pL = 3 * pe**2 - 2 * pe**3
out = sum(corrected)
assert close(out, (1 - pL) * rhoL + pL * XL @ rhoL @ XL)
post = pe**3 / ((1 - pe) ** 3 + pe**3)
print(f"postselecting on syndrome 00: logical error {post:.5f}, kept with probability {(1 - pe) ** 3 + pe**3:.3f}")
print(
    "after decoding and forgetting the syndrome: (1-pL) rho_L + pL X_L rho_L X_L,",
    f"pL = 3p^2-2p^3 = {pL:.4f} at p = {pe}",
)
assert close(corrected[0] / ps[(0, 0)], ((1 - pe) ** 3 * rhoL + pe**3 * XL @ rhoL @ XL) / ps[(0, 0)])

# ---------------------------------------------------------------- Lessons 16-18: MBQC
section("L16  One MBQC measurement as a cq state")
theta = 0.9
for _ in range(3):
    r = random_rho(2)
    w = CZ @ kron(r, proj(plus)) @ CZ
    bl = [kron(s_theta(s, theta).conj()[None, :], I2) @ w @ kron(s_theta(s, theta)[:, None], I2) for s in range(2)]
    for s in range(2):
        Xs_ = np.linalg.matrix_power(X, s)
        assert close(bl[s], 0.5 * Xs_ @ J(theta) @ r @ J(theta).conj().T @ Xs_)
    avg = bl[0] + bl[1]
    Jr = J(theta) @ r @ J(theta).conj().T
    bloch = lambda m: np.real([np.trace(m @ P_) for P_ in (X, Y, Z)])
    assert close(bloch(avg), bloch(Jr) * np.array([1, 0, 0]))
print("blocks = X^s J rho J^dag X^s / 2; forgetting s keeps only the X component of the Bloch vector")

section("L17  Adaptive angle: second measurement at (-1)^{s1} theta2")
t1, t2 = 0.9, 0.4
for s1 in range(2):
    # after step 1 the state on qubit 2 is X^{s1} J(t1)|+>; step 2 at angle phi gives X^{s2} J(phi) X^{s1} J(t1)|+>
    for adaptive in (False, True):
        ph = (-1) ** s1 * t2 if adaptive else t2
        for s2 in range(2):
            got = np.linalg.matrix_power(X, s2) @ J(ph) @ np.linalg.matrix_power(X, s1) @ J(t1) @ plus
            want = np.linalg.matrix_power(X, s2) @ np.linalg.matrix_power(Z, s1) @ J(t2) @ J(t1) @ plus
            ok = equal_up_to_phase(got, want)
            if adaptive or s1 == 0:
                assert ok
            else:
                fid = abs(np.vdot(want, got)) ** 2
                # no Pauli fixes it: best fidelity over the four Paulis
                best = max(abs(np.vdot(Pp @ want, got)) ** 2 for Pp in (I2, X, Y, Z))
                assert best < 0.99
                if s2 == 0:
                    print(f"non-adaptive, s1=1: best fidelity over Pauli corrections = {best:.4f}")
# Clifford angle: J(-pi/2) = J(pi/2) Z, so the wrong branch differs only by a Pauli
assert equal_up_to_phase(J(-np.pi / 2), J(np.pi / 2) @ Z)
print("adaptive: every branch is X^{s2} Z^{s1} J(t2) J(t1)|+>")

section("L18  Sequential protocol = cluster state first, as cq states")


def op_on(n: int, k: int, A: C) -> C:
    ops = [I2] * n
    ops[k] = A
    return kron(*ops)


def cz(n: int, i: int, j: int) -> C:
    d = 2**n
    diag = np.ones(d, dtype=complex)
    for x in range(d):
        bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[i] and bits[j]:
            diag[x] = -1
    return np.diag(diag)


def measure(rho: C, n: int, k: int, s: int, ang: float) -> C:
    """Unnormalized <s_ang|_k rho |s_ang>_k: qubit k removed."""
    bra = [I2] * n
    bra[k] = s_theta(s, ang).conj()[None, :]
    B = kron(*bra)
    return B @ rho @ B.conj().T


# sequential: P1,P2, CZ12, M1, P3, CZ23, M2(adaptive)
seq = {}
r12 = cz(2, 0, 1) @ proj(np.kron(plus, plus)) @ cz(2, 0, 1)
for s1 in range(2):
    after1 = measure(r12, 2, 0, s1, t1)  # qubit 2 only
    r23 = cz(2, 0, 1) @ kron(after1, proj(plus)) @ cz(2, 0, 1)
    for s2 in range(2):
        seq[(s1, s2)] = measure(r23, 2, 0, s2, (-1) ** s1 * t2)
clu = {}
C3 = cz(3, 1, 2) @ cz(3, 0, 1) @ kron(plus[:, None], plus[:, None], plus[:, None]).ravel()
rC = proj(C3)
for s1 in range(2):
    a1 = measure(rC, 3, 0, s1, t1)
    for s2 in range(2):
        clu[(s1, s2)] = measure(a1, 2, 0, s2, (-1) ** s1 * t2)
for k in seq:
    assert close(seq[k], clu[k])
    want = np.linalg.matrix_power(X, k[1]) @ np.linalg.matrix_power(Z, k[0]) @ J(t2) @ J(t1) @ plus
    assert close(seq[k], 0.25 * proj(want))
print("all four branches agree, each = |X^{s2} Z^{s1} J(t2) J(t1) +><...|/4")

# ---------------------------------------------------------------- Lesson 19: branch-wise entanglement
section("L19  GHZ: measure qubit A in X or in Z")
ghz = (
    kron(ket0[:, None], ket0[:, None], ket0[:, None]) + kron(ket1[:, None], ket1[:, None], ket1[:, None])
).ravel() / np.sqrt(2)
rG = proj(ghz)
for name, basis_vecs in (("X", [plus, minus]), ("Z", [ket0, ket1])):
    bls = [kron(v.conj()[None, :], I2, I2) @ rG @ kron(v[:, None], I2, I2) for v in basis_vecs]
    ents = [vn_entropy(ptrace(bl / np.trace(bl), [2, 2], [0])) for bl in bls]
    avg = bls[0] + bls[1]
    pt = avg.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    print(
        f"measure A in {name}: p = {[round(float(np.trace(b).real), 3) for b in bls]},",
        f"branch entanglement = {np.round(ents, 4)} ebit, averaged rho_BC = diag {np.round(np.diag(avg).real, 3)},",
        f"min eig of partial transpose = {np.linalg.eigvalsh(pt).min():.3f}",
    )
    assert close(avg, 0.5 * (proj(np.kron(ket0, ket0)) + proj(np.kron(ket1, ket1))))
    assert close(np.mean(ents), 1.0 if name == "X" else 0.0)

print("\nAll checks passed.")
