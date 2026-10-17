# mbqc-course

A short course on **measurement-based quantum computation (MBQC)**, taught through small, readable NumPy code.
Six notebooks, each starting from the intuition and then checking it numerically, plus a short lesson 0 for newcomers. PennyLane appears only where it
shows something extra: mid-circuit measurements with feedforward (`pennylane.ftqc`).

| # | Notebook | What you learn |
|---|----------|----------------|
| 0 | [Start here](notebooks/00_start_here.ipynb) | For newcomers: the idea of computing by measuring, one measurement as a gate, any single-qubit gate on a wire |
| 1 | [Graph states](notebooks/01_graph_states.ipynb) | $\vert+\rangle$ + CZ on edges, stabilizers $K_a = X_a\prod Z_b$, cut entanglement, Z measurements delete vertices |
| 2 | [Teleportation and feedforward](notebooks/02_teleportation_and_feedforward.ipynb) | One measurement moves a qubit and applies $J(\theta)=HP(-\theta)$; byproducts and why the next angle flips sign |
| 3 | [Single-qubit rotations](notebooks/03_single_qubit_rotations.ipynb) | Any $U$ on a 5-qubit wire via $XZX$ Euler angles; all 16 outcome branches checked |
| 4 | [CNOT on a 2D patch](notebooks/04_cnot_on_a_2d_patch.ipynb) | Carve a $2\times3$ cluster patch, an edge is a free CZ, byproducts through CZ |
| 5 | [Universality and circuits](notebooks/05_universality_and_circuits.ipynb) | A 20-line circuit-to-pattern compiler; Clifford circuits run in one round; PennyLane's `convert_to_mbqc_formalism` |
| 6 | [Photonics, fusion and noise](notebooks/06_photonics_fusion_and_noise.ipynb) | Z errors = flipped outcomes, entanglement sudden death at $p^*=1-1/\sqrt2$, fusion growth, GKP error vs squeezing (Xanadu's blueprint) |

## The code

* `mbqc/state.py`: a state-vector simulator with **named qubits** (an array with one axis per qubit). Measuring removes the axis.
* `mbqc/patterns.py`: the Pauli frame (byproduct bookkeeping), the 1D wire, the CNOT patch, and a circuit-to-pattern compiler.
* `mbqc/noise.py`: dephasing, negativity, fusion, GKP shift errors.

The central formula of the whole course (lesson 2): measuring qubit 1 of $CZ|\psi\rangle|+\rangle$ at angle $\theta$ with outcome $s$ leaves
$X^s\,H\,P(-\theta)\,|\psi\rangle$ on qubit 2.

## Install

```bash
pip install -e ".[notebooks,test]"
pytest
```

## References

* R. Raussendorf, H. J. Briegel, A one-way quantum computer, [arXiv:quant-ph/0010033](https://arxiv.org/abs/quant-ph/0010033)
* R. Raussendorf, D. E. Browne, H. J. Briegel, Measurement-based quantum computation on cluster states, [arXiv:quant-ph/0301052](https://arxiv.org/abs/quant-ph/0301052)
* D. E. Browne, H. J. Briegel, One-way quantum computation: a tutorial introduction, [arXiv:quant-ph/0603226](https://arxiv.org/abs/quant-ph/0603226)
* M. Hein et al., Entanglement in graph states and its applications, [arXiv:quant-ph/0602096](https://arxiv.org/abs/quant-ph/0602096)
* V. Danos, E. Kashefi, P. Panangaden, The measurement calculus, [arXiv:0704.1263](https://arxiv.org/abs/0704.1263)
* V. Danos, E. Kashefi, Determinism in the one-way model, [arXiv:quant-ph/0506062](https://arxiv.org/abs/quant-ph/0506062)
* D. E. Browne, T. Rudolph, Resource-efficient linear optical quantum computation, [arXiv:quant-ph/0405157](https://arxiv.org/abs/quant-ph/0405157)
* S. Bartolucci et al., Fusion-based quantum computation, [arXiv:2101.09310](https://arxiv.org/abs/2101.09310)
* J. E. Bourassa et al., Blueprint for a scalable photonic fault-tolerant quantum computer, [arXiv:2010.02905](https://arxiv.org/abs/2010.02905)
* D. Gottesman, A. Kitaev, J. Preskill, Encoding a qubit in an oscillator, [arXiv:quant-ph/0008040](https://arxiv.org/abs/quant-ph/0008040)
* H. J. Briegel et al., Measurement-based quantum computation (review), [arXiv:0910.1116](https://arxiv.org/abs/0910.1116)
