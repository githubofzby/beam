# BEAM-X Stage 5A: Fixed-Candidate Commit Policy Audit

## Scope

Phase A keeps the legacy candidate source, exact-scored candidate graph,
threshold, and discovery order fixed for each epoch. It changes only endpoint
matching priority, commit order, micro-batch size, and current-state marginal
validation. Candidate discovery is never rerun between micro-batches.

The opt-in policies are:

| Policy | Semantics |
|---|---|
| `g0` | Existing greedy endpoint-disjoint full batch |
| `s1` | Fixed candidate graph, current-gain ordered commits one at a time |
| `t2` | Transactional current-gain ordering in micro-batches of two |
| `t4` | Transactional current-gain ordering in micro-batches of four |
| `t8` | Transactional current-gain ordering in micro-batches of eight |
| `m4` | Mutual-best candidates first, then transactional batches of four |

For transactional policies, remaining endpoint-disjoint pairs are rescored at
each micro-batch boundary. The chosen micro-batch is ordered by current gain,
then minimum endpoint id, then maximum endpoint id. Every pair is rescored on
the latest quotient state immediately before commit and is accepted only for a
strictly positive marginal gain.

Because each accepted merge has positive current-state exact gain, the sum of
accepted marginals proves that the transaction lowers exact quotient cost; no
rollback or production-path full `ExactCost()` is required. Optional audit mode
checks the before/after quotient cost and records trajectory rows. Its time is
reported as `audit_oracle_ms`, separate from required
`commit_validation_ms`.

This phase does not modify CostOracle, candidate discovery, exact scoring,
threshold acceptance, quotient update formulas, encoding, or CUDA semantics.
Phase B candidate refresh is explicitly out of scope until Phase A passes its
compression feasibility gate.
