# Exact constrained sampling with sparse variable-order context graphs

This report summarizes two evaluations for the `vo_regular_bp` artifact. The
first verifies exactness against brute force on tiny integer examples. The
second measures scalability on a pitch-only Bach Prelude in C corpus, where
brute-force enumeration is infeasible.

All experiments use the same algorithmic object: backward dynamic programming
on the reachable product of a sparse context graph and a deterministic regular
acceptor. No approximate pruning, beam search, bounded MDDs, or heuristic
filtering is used.

## Model and constraints

The context graph has states equal to observed suffix contexts up to order `K`.
For exactness experiments, continuation probabilities are maximum-likelihood
counts with longest-suffix canonical updates. For the Bach scalability
experiment, the graph uses an explicit variable-order backoff mixture:

```text
p(y | s) proportional to sum_{k=0}^{|s|} alpha^(|s|-k) p_MLE(y | suffix_k(s))
```

with `alpha = 0.25` in the reported run. This is still an exact sparse
variable-order context graph: BP is run on the resulting explicit transition
probabilities. For each sampled pitch, the implementation also records a latent
order sampled from the posterior contribution of each suffix order to the
selected symbol.

The regular constraints are represented as deterministic acceptors:

- Positional/final-pitch constraints use `positional_acceptor`.
- Non-positional MAXORDER constraints use `forbidden_substring_acceptor`, which
  rejects generated pitch `M`-grams that occur in the training pitch sequence.
- Composite constraints are formed by DFA intersection.

## Tiny exactness evaluation

Command:

```bash
python scripts/eval_tiny_exactness.py --samples 20000
```

### Integer future-position example

Training multiset:

```text
10   x [0,1,2,4]
10   x [0,1,3,5]
1    x [0,1,3,4]
1000 x [6,2,5]
1000 x [6,3,4]
```

With `K = 2`, prefix `(0,1)`, horizon `2`, and positional constraint `x1 = 4`,
the sampler must choose `x0` while accounting for the future constrained mass at
`x1`. The exact conditional distribution is recovered:

| Sequence | Brute mass | Exact conditional | BP conditional | Sample frequency |
|---:|---:|---:|---:|---:|
| `(2,4)` | `0.4761904762` | `0.9090909091` | `0.9090909091` | `0.9108` |
| `(3,4)` | `0.04761904762` | `0.09090909091` | `0.09090909091` | `0.0892` |

Summary:

| Metric | Value |
|---|---:|
| `Z_brute` | `0.52380952381 = 11/21` |
| `Z_BP` | `0.52380952381 = 11/21` |
| `|Z_brute - Z_BP|` | `0` |
| `TV(exact, BP)` | `0` |
| `TV(exact, empirical)` | `0.00170909` |
| Constraint violations | `0` |
| Context states / edges | `11 / 23` |
| Acceptor states | `3` |
| Reachable product states / edges | `4 / 4` |

### Non-positional forbidden substring example

The second tiny experiment uses a regular, non-positional constraint: generated
integer sequences must avoid substring `(2,2)`. The DFA has states `q0`, `q1`,
and `dead`, where `q1` means the previous generated symbol was `2`.

Summary:

| Metric | Value |
|---|---:|
| Horizon | `4` |
| `Z_brute` | `0.684766214178` |
| `Z_BP` | `0.684766214178` |
| `|Z_brute - Z_BP|` | `0` |
| `TV(exact, BP)` | `4.59702e-17` |
| `TV(exact, empirical)` | `0.00924597` |
| Constraint violations | `0` |
| Context states / edges | `18 / 40` |
| Acceptor states | `3` |
| Reachable product states, unique / time-indexed | `19 / 32` |
| Reachable product edges | `36` |

These two experiments verify that the product-BP partition function matches
exhaustive enumeration and that the sampler draws from the exact constrained
conditional distribution.

## Bach Prelude scalability evaluation

Command:

```bash
python scripts/eval_bach_scalability.py --samples 100
```

The data file is:

```text
data/bach_prelude_c_major_pitches.txt
```

It contains a deterministic static pitch-only MIDI-number sequence for the Bach
Prelude in C evaluation. The corpus length is `592` pitch events and the
alphabet size is `25`. The reported sweep uses:

- `K in {1,2,3,4,5,6}`
- horizon `n = 32`
- MAXORDER constraint `M = 5`, rejecting any generated 5-gram seen in training
- final pitch class `C`
- `100` samples per configuration

The CSV output is written to:

```text
outputs/bach_scalability.csv
```

### Scalability table

| K | \|T\| | \|E_T\| | \|Q\| | Reachable product states | Reachable product edges | \|Q\|*\|E_T\| | Dense states \|Sigma\|^K | BP time (s) | Sample time / seq (s) | Z | Violations | Avg copied span | Avg latent order |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 650 | 21681 | 19003 | 448354 | 14092650 | 25 | 2.898401 | 0.00018051 | `3.41675743685e-02` | 0 | 3.940 | 0.767 |
| 2 | 143 | 3575 | 21681 | 19003 | 448354 | 77509575 | 625 | 2.937471 | 0.00018049 | `3.29819197814e-05` | 0 | 4.000 | 1.415 |
| 3 | 358 | 8950 | 21681 | 19003 | 448354 | 194044950 | 15625 | 2.937680 | 0.00019604 | `2.26791792961e-10` | 0 | 4.000 | 1.604 |
| 4 | 657 | 16425 | 21681 | 19003 | 448354 | 356110425 | 390625 | 3.029545 | 0.00022720 | `1.28845205776e-13` | 0 | 3.990 | 1.369 |
| 5 | 1008 | 25200 | 21681 | 19003 | 448354 | 546361200 | 9765625 | 3.106625 | 0.00022368 | `3.30757202064e-14` | 0 | 4.000 | 1.471 |
| 6 | 1401 | 35025 | 21681 | 19003 | 448354 | 759377025 | 244140625 | 3.148563 | 0.00022757 | `8.45471416734e-15` | 0 | 3.990 | 1.539 |

The reachable product remains small relative to both naive baselines. At
`K = 6`, the exact reachable product has `19,003` unique states and `448,354`
product edges. The full product edge upper bound is `759,377,025`, and the
dense lifted state count is `25^6 = 244,140,625`.

### Latent order usage

For the backoff model, each selected symbol may receive probability mass from
several suffix orders. The following histograms count the sampled latent orders
over `100 x 32 = 3200` generated events per configuration:

| K | Latent order histogram |
|---:|---|
| 1 | `0:745 1:2455` |
| 2 | `0:456 1:959 2:1785` |
| 3 | `0:617 1:877 2:863 3:843` |
| 4 | `0:933 1:1001 2:679 3:326 4:261` |
| 5 | `0:970 1:969 2:653 3:274 4:60 5:274` |
| 6 | `0:970 1:1019 2:639 3:234 4:23 5:55 6:260` |

This confirms that the large-scale experiment is genuinely variable-order:
sampled events use a mixture of suffix orders, including high-order contexts
where they have posterior support, rather than collapsing to a first-order
chain.

### Example generated sequence

Representative configuration: `K = 4`, `n = 32`, `M = 5`.

```text
pitches:
72 76 60 64 55 50 69 76 60 65 50 62 67 62 64 64 69 76 65 69 81 67 71 74 77 64 62 64 62 76 79 72

latent orders:
4 4 4 3 2 0 0 1 1 1 1 0 1 0 0 0 1 2 0 1 0 1 2 3 1 0 0 0 0 0 1 2
```

The MAXORDER constraint is exact: no generated sequence contains a training
5-gram, and the average longest copied span is approximately `4`, as expected
for an `M = 5` forbidden-substring constraint.

## Main conclusion

The tiny experiments establish exactness by matching brute-force enumeration.
The Bach experiment demonstrates that the same exact product-BP computation can
be run on a realistic musical sequence using sparse context states and regular
constraints. The reachable sparse product is orders of magnitude smaller than
the full product edge upper bound and the dense `|Sigma|^K` lifted state space,
while preserving exact regular-constraint conditioning and exact sampling from
the constrained distribution.
