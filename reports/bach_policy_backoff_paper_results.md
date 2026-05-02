# vo-regular-bp paper results

Generated on 2026-05-02 for the Bach Prelude exact regular-constraint
experiments after distinguishing fixed-source pure backoff from
Continuator-style constrained policy backoff.

## Repository and environment

- Git branch: `main`
- Git commit: `0c3f0ffced6cfac58b7cadccacf55427bb5763e3`
- Worktree status when generated: dirty
  - `scripts/eval_bach_scalability.py`
  - `tests/test_bach_scalability_smoke.py`
  - `vo_regular_bp/__init__.py`
  - `vo_regular_bp/order_stack_bp.py`
- Python: `Python 3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:21:08) [Clang 20.1.8 ]`
- OS: `Darwin macbookpro-4.home 24.6.0 ... RELEASE_ARM64_T6000 arm64`
- CPU: Apple M1 Pro, 10 cores, 32 GB RAM
- Package version: `vo-regular-bp 0.1.0`
- Runtime dependencies declared by project: none
- Threads/processes: single Python process, no multiprocessing

## Commands

Tests:

```bash
python -m pytest -q
```

Result:

```text
14 passed in 1.44s
```

Bach paper run:

```bash
python scripts/eval_bach_scalability.py \
  --source-policy all \
  --orders 1 2 3 4 5 6 \
  --horizons 32 \
  --maxorder-grams 5 \
  --samples 100 \
  --seed 0 \
  --warmups 1 \
  --repeats 5 \
  --output outputs/bach_scalability_policy_stack_paper.csv \
  --raw-output outputs/bach_scalability_policy_stack_paper_raw.csv
```

CSV outputs:

```text
outputs/bach_scalability_policy_stack_paper.csv
outputs/bach_scalability_policy_stack_paper_raw.csv
```

The reported medians use one warmup sweep plus five measured repeats. `bp_s`
excludes corpus parsing, source graph/model construction, and acceptor
construction. It includes the regular BP/message computation. Sampling time is
reported separately per generated sequence.

## Experimental setup

- Corpus: `data/bach_prelude_c_major_pitches.txt`
- Corpus length: `592` pitch events
- Alphabet size: `25`
- Prefix: `60 64 67 72 76 67`
- Horizon: `n = 32`
- Maximum copied n-gram constraint: `M = 5`
- Final pitch-class constraint: generated final pitch class is C
- Samples: `100` generated sequences per configuration
- Random seed base: `0`

The regular constraint is the intersection of:

1. a positional final pitch-class acceptor;
2. a forbidden-substring acceptor rejecting every generated 5-gram observed in
   the Bach pitch corpus.

No pruning, beam search, bounded MDD, or approximate filtering is used.

## Semantics

### A. Fixed longest-observed-suffix source

`source_policy=pure` builds one stochastic source graph before constraints.
Every context state uses the longest observed suffix up to order `K`, and
outgoing probabilities are normalized continuation counts at that suffix. The
reported value `partition_function` is the exact constrained mass:

```text
Z = P_source(x in L(A))
```

BP samples exactly from `P_source(x | x in L(A))` when `Z > 0`. Under the Bach
MAXORDER-5 constraint, this fixed source has zero constrained mass for `K >= 3`
with the reported prefix and horizon. This is not the intended practical
backoff policy; it is a diagnostic showing why a single preselected longest
suffix source is too rigid for MAXORDER constraints.

### B. ContextBPContinuator-style constrained policy backoff

`source_policy=policy_stack` is the main Bach experiment. It is not exact
conditioning of one fixed stochastic source; it is an exact constrained
generation policy over an order stack. For each order, BP messages compute
feasible future mass under the same regular acceptor. At each generation step:

1. candidate orders are considered from high to low;
2. orders with zero feasible future mass are skipped;
3. `LongestFeasiblePolicy` selects the highest order with positive future mass;
4. symbols at the selected order are sampled proportionally to
   `p_order(y | context) * beta_next`;
5. probabilities are normalized over the feasible outgoing edges of that
   selected order.

The reported `partition_function` column is therefore a policy success mass,
not a fixed-source partition function. In this experiment it is `1` for all K:
the policy always finds a constrained continuation. Samples are guaranteed by
construction to satisfy final C and MAXORDER-5.

This is the relevant ContextBPContinuator-style behavior: higher orders are
preferred when they can lead to an accepted completion, and the generator backs
off only when the regular constraint makes the higher-order continuation
infeasible.

No singleton avoidance is included in the main policy-stack run.

### C. Smoothed suffix-mixture source

`source_policy=mixture` is a secondary comparison. It is again exact
conditioning of one fixed stochastic graph, but the source graph mixes suffix
orders using `backoff_weight = 0.25`, preserving lower-order support at every
context.

## Main Bach result: constrained policy backoff

| K | \|T\| | \|E_T\| | \|Q\| | Reachable product states | Reachable product edges | \|Q\|\|E_T\| | \|V\|^K | Success mass | BP s | Sample s/seq | Violations | Avg copied span |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 117 | 21681 | 16253 | 74115 | 2536677 | 25 | 1 | 0.279788 | 0.00034700 | 0 | 4.00 |
| 2 | 167 | 449 | 21681 | 22059 | 82395 | 9734769 | 625 | 1 | 0.321633 | 0.00047585 | 0 | 4.00 |
| 3 | 524 | 1080 | 21681 | 23752 | 83162 | 23415480 | 15625 | 1 | 0.327898 | 0.00054018 | 0 | 4.00 |
| 4 | 1180 | 2062 | 21681 | 23784 | 83195 | 44706222 | 390625 | 1 | 0.322440 | 0.00058276 | 0 | 4.00 |
| 5 | 2187 | 3437 | 21681 | 23751 | 83195 | 74517597 | 9765625 | 1 | 0.316798 | 0.00059253 | 0 | 4.00 |
| 6 | 3587 | 5246 | 21681 | 23826 | 83207 | 113738526 | 244140625 | 1 | 0.318622 | 0.00059951 | 0 | 4.00 |

### K=6 policy-stack order counts

Each repeat sampled `100 x 32 = 3200` events.

| Repeat | Order histogram |
|---:|---|
| 0 | `2:3114 3:86` |
| 1 | `2:3106 3:94` |
| 2 | `2:3110 3:90` |
| 3 | `2:3104 3:96` |
| 4 | `2:3099 3:101` |

Start fixed-order masses for policy-stack K=6:

```text
1:0.00695492288417 2:3.2120549058e-09 3:0 4:0 5:0 6:0
```

This explains the sampled order behavior: the policy usually has to back off
to order 2 under MAXORDER-5, with occasional order-3 feasible choices later in
the generated sequence.

## Diagnostic: fixed pure longest-suffix source

| K | \|T\| | \|E_T\| | \|Q\| | Reachable product states | Reachable product edges | \|Q\|\|E_T\| | \|V\|^K | Z | BP s | Sample s/seq | Samples generated | Violations | Avg copied span |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 142 | 21681 | 16253 | 74115 | 3078702 | 25 | 6.955e-03 | 0.708601 | 0.00013506 | 100 | 0 | 4.00 |
| 2 | 143 | 357 | 21681 | 5806 | 8280 | 7740117 | 625 | 3.212e-09 | 0.102857 | 0.00005865 | 100 | 0 | 4.00 |
| 3 | 358 | 656 | 21681 | 34 | 33 | 14222736 | 15625 | 0 | 0.000500 | 0.00000000 | 0 | 0 | 0.00 |
| 4 | 657 | 1007 | 21681 | 22 | 21 | 21832767 | 390625 | 0 | 0.000381 | 0.00000000 | 0 | 0 | 0.00 |
| 5 | 1008 | 1400 | 21681 | 22 | 21 | 30353400 | 9765625 | 0 | 0.000395 | 0.00000000 | 0 | 0 | 0.00 |
| 6 | 1401 | 1834 | 21681 | 22 | 21 | 39762954 | 244140625 | 0 | 0.000395 | 0.00000000 | 0 | 0 | 0.00 |

The zero-mass rows are exact results for the fixed-source diagnostic. They do
not mean that the Bach MAXORDER problem has no solution. They mean that the
particular fixed longest-suffix source has no accepted path at those orders
because it is not allowed to back off after seeing future infeasibility. The
policy-stack experiment above is the intended backoff solution.

## Secondary comparison: smoothed suffix-mixture source

| K | \|T\| | \|E_T\| | \|Q\| | Reachable product states | Reachable product edges | \|Q\|\|E_T\| | \|V\|^K | Z | BP s | Sample s/seq | Violations | Avg copied span | Avg latent order |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 650 | 21681 | 19003 | 448354 | 14092650 | 25 | 3.417e-02 | 3.210852 | 0.00026667 | 0 | 3.98 | 0.765 |
| 2 | 143 | 3575 | 21681 | 19003 | 448354 | 77509575 | 625 | 3.298e-05 | 3.167956 | 0.00025665 | 0 | 4.00 | 1.437 |
| 3 | 358 | 8950 | 21681 | 19003 | 448354 | 194044950 | 15625 | 2.268e-10 | 3.200353 | 0.00125238 | 0 | 4.00 | 1.605 |
| 4 | 657 | 16425 | 21681 | 19003 | 448354 | 356110425 | 390625 | 1.288e-13 | 3.301162 | 0.00025361 | 0 | 3.99 | 1.362 |
| 5 | 1008 | 25200 | 21681 | 19003 | 448354 | 546361200 | 9765625 | 3.308e-14 | 3.204173 | 0.00027082 | 0 | 4.00 | 1.468 |
| 6 | 1401 | 35025 | 21681 | 19003 | 448354 | 759377025 | 244140625 | 8.455e-15 | 3.233282 | 0.00026686 | 0 | 4.00 | 1.566 |

K=6 mixture latent-order histogram, median row:

```text
0:942 1:995 2:663 3:260 4:16 5:67 6:257
```

## Raw timing runs

### Policy stack BP times

| K | Raw BP times (s) | Raw sample times / seq (s) |
|---:|---|---|
| 1 | `0.291119, 0.294342, 0.279788, 0.276418, 0.272213` | `0.00032503, 0.00037288, 0.00034983, 0.00034700, 0.00034686` |
| 2 | `0.316465, 0.321724, 0.321633, 0.322637, 0.314593` | `0.00046058, 0.00048340, 0.00049404, 0.00046765, 0.00047585` |
| 3 | `0.337208, 0.324621, 0.336782, 0.327898, 0.318643` | `0.00053296, 0.00057360, 0.00058222, 0.00054018, 0.00052761` |
| 4 | `0.313678, 0.331333, 0.322440, 0.337292, 0.318270` | `0.00054233, 0.00061860, 0.00058623, 0.00056196, 0.00058276` |
| 5 | `0.310244, 0.327675, 0.313813, 0.316798, 0.344527` | `0.00055476, 0.00062769, 0.00058126, 0.00059253, 0.00059426` |
| 6 | `0.313267, 0.322313, 0.313993, 0.335465, 0.318622` | `0.00057711, 0.00060893, 0.00058452, 0.00059951, 0.00060426` |

### Fixed pure BP times

| K | Raw BP times (s) | Raw sample times / seq (s) |
|---:|---|---|
| 1 | `0.696355, 0.775859, 0.768932, 0.674512, 0.708601` | `0.00010682, 0.00012422, 0.00013826, 0.00013821, 0.00013506` |
| 2 | `0.102768, 0.102857, 0.102855, 0.109478, 0.105290` | `0.00005318, 0.00004599, 0.00006574, 0.00007564, 0.00005865` |
| 3 | `0.000494, 0.000515, 0.000484, 0.000744, 0.000500` | `0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000` |
| 4 | `0.000381, 0.000366, 0.000367, 0.000476, 0.000415` | `0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000` |
| 5 | `0.000367, 0.000359, 0.000395, 0.000434, 0.000401` | `0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000` |
| 6 | `0.000403, 0.000395, 0.000371, 0.000391, 0.000459` | `0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000` |

### Mixture BP times

| K | Raw BP times (s) | Raw sample times / seq (s) |
|---:|---|---|
| 1 | `3.126448, 3.210852, 3.194302, 3.402521, 3.244516` | `0.00124286, 0.00022725, 0.00025094, 0.00026821, 0.00026667` |
| 2 | `3.085799, 3.146608, 3.303626, 3.167956, 3.230046` | `0.00025709, 0.00023556, 0.00026652, 0.00025665, 0.00024953` |
| 3 | `3.289893, 3.200353, 3.170280, 3.000276, 3.263898` | `0.00131788, 0.00023036, 0.00025951, 0.00125238, 0.00132321` |
| 4 | `3.073541, 3.537870, 3.394325, 3.102086, 3.301162` | `0.00025071, 0.00025361, 0.00145704, 0.00022642, 0.00026835` |
| 5 | `3.408020, 3.357975, 3.177235, 3.138516, 3.204173` | `0.00155904, 0.00026853, 0.00027082, 0.00029732, 0.00026984` |
| 6 | `3.633221, 3.225377, 3.278119, 3.119670, 3.233282` | `0.00032065, 0.00027172, 0.00024908, 0.00024047, 0.00026686` |

## LaTeX replacement rows

### Main policy-stack size rows

```latex
1 & 25 & 117 & 21681 & 16253 & 74115 & 2536677 & 25 \\
2 & 167 & 449 & 21681 & 22059 & 82395 & 9734769 & 625 \\
3 & 524 & 1080 & 21681 & 23752 & 83162 & 23415480 & 15625 \\
4 & 1180 & 2062 & 21681 & 23784 & 83195 & 44706222 & 390625 \\
5 & 2187 & 3437 & 21681 & 23751 & 83195 & 74517597 & 9765625 \\
6 & 3587 & 5246 & 21681 & 23826 & 83207 & 113738526 & 244140625 \\
```

### Main policy-stack runtime rows

```latex
1 & 1 & 0.279788 & 0.00034700 & 0 & 4.00 \\
2 & 1 & 0.321633 & 0.00047585 & 0 & 4.00 \\
3 & 1 & 0.327898 & 0.00054018 & 0 & 4.00 \\
4 & 1 & 0.322440 & 0.00058276 & 0 & 4.00 \\
5 & 1 & 0.316798 & 0.00059253 & 0 & 4.00 \\
6 & 1 & 0.318622 & 0.00059951 & 0 & 4.00 \\
```

### Fixed pure diagnostic runtime rows

```latex
1 & 6.955e-03 & 0.708601 & 0.00013506 & 0 & 4.00 \\
2 & 3.212e-09 & 0.102857 & 0.00005865 & 0 & 4.00 \\
3 & 0.000e+00 & 0.000500 & 0.00000000 & 0 & 0.00 \\
4 & 0.000e+00 & 0.000381 & 0.00000000 & 0 & 0.00 \\
5 & 0.000e+00 & 0.000395 & 0.00000000 & 0 & 0.00 \\
6 & 0.000e+00 & 0.000395 & 0.00000000 & 0 & 0.00 \\
```

## Suggested paper paragraph

The Bach experiment uses a ContextBPContinuator-style constrained backoff
policy. For each order in the stack, BP messages compute whether that order can
lead to an accepted completion under the regular constraints. Generation tries
higher orders first, backs off only when the higher-order feasible future mass
is zero, and then samples from the selected order with probabilities normalized
over `p_order(y | context) beta_next`. This is the intended variable-order
backoff mechanism: regular constraints guide the order choice without pruning,
beam search, or constraint relaxation. As a diagnostic, we also report the
fixed longest-suffix source, which defines one stochastic graph before
constraints; exact BP shows that this naive fixed source has zero constrained
mass for K >= 3 under MAXORDER-5, precisely motivating constrained backoff.

## Tiny exactness results

The tiny exactness results are unchanged by the Bach source-policy
clarification.

Integer future-position example:

| Sequence | Brute mass | Exact conditional | BP conditional | Sample frequency |
|---:|---:|---:|---:|---:|
| `(2,4)` | `0.4761904762` | `0.9090909091` | `0.9090909091` | `0.9108` |
| `(3,4)` | `0.04761904762` | `0.09090909091` | `0.09090909091` | `0.0892` |

Summary:

| Metric | Value |
|---|---:|
| `Z_brute` | `0.52380952381 = 11/21` |
| `Z_BP` | `0.52380952381 = 11/21` |
| `TV(exact, BP)` | `0` |
| `TV(exact, empirical)` | `0.00170909` |
| Constraint violations | `0` |

Forbidden substring example:

| Metric | Value |
|---|---:|
| `Z_brute` | `0.684766214178` |
| `Z_BP` | `0.684766214178` |
| `TV(exact, BP)` | `4.59702e-17` |
| `TV(exact, empirical)` | `0.00924597` |
| Constraint violations | `0` |
