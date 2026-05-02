# Optimization benchmark

This note compares commit `85f5cf3` against the optimized working tree after
three exactness-preserving changes:

1. Fast fixed-length forbidden-substring transition checks.
2. Cached transition weights in `ProductBPResult` for repeated sampling.
3. Cached graph/acceptor construction in `scripts/eval_bach_scalability.py`.

No pruning, beam search, bounded MDDs, or approximate inference was introduced.

## Default Bach sweep

Command:

```bash
/usr/bin/time -p python scripts/eval_bach_scalability.py --samples 100
```

Configuration:

- `K in {1,2,3,4,5,6}`
- `n = 32`
- `M = 5`
- final pitch class `C`
- `100` samples per `K`

| Metric | Before | After | Speedup |
|---|---:|---:|---:|
| Wall time | `22.93s` | `20.43s` | `1.12x` |
| Context graph build time, summed | `0.386319s` | `0.373681s` | `1.03x` |
| Acceptor build time, summed | `2.525190s` | `0.102299s` | `24.68x` |
| BP time, summed | `18.653209s` | `18.524921s` | `1.01x` |

The primary win is acceptor construction. In this sweep all six rows share the
same `(n, M, final pitch class, alphabet)` constraint, so the composite acceptor
is built once and reused. The fixed-length forbidden-pattern check also reduces
the first acceptor build itself from roughly `0.41s` to `0.10s`.

## Sampling-heavy representative run

Command:

```bash
/usr/bin/time -p python scripts/eval_bach_scalability.py \
  --orders 4 --horizons 32 --maxorder-grams 5 --samples 5000
```

| Metric | Before | After | Speedup |
|---|---:|---:|---:|
| Wall time | `5.18s` | `4.51s` | `1.15x` |
| Sampling time / sequence | `0.00020521s` | `0.00013917s` | `1.47x` |
| BP time | `3.057735s` | `3.050018s` | `1.00x` |

This isolates the transition-weight cache: BP is essentially unchanged, but
repeated exact sampling from the same BP table is faster.

## Interpretation

The easy optimizations improve the evaluation script without changing the
algorithmic result:

- Constraints and generated examples are unchanged under the same RNG seed.
- Partition functions are unchanged.
- Constraint violations remain `0`.

The remaining runtime is dominated by the BP pass over the reachable product.
Further gains would likely require structural changes such as integer-state
interning or a time-aware final-symbol constraint that avoids multiplying the
automaton by the positional DFA state space.

## Positional-only Bach benchmark

A later optimization targets the special case where the constraint is purely
positional and can be applied as time-indexed edge masks instead of a regular
acceptor product. The baseline path built the full backoff `ContextGraph` before
running BP. The optimized path keeps exact counts and normalized suffix
distributions, compiles outgoing context edges only on demand, and runs memoized
backward DP over reachable `(time, context)` states.

Command:

```bash
python scripts/eval_bach_positional_direct.py --compare-baseline --repeats 50 --samples 1
```

Configuration:

- Bach Prelude in C, pitch-only
- `K = 4`
- `n = 32`
- prefix `(60, 64, 67, 72, 76, 67)`
- generated first and last pitch class `C`

| Metric | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| Total time / run | `0.133772s` | `0.083307s` | `1.61x` |
| Model/graph build | `0.060995s` | `0.001337s` | `45.62x` |
| BP time | `0.072312s` | `0.081520s` | `0.89x` |
| Sampling time / sequence | `0.000236s` | `0.000235s` | `1.00x` |

Exactness checks are unchanged:

- `Z = 0.102472718036` in both paths.
- Reachable time-indexed states: `18452` in both paths.
- Reachable edges: `444096` in both paths.
- Constraint violations: `0`.

For `100` samples per BP table, the same benchmark gives:

| Metric | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| Total time / run | `0.143116s` | `0.082563s` | `1.73x` |
| Sampling time / 100 sequences | `0.012954s` | `0.011406s` | `1.14x` |

The main win is avoiding full graph materialization when only one prefix,
horizon, and positional constraint set are needed. The BP frontier is the same;
the optimization removes setup work without pruning or approximating paths.

## Order-stack comparison backend

The Continuator ContextBP engine is faster than explicit-backoff BP because it
does not run BP on the dense explicit backoff-mixture graph. It keeps one sparse
fixed-order graph per order and chooses among orders with a policy. To make this
comparison explicit, `vo_regular_bp.order_stack_bp` implements the same style of
backend separately from the paper's single-context-graph BP.

This is intentionally a different model family:

- `vo_regular_bp_direct` is exact BP for one explicit stochastic context graph.
- `vo_regular_bp_order_stack` is exact backward messaging inside each fixed-order
  graph plus policy-based order selection.

Command:

```bash
python scripts/eval_bach_continuator_compare.py --repeats 50 --samples 1
```

| Method | Total mean | Context states | Context edges | Edge relax upper bound |
|---|---:|---:|---:|---:|
| `vo_regular_bp_direct` | `0.0823945s` | `657` | `16400` | `444096` |
| `vo_regular_bp_order_stack` | `0.0113387s` | `1210` | `2102` | `67264` |
| `continuator_classic` | `0.0056632s` | `661` | `991` | |
| `continuator_context_bp` | `0.0211527s` | `1210` | `2102` | |

The in-repository order-stack backend matches Continuator ContextBP's stack size
for this setup and, with the same RNG seed, produces the same representative
sample and order trace. This supports the diagnosis that the previous speed gap
was mostly semantic/workload-related rather than evidence that the single-graph
implementation was unusually inefficient.
