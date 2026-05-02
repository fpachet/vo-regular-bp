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
