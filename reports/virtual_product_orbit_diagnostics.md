# Virtual Product-Orbit Diagnostics

Generated on: `2026-05-03`

Branch: `virtual-data-augmentation`

Base commit before these diagnostics: `3edeab9968063f57c7ec63a3eb37880c2fe9c75e`

## Purpose

This report probes the next optimization frontier for virtual data
augmentation: reducing regular-product BP work, not only model storage or
graph materialization.

The current virtual augmentation backend preserves the same continuation
semantics as explicitly materializing all transpositions, but regular BP still
expands many product edges over absolute symbols. The diagnostic added in this
branch counts how much of the already-expanded product is repeated modulo a
common integer transposition.

This is not a new sampler and does not change the target distribution.

The branch now also includes the first safe hot-loop optimization suggested by
these diagnostics: each regular BP cache stores the DFA-accepted transition row
for an actual `(context state, DFA state)` product row. Positional constraints
are still applied at the current time, after retrieving the cached row, so
absolute anchor semantics are unchanged.

## Commands

The diagnostic script was first run before the row-cache optimization:

```bash
python scripts/eval_virtual_product_orbits.py \
  --output-dir outputs/virtual_product_orbits_2026_05_03
```

That historical run wrote:

```text
outputs/virtual_product_orbits_2026_05_03/virtual_product_orbit_diagnostics.csv
```

After adding the regular transition-row cache, the same diagnostic was rerun on
the current optimized code:

```bash
python scripts/eval_virtual_product_orbits.py \
  --output-dir outputs/virtual_product_orbits_2026_05_03_row_cache
```

The optimized CSV is:

```text
outputs/virtual_product_orbits_2026_05_03_row_cache/virtual_product_orbit_diagnostics.csv
```

The stricter exact row-signature diagnostic was then run:

```bash
python scripts/eval_virtual_product_orbits.py \
  --output-dir outputs/virtual_product_orbits_2026_05_03_exact_signatures
```

That CSV is:

```text
outputs/virtual_product_orbits_2026_05_03_exact_signatures/virtual_product_orbit_diagnostics.csv
```

## Setup

- Corpus: Bach Prelude pitch-only sequence, 592 tokens
- Virtual augmentation: integer shifts `-6 -5 -4 -3 -2 -1 0 1 2 3 4 5`
- Virtual corpus size: 7104 tokens
- Max context order: `K = 4`
- Horizon: `n = 64`
- Positional constraint: ascending bar-start anchors
  `60 62 64 65 67 69 71 72` at positions `0 8 16 24 32 40 48 56`
- Regular constraint: MAXORDER / forbidden copied n-grams from the full
  12-key augmented corpus
- Source backend: current exact `VirtualAugmentedOrderStackModel`
- Policy used to initialize the regular order-stack result:
  `LongestFeasiblePolicy`

## Results

| copy n-gram | BP time (s) | product states | state orbits | state reduction | product edges | time edge orbits | time edge reduction | structural edge orbits | structural edge reduction | DFA prefix states | DFA prefix orbits | DFA prefix reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.450241 | 4693 | 290 | 16.18x | 1446142 | 999416 | 1.45x | 40657 | 35.57x | 4806 | 290 | 16.57x |
| 6 | 0.549778 | 7811 | 503 | 15.53x | 2006934 | 1414999 | 1.42x | 71317 | 28.14x | 7956 | 503 | 15.82x |
| 7 | 0.679160 | 11432 | 756 | 15.12x | 2310795 | 1670439 | 1.38x | 108344 | 21.33x | 11609 | 756 | 15.36x |
| 8 | 0.747571 | 15515 | 1048 | 14.80x | 2468719 | 1825129 | 1.35x | 148886 | 16.58x | 15753 | 1048 | 15.03x |

## Row-Cache Optimization

The row cache is exact but deliberately conservative. It caches the regular
transition row for each actual product state:

```text
(context graph state, DFA state) -> accepted outgoing transitions
```

It does not cache beta values, and it does not quotient away absolute pitch
information. Therefore each time-dependent positional mask is still applied
with the original absolute symbol.

Single-run timings on the same script:

| copy n-gram | before BP time (s) | after BP time (s) | speedup | cached rows | accepted transitions cached | row cache hits | row cache misses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.450241 | 0.328443 | 1.37x | 4693 | 62231 | 137808 | 4693 |
| 6 | 0.549778 | 0.508163 | 1.08x | 7811 | 109199 | 192150 | 7811 |
| 7 | 0.679160 | 0.646080 | 1.05x | 11432 | 166274 | 227530 | 11432 |
| 8 | 0.747571 | 0.749883 | 1.00x | 15507 | 231055 | 242331 | 15507 |

The optimization helps most when row construction and DFA transition lookup are
a visible part of runtime. For larger MAXORDER windows, the remaining cost is
dominated by the exact beta recurrence over many time-dependent edge
relaxations, so this cache alone does not realize the larger orbit reductions.

## Exact Row-Signature Diagnostic

The next question was whether we could safely quotient rows beyond actual
`(context state, DFA state)` caching. The stricter diagnostic groups two rows
only if the full row is identical after an integer shift:

- canonical source context and DFA suffix;
- every accepted emitted symbol;
- successor context;
- successor DFA suffix;
- transition probability.

This means finite transposition-support boundary effects and nonuniform
continuation probabilities split into separate signatures.

| copy n-gram | product rows | exact row signatures | exact row reduction | reusable row fraction | max row signature size | time-masked rows | time-masked signatures | time-masked row reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 4693 | 3070 | 1.53x | 0.690 | 4 | 142501 | 12760 | 11.17x |
| 6 | 7811 | 5123 | 1.52x | 0.686 | 4 | 199961 | 22137 | 9.03x |
| 7 | 11432 | 7514 | 1.52x | 0.684 | 4 | 238962 | 32339 | 7.39x |
| 8 | 15507 | 10219 | 1.52x | 0.681 | 4 | 257838 | 39600 | 6.51x |

Interpretation:

- A safe product-row quotient cache would reduce actual row construction by
  only about `1.5x` in this finite-shift Bach setup. Since actual row
  construction is already cached and is not the dominant cost, implementing a
  second quotient row cache is unlikely to produce a large runtime win here.
- Time-masked row signatures collapse much more strongly, but this does not by
  itself justify caching beta sums. The backward value still depends on the
  actual successor beta values at the next time. Absolute anchors break the
  simple equivariance needed to reuse those sums safely.
- Therefore the safe conclusion is to keep the actual-row cache and diagnostics,
  but not to implement the deeper quotient recurrence yet for this finite
  absolute-transposition experiment.

Definitions:

- `state orbits` canonicalize `(context state, DFA prefix state)` modulo a
  common integer pitch shift.
- `time edge orbits` canonicalize accepted product-edge relaxations but keep
  time distinct. This is the conservative target for exact BP-time reduction
  under absolute positional constraints.
- `structural edge orbits` canonicalize accepted product-edge relaxations
  without time. This measures reusable transition structure, but it is not by
  itself a BP-time speedup because backward messages are time dependent.
- `DFA prefix orbits` canonicalize the forbidden-substring automaton prefix
  states modulo transposition.

## Interpretation

The structural symmetry is large. MAXORDER DFA prefix states collapse by about
`15x`, and unique product states collapse by about `15x` in this Bach 12-key
setting. This confirms that the product is carrying substantial transposed
duplicate structure.

The immediately relevant BP edge-work quotient is smaller: time-indexed product
edges collapse by about `1.35x` to `1.45x` once the absolute ascending-anchor
masks are respected by time. This is still meaningful, but it is not a simple
12x runtime opportunity.

The large gap between `time edge reduction` and `structural edge reduction`
suggests the deeper implementation is not only a smaller graph. It should use
transformation-aware BP rows: shared symbolic transition families, with
time/offset-specific backward values and positional masks applied exactly.

The row-cache implementation is the safe first layer of that idea. It avoids
rebuilding actual product rows. A fuller transformation-aware recurrence would
also reuse shifted row families across product-state orbits when the declared
augmentation/constraint pair guarantees exact equivariance, otherwise falling
back to this actual-row cache.

The exact row-signature diagnostic suggests that such a recurrence should be
developed only under a stronger contract, for example a genuinely closed
transposition action or constraints whose future masks are transformation-aware
enough to make shifted beta sums provably identical. The current Bach setup
uses a finite absolute shift set `-6..+5` plus absolute anchors, so the exact
quotient opportunity is real but limited.

## Exactness Caveat

These counts are diagnostics over the current exact BP product. They do not
modify semantics and do not approximate the target distribution.

A future compressed recurrence must keep enough information to evaluate
absolute positional masks, e.g. anchor pitch `64`, while quotienting only the
shift-equivalent parts of the context graph and MAXORDER automaton. For
constraints that are not equivariant under the declared transform family, the
library should fall back to the generic exact path.
