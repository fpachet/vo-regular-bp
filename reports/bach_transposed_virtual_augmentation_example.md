# Bach 12-Key Virtual Augmentation Example

This report summarizes a Bach Prelude pitch-only generation example using
virtual transposition augmentation. It is intended as a handoff note for the
paper project.

Generated on: `2026-05-03`

## Setup

- Branch: `virtual-data-augmentation`
- Commit: `23e9fd6986499bda625d59030bfdfa8290d2ece0`
- Corpus: `data/bach_prelude_c_major_pitches.txt`
- Base training tokens: 592
- Virtual augmentation: integer pitch shifts `-6 -5 -4 -3 -2 -1 0 1 2 3 4 5`
- Virtual training tokens: 7104
- Virtual alphabet size: 52
- Model: `VirtualAugmentedOrderStackModel`
- Max context order: `K = 4`
- Horizon: `n = 64`
- Policy: `SingletonAvoidingBackoffPolicy`
- Prefix: fixed start sentinel
- Bar-start anchor period: 8 generated tokens
- Anchor positions: `0 8 16 24 32 40 48 56`
- Required anchor pitches: `60 62 64 65 67 69 71 72`
- MAXORDER / anti-copy constraints: exact forbidden copied n-grams from the
  full 12-key virtually augmented corpus.

The MIDI export holds the final anchor at position 56 as a full-measure note.
Positions 57-63 remain in the trace/sequence diagnostics but are not audible in
the MIDI rendering.

The generation was run programmatically from the current branch using the
current `vo_regular_bp` virtual augmentation backend and the helper routines
from `examples/bach_c_anchored_continuation.py`. Raw outputs are under:

```text
outputs/bach_c_anchored_transposed_virtual/
```

## Exactness And Constraint Checks

All generated cases below satisfy:

- all required bar-start anchors;
- no copied n-gram violation against the full 12-key augmented corpus;
- successful constrained backend preparation (`success_mass = 1.0`).

Virtual augmentation preserves explicit augmentation semantics: continuation
counts are computed as if all 12 transposed corpora were physically present,
while the stored model keeps only the original sequence and the transform
family.

## Summary

| copy n-gram | anchors match | longest copied span | violates copy n-gram | context states | context edges | product states | product edges | max order used |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 8 | true | 7 | false | 2667 | 5573 | 31790 | 2824222 | 4 |
| 7 | true | 6 | false | 4226 | 7404 | 22531 | 2610565 | 3 |
| 6 | true | 5 | false | 4765 | 7886 | 14394 | 2245042 | 3 |
| 5 | true | 4 | false | 4803 | 7914 | 8130 | 1587777 | 2 |

## Generated Artifacts

| copy n-gram | MIDI | order trace |
|---:|---|---|
| 8 | `outputs/bach_c_anchored_transposed_virtual/midi/bach_c_anchor_transposed_h64_K4_singleton_copy8.mid` | `outputs/bach_c_anchored_transposed_virtual/traces/bach_c_anchor_transposed_h64_K4_singleton_copy8_orders.csv` |
| 7 | `outputs/bach_c_anchored_transposed_virtual/midi/bach_c_anchor_transposed_h64_K4_singleton_copy7.mid` | `outputs/bach_c_anchored_transposed_virtual/traces/bach_c_anchor_transposed_h64_K4_singleton_copy7_orders.csv` |
| 6 | `outputs/bach_c_anchored_transposed_virtual/midi/bach_c_anchor_transposed_h64_K4_singleton_copy6.mid` | `outputs/bach_c_anchored_transposed_virtual/traces/bach_c_anchor_transposed_h64_K4_singleton_copy6_orders.csv` |
| 5 | `outputs/bach_c_anchored_transposed_virtual/midi/bach_c_anchor_transposed_h64_K4_singleton_copy5.mid` | `outputs/bach_c_anchored_transposed_virtual/traces/bach_c_anchor_transposed_h64_K4_singleton_copy5_orders.csv` |

The raw summary table is:

```text
outputs/bach_c_anchored_transposed_virtual/summary.csv
```

## Case Details

### MAXORDER / Copy N-Gram 8

Sequence:

```text
60 64 69 73 57 62 66 69 62 68 71 74 67 72 76 60 64 67 60 64 69 72 58 61 65 56 61 68 51 56 60 63 67 70 75 78 82 75 60 65 69 55 60 64 67 59 62 68 71 63 68 72 56 60 63 69 72 57 61 52 57 64 40 52
```

Orders:

```text
1 2 2 2 3 3 3 2 2 1 2 2 2 1 2 3 1 3 3 1 2 3 3 2 1 2 3 3 2 2 2 2 1 2 2 2 2 2 1 1 1 3 1 2 2 2 2 2 1 2 2 2 3 3 3 2 3 3 1 2 1 2 3 4
```

Order histogram: `1:14, 2:32, 3:17, 4:1`

Anchor pitches at positions `0,8,...,56`:

```text
60 62 64 65 67 69 71 72
```

Longest copied span in augmented corpus: length 7

```text
64 69 73 57 62 66 69
```

### MAXORDER / Copy N-Gram 7

Sequence:

```text
60 64 67 72 76 62 66 57 62 66 69 75 66 70 76 79 64 69 73 64 67 72 76 60 65 68 73 77 68 74 77 63 67 72 76 60 64 69 73 64 69 72 62 67 50 55 58 67 71 62 67 71 76 79 65 69 72 59 64 67 59 62 48 51
```

Orders:

```text
1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 1 2 2 2 2 2 3 1 2 2 1 1 1 2 2 2 2 3 2 3 1 3 1 2 3 2 2 1
```

Order histogram: `1:12, 2:46, 3:6`

Anchor pitches at positions `0,8,...,56`:

```text
60 62 64 65 67 69 71 72
```

Longest copied span in augmented corpus: length 6

```text
77 68 74 77 63 67
```

### MAXORDER / Copy N-Gram 6

Sequence:

```text
60 65 72 60 65 69 74 77 62 55 59 62 49 52 57 61 64 68 54 58 61 67 71 74 65 69 74 77 64 67 73 63 67 70 76 82 73 76 81 85 69 73 64 69 72 59 62 67 71 77 61 65 70 74 65 68 72 75 67 70 75 79 63 66
```

Orders:

```text
1 1 2 2 2 2 2 2 1 1 1 2 2 1 2 2 1 2 2 2 2 2 1 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 1 2 2 1 2 2 2 2 1 2 2 1 3 3 3 1
```

Order histogram: `1:15, 2:46, 3:3`

Anchor pitches at positions `0,8,...,56`:

```text
60 62 64 65 67 69 71 72
```

Longest copied span in augmented corpus: length 5

```text
60 65 72 60 65
```

### MAXORDER / Copy N-Gram 5

Sequence:

```text
60 65 69 72 76 67 72 58 62 53 57 62 66 72 75 80 64 69 72 64 68 52 56 60 65 57 62 65 70 73 76 62 67 59 64 67 72 75 61 65 69 72 77 63 67 58 62 65 71 74 78 69 74 77 64 67 72 64 69 73 76 68 71 64
```

Orders:

```text
1 1 2 2 2 2 2 2 1 2 1 2 2 2 2 2 2 1 2 2 1 2 2 1 2 2 2 2 2 2 2 1 1 2 2 2 2 2 2 1 1 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2
```

Order histogram: `1:13, 2:51`

Anchor pitches at positions `0,8,...,56`:

```text
60 62 64 65 67 69 71 72
```

Longest copied span in augmented corpus: length 4

```text
60 65 69 72
```

## Paper Interpretation

This example illustrates two points:

1. Explicit transposition augmentation can be used musically with exact
   positional and MAXORDER constraints, while preserving absolute-pitch branch
   sharing across keys.
2. Virtual augmentation gives the same augmented-corpus semantics without
   storing all transformed training sequences in the model.

The present implementation is an exact investigation backend. It reduces stored
training data and graph materialization, but the regular product expansion is
still large for the MAXORDER constraint. This makes it a useful paper example
and also motivates future transformation-aware product/automaton compression.

This report does not introduce a new exactness claim beyond the existing
virtual-vs-explicit augmentation equivalence tested in the repository. It is a
concrete musical example showing that the transposed augmented corpus semantics
can be used with the ascending-bar anchor constraint and MAXORDER anti-copy
constraints.
