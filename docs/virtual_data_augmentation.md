# Virtual Data Augmentation

Virtual data augmentation represents transformed training corpora without
materializing every transformed sequence.

For a finite transformation family `G`, such as the 12 integer pitch
transpositions, the virtual model keeps absolute symbols and defines
continuation counts as:

```text
C_aug(c -> y) = sum_g C_base(g^-1(c) -> g^-1(y))
```

This is not interval modeling. The context `c` and emitted symbol `y` remain
absolute symbols. If several transformed copies explain the same absolute
context, their continuation counts are accumulated, exactly as in explicit
data augmentation.

## Why This Matters

Explicit data augmentation is natural for Markov/context models but can be
costly. For the Bach Prelude pitch sequence, physically adding 12 transpositions
changes the training material from 592 events to 7104 events. The absolute
contexts also grow because common pitches allow branches from different
transposed copies to meet.

The virtual model stores the original counts and transformation family, then
answers continuation queries lazily. For regular BP, it also exposes lazy
fixed-order graphs so graph states are materialized only when the backward pass
or sampler asks for them.

## Exactness

Virtual augmentation is intended to match explicit transformed-corpus
augmentation. The tests compare:

- augmented continuation counts;
- full fixed-order graph edges;
- positional order-stack distributions;
- regular/MAXORDER order-stack distributions.

The target distribution is unchanged relative to explicit augmentation.

## Bach 12-Transposition Measurement

Command:

```bash
python scripts/eval_virtual_augmentation.py
```

First measurement on the Bach MAXORDER + final-C setup:

| method | stored events | full graph states | BP graph states | product edges |
|---|---:|---:|---:|---:|
| original corpus | 592 | 3587 | 3587 | 82491 |
| explicit 12-key augmentation | 7104 | 27371 | 27371 | 1968532 |
| virtual 12-key augmentation | 592 | 27371 | 896 | 1968532 |

The virtual backend matched explicit augmentation success mass and start-order
masses in this setup. The current implementation reduces stored training data
and graph materialization, but product-edge work is still the same. Further BP
speedups would require combining virtual augmentation with a compact
transformation-aware MAXORDER/product representation.
