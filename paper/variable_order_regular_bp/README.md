# Variable-Order Regular BP Paper Experiments

This folder contains the experiment implementations used for
[*Exact Regular-Constrained Variable-Order Markov Generation via Sparse
Context-State Belief Propagation*](https://arxiv.org/abs/2605.07839), accepted to
the **NeurIPS 2026 main track**. See the [paper citation](../../README.md#paper)
for BibTeX.

The experiments are kept outside the library package so
the reusable API can evolve without mixing in paper-specific benchmarking code.

The top-level `scripts/` files remain as compatibility launchers, so existing
commands still work:

```bash
python scripts/eval_tiny_exactness.py
python scripts/eval_scalability.py --help
python scripts/eval_bach_scalability.py --help
python scripts/eval_neurips_ablation.py --help
python scripts/eval_virtual_augmentation.py --help
python scripts/eval_bach_continuator_compare.py --help
python scripts/eval_bach_contextbp_final_compare.py --help
python scripts/eval_bach_positional_direct.py --help
```

The Bach pitch corpus remains at `data/bach_prelude_c_major_pitches.txt`.
Generated CSVs and reports should still be written under `outputs/` and
`reports/` unless a command line option says otherwise.
