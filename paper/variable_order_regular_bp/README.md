# Variable-Order Regular BP Paper Experiments

This folder contains the experiment implementations used for the
variable-order regular BP paper.  They are kept outside the library package so
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
