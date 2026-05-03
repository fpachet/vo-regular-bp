#!/usr/bin/env python
"""Compatibility launcher for the NeurIPS Bach ablation benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.variable_order_regular_bp.scripts.eval_neurips_ablation import *  # noqa: F401,F403
from paper.variable_order_regular_bp.scripts.eval_neurips_ablation import main


if __name__ == "__main__":
    main()
