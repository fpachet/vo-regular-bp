#!/usr/bin/env python
"""Compatibility launcher for the paper Bach positional benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.variable_order_regular_bp.scripts.eval_bach_positional_direct import *  # noqa: F401,F403
from paper.variable_order_regular_bp.scripts.eval_bach_positional_direct import main


if __name__ == "__main__":
    main()
