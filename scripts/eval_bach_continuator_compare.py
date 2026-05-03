#!/usr/bin/env python
"""Compatibility launcher for the paper Bach Continuator comparison."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.variable_order_regular_bp.scripts.eval_bach_continuator_compare import *  # noqa: F401,F403
from paper.variable_order_regular_bp.scripts.eval_bach_continuator_compare import main


if __name__ == "__main__":
    main()
