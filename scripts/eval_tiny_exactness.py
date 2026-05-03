#!/usr/bin/env python
"""Compatibility launcher for the paper tiny exactness experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.variable_order_regular_bp.scripts.eval_tiny_exactness import *  # noqa: F401,F403
from paper.variable_order_regular_bp.scripts.eval_tiny_exactness import main


if __name__ == "__main__":
    main()
