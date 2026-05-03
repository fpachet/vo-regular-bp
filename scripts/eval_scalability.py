#!/usr/bin/env python
"""Compatibility launcher for the paper scalability benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.variable_order_regular_bp.scripts.eval_scalability import *  # noqa: F401,F403
from paper.variable_order_regular_bp.scripts.eval_scalability import main


if __name__ == "__main__":
    main()
