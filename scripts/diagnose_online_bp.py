"""Measure beta dependency propagation after one phrase addition, without reuse.

python scripts/diagnose_online_bp.py
Compares complete tables by context label; does not time incremental BP or claim
that changed cells can be discovered or updated for free.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vo_regular_bp import ConstraintSet, OrderStackModel, prepare_constrained_order_stack_plan


def main():
    bach = tuple(int(t) for line in (ROOT / "data/bach_prelude_c_major_pitches.txt")
                 .read_text().splitlines() for t in line.split("#")[0].split())[:549]
    cases = {
        "bach-duplicate": ([bach], bach[:16], 4, 64, {60, 72}),
        "bach-new-symbol": ([bach], bach[:15] + (91,), 4, 64, {60, 72}),
        "disconnected": ([(0, 0, 1, 0), (0, 1, 1, 0), (7, 7, 8, 7)],
                         (0, 0, 0, 1), 2, 16, {0, 7}),
    }
    for name, (memory, phrase, order, length, final) in cases.items():
        before = OrderStackModel.from_sequences(memory, max_order=order)
        after = OrderStackModel.from_sequences(memory + [phrase], max_order=order)
        constraints = ConstraintSet(positional={length - 1: final})
        left, right = [prepare_constrained_order_stack_plan(m, constraints, length=length).plan
                       for m in (before, after)]
        orders = []
        for k in range(1, order + 1):
            a, b = left.graphs[k], right.graphs[k]
            common = set(a.contexts) & set(b.contexts)
            def row(g, c):
                return [(e.symbol, e.probability, e.order, g.contexts[e.dst])
                        for e in g.outgoing[g.state_id(c)]]
            changed_rows = sum(row(a, c) != row(b, c) for c in common)
            times = []
            for t in range(length + 1):
                changed = sum(
                    abs(left.backwards[k][t][a.state_id(c)] - right.backwards[k][t][b.state_id(c)])
                    > 1e-12 * max(abs(left.backwards[k][t][a.state_id(c)]),
                                  abs(right.backwards[k][t][b.state_id(c)]), 1e-300)
                    for c in common
                )
                times.append(changed)
            orders.append(dict(order=k, comparable_states=len(common), changed_rows=changed_rows,
                               added_states=len(set(b.contexts) - set(a.contexts)),
                               removed_states=len(set(a.contexts) - set(b.contexts)),
                               changed_beta_cells_by_time=times))
        print(json.dumps(dict(case=name, length=length, orders=orders), sort_keys=True))


if __name__ == "__main__":
    main()
