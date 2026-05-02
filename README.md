# vo-regular-bp

Small Python library for exact constrained sampling from sparse variable-order
Markov/context models under regular constraints.

The core algorithm runs backward dynamic programming on the reachable product of
a context graph and a deterministic acceptor, then samples each next symbol
proportionally to its model probability times the downstream beta value.

```python
from vo_regular_bp import ContextGraph, positional_acceptor, run_bp

graph = ContextGraph.from_counts({(): {"a": 10, "b": 1}}, max_order=0)
acceptor = positional_acceptor(length=1)
bp = run_bp(graph, acceptor, length=1)

print(bp.partition_function)
print(bp.sample(rng=0))
```
