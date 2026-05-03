# Library Usage Guide

`vo_regular_bp` is meant to be imported by other projects through the top-level
package:

```python
from vo_regular_bp import ConstraintSet, OrderStackModel, prepare_constrained_order_stack
```

The public surface is project-agnostic. Symbols may be pitches, tokens, tuples,
or any other hashable values. Projects with richer event objects can use
`EventCodec` or the Continuator-shaped facade to encode events at the boundary.

## Backends

Use `prepare_constrained_order_stack(...)` when you already have an
`OrderStackModel`. It compiles constraints, runs the backward pass once, and
returns a reusable `ConstrainedOrderStackBackend`.

Use `prepare_constrained_order_stack_from_sequences(...)` when your training
material is already a sequence of hashable symbols.

Use `prepare_constrained_order_stack_from_events(...)` when your project keeps
rich event objects and needs explicit encode/decode functions.

Use `prepare_continuation_backend(...)` when integrating with a Continuator-like
project vocabulary. This facade is still dependency-free and does not import a
Continuator package.

For the lower-level exact product-BP engine, use `ContextGraph`, DFA helpers,
and `run_bp(...)`.

## Constraint Semantics

All high-level constraints are fixed-horizon constraints over the generated
suffix only. Positions are zero-based in the generated sequence, not in the
training corpus and not in `prefix + generated`.

`positional` constraints are per-time symbol masks or predicates. They do not
inflate the regular DFA state; the backend applies them directly at the relevant
time step.

`forbidden_substrings` are regular constraints over generated substrings. The
compiler uses a dense finite-alphabet forbidden-substring DFA by default. This
is the library form of the paper's MAXORDER copied n-gram constraint.

`regular_acceptors` are caller-supplied deterministic acceptors. A transition
returning `None` rejects the emitted symbol from the current acceptor state.
Multiple regular constraints are intersected exactly.

`meter` is a finite per-position class pattern. It is useful for constraints
such as strong/weak beats, stress classes, or symbolic duration classes.

`cumulative_meter` tracks an integer cumulative cost such as duration. It can
enforce a final accepted total, a maximum total, and optional predicates at each
step.

## Exactness Notes

`run_bp(...)` samples exactly from one probabilistic context graph conditioned
on the supplied regular language.

The order-stack backend implements a Continuator-style generation policy over
separate fixed-order graphs. It computes exact feasible future mass for each
order and samples exactly from the distribution induced by the chosen
`OrderPolicy`. It should not be described as conditioning one single stochastic
source unless the caller has explicitly made that modeling choice.

No high-level backend uses beam search, heuristic pruning, or approximate
filtering.

## Examples

- `examples/symbolic_order_stack_backend.py`: plain symbolic sequence backend
  with final pitch-class and MAXORDER constraints.
- `examples/event_order_stack_backend.py`: rich event objects encoded through
  `EventCodec`, with pitch-class and meter constraints.
- `examples/continuator_style_backend.py`: Continuator-shaped facade with final
  pitch class, total duration, and meter-cycle constraints.

Paper experiments are kept separately under `paper/variable_order_regular_bp/`.
Compatibility wrappers remain under `scripts/` for the existing evaluation
commands.
