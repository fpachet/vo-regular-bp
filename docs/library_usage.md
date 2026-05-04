# Library Usage Guide

`vo_regular_bp` is meant to be imported by other projects through the top-level
package:

```python
from vo_regular_bp import (
    ConstraintSet,
    OrderStackModel,
    prepare_constrained_order_stack,
    prepare_until_order_stack,
)
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

Use `prepare_until_order_stack(...)` when you want a variable-length generated
suffix that stops at the first generated symbol matching a stop constraint.
The prefix is conditioning context only and is not included in the returned
suffix. `stop` can be one symbol, an iterable/set of symbols, or a predicate
`stop(symbol) -> bool`. `prepare_until_end_order_stack(..., end_symbol=...)`
is the convenience form for learned END/sentinel stops.

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

`prepare_until_order_stack(...)` composes the caller's `ConstraintSet` with
first-hit positional masks for each candidate length: positions before the
final one must not satisfy `stop`, and the final position must satisfy `stop`.
Conflicting caller constraints simply make that length infeasible. The prepared
backend exposes `sample(...)`, `sample_with_orders(...)`,
`sample_with_trace(...)`, `sample_many(...)`, `diagnostics`, and
`feasible_lengths`. Feasible lengths are weighted by the sum of positive
start-order masses reported by their fixed-length backend; if a backend can
only report success, that feasible length receives unit weight.

Learned START/END sentinels remain forbidden in ordinary unconstrained
fixed-length generation. First-hit generation allows a forbidden stop symbol
only at the explicit final stop position, so
`prepare_until_end_order_stack(...)` can emit END without changing the default
fixed-length behavior.

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
- `examples/until_order_stack_backend.py`: variable-length first-hit
  continuation, including learned END/sentinel generation.
- `examples/event_order_stack_backend.py`: rich event objects encoded through
  `EventCodec`, with pitch-class and meter constraints.
- `examples/continuator_style_backend.py`: Continuator-shaped facade with final
  pitch class, total duration, and meter-cycle constraints.

Paper experiments are kept separately under `paper/variable_order_regular_bp/`.
Compatibility wrappers remain under `scripts/` for the existing evaluation
commands.
