# Library Usage Guide

`vo_regular_bp` is meant to be imported by other projects through the top-level
package:

```python
from vo_regular_bp import (
    ConstraintSet,
    OrderStackModel,
    prepare_constrained_order_stack,
    prepare_constrained_order_stack_plan,
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

Use `prepare_constrained_order_stack_plan(...)` when the same
`model + length + constraints` will be reused with several prefixes. The plan is
prefix-independent; call `plan.for_prefix(prefix)` to get the ordinary
`ConstrainedOrderStackBackend` for that prefix. Existing prefix-taking APIs are
kept as convenience wrappers, so callers can adopt plans only where repeated
prefixes make caching worthwhile.

Use `prepare_constrained_order_stack_from_sequences(...)` when your training
material is already a sequence of hashable symbols.

Use `prepare_constrained_order_stack_plan_from_sequences(...)` for the same
sequence-building convenience without binding a prefix yet.

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

Use `prepare_continuation_plan(...)` for Continuator-like integrations that make
many repeated calls with different prefixes but the same training material,
horizon, and constraints.

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
Acceptors may also implement `transition_weight(state, symbol) -> float` to add
soft multiplicative transition weights. The default weight is `1.0`, and a
weight of `0.0` rejects the transition. Multiple regular constraints are
intersected exactly, multiplying their transition weights.

For example, an adaptive repetition penalty can be represented by a DFA whose
state tracks recent symbols and whose `transition_weight(...)` returns
`exp(-lambda * repetition_cost)`. For a fixed BP run these weights are treated
as ordinary edge potentials, so beta messages and sampling use the same
normalized distribution.

`meter` is a finite per-position class pattern. It is useful for constraints
such as strong/weak beats, stress classes, or symbolic duration classes.

`cumulative_meter` tracks an integer cumulative cost such as duration. It can
enforce a final accepted total, a maximum total, and optional predicates at each
step.

For note symbols such as `(pitch, duration)`, an exact generated duration of
32 over a fixed eight-note horizon can be expressed as:

```python
from vo_regular_bp import duration_total_constraint, prepare_continuation_backend

backend = prepare_continuation_backend(
    [training_notes],
    prefix=prefix_notes,
    horizon=8,
    max_order=3,
    constraints=duration_total_constraint(
        32,
        horizon=8,
        symbol_to_duration=lambda symbol: symbol[1],
    ),
    event_to_symbol=lambda note: (note.pitch, note.duration),
)

generated = backend.sample_events_with_orders(rng=0)
assert sum(note.duration for note in generated.events) == 32
```

This is a regular/cumulative constraint, so the backend uses the regular
order-stack path. It remains exact: no beam search or rejection sampling is
involved. See `examples/duration_total_order_stack_backend.py` for a complete
small example.

For variable musical length inside a fixed BP horizon, use explicit PAD
symbols. PAD has zero duration, may appear only after the target duration has
been reached, and is absorbing:

```python
from vo_regular_bp import (
    append_padding,
    padded_duration_total_constraint,
    prepare_continuation_backend,
)

PAD = Note(-1, 0)

def encode(note):
    return "<PAD>" if note == PAD else (note.pitch, note.duration)

backend = prepare_continuation_backend(
    append_padding([training_phrase], pad_symbol=PAD, pad_count=max_order + 1),
    prefix=prefix_notes,
    horizon=16,
    max_order=max_order,
    constraints=padded_duration_total_constraint(
        32,
        horizon=16,
        pad_symbol="<PAD>",
        symbol_to_duration=lambda symbol: symbol[1],
    ),
    event_to_symbol=encode,
)
```

Appending at least `max_order + 1` PAD events gives the order-stack model PAD
self-loops at every fixed order. The cumulative constraint then guarantees that
all generated real notes sum to the requested total and every following symbol
is PAD. See `examples/padded_duration_order_stack_backend.py`.

This is exact for the padded fixed-horizon model. Its probabilities include the
model probability of entering PAD from the last real context. That is usually
appropriate when PAD marks learned phrase/bar endings. If a project wants
length-neutral probabilities after marginalizing over possible stop lengths, it
should use a dedicated first-hit cumulative-duration backend rather than
treating PAD as an ordinary learned symbol.

## Prefix-Independent Plans

Prefix-independent plans separate reusable constrained preparation from the
prefix-dependent sampling start:

```python
plan = prepare_constrained_order_stack_plan(
    model,
    constraints,
    length=8,
)

backend = plan.for_prefix(prefix)
generated = backend.sample_with_orders(rng=0)
```

For positional-only constraints, the fixed-order backward tables are prepared for
all graph states up front. For regular constraints, the plan owns shared lazy beta
caches keyed by time, context state, and DFA state. Calling `for_prefix(...)`
warms the messages reachable from that prefix; later prefixes reuse any
overlapping entries. This avoids rebuilding graph/constraint/backward objects
when an external project repeatedly asks for continuations under the same model,
horizon, and constraints.

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
- `examples/duration_total_order_stack_backend.py`: note events encoded as
  `(pitch, duration)` symbols with exact total generated duration.
- `examples/padded_duration_order_stack_backend.py`: variable musical length
  in a fixed horizon using zero-duration absorbing PAD symbols.

Paper experiments are kept separately under `paper/variable_order_regular_bp/`.
Compatibility wrappers remain under `scripts/` for the existing evaluation
commands.
