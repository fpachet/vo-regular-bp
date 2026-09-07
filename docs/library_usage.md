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

## Cache Lifetime and Numerical Range

Treat a prepared plan's training model, transforms, positional predicates, and
acceptors as immutable. For changing soft weights, use the support-plan API to
obtain fresh regular messages. A regular plan shares its DFA-symbol transition
cache across source orders. Each order retains at most 65,536 transition-cache
entries (an empty row costs one entry); additional rows are evaluated without
being retained. Beta messages and sampled candidate sets remain reusable.
Diagnostics expose `regular_transition_cache_entries` and
`regular_transition_cache_skipped_rows`; `regular_transition_rows` counts
currently retained rows, not all rows ever evaluated.

Virtual models now share lazy source graphs across prefixes and horizons, with
stable context IDs. `virtual_model.clear_caches()` releases model-owned compiled
views. Existing plans keep their graph references and remain usable; discard
those plans too when their memory is no longer needed.

`sample()` avoids order-output allocation. For the exact built-in
`LongestFeasiblePolicy`, non-trace sampling also avoids decision metadata and
stops evaluating orders at the first feasible one. Custom policies and traces
retain the full candidate interface. Consequently, requesting a trace for the
first time can perform work that sequence-only sampling did not need.

Product and positional BP results expose `log_partition_function` and
`log_conditional_probability(sequence)`. Ordinary float masses may round to
zero or infinity outside their representable range; use logarithmic values to
distinguish underflow from an impossible language (`-inf`). Sampling uses stable
relative weights when message arithmetic approaches those limits. Order-stack
results expose `start_log_order_masses()`, also available in backend diagnostics.
Long regular horizons use an iterative log-space evaluator without changing the
process-wide recursion limit. Short ordinary queries retain the fast recurrence.
Individual transition weights must still be finite; values already underflowed
inside a caller's weight function cannot be recovered by BP.

For first-hit generation with `constraints=None`, candidate lengths share suffix
views of one backward table. Additional constraints use the generic per-length
backend. The same length-selection policy is preserved. If absolute length
masses are outside float range, `length_weights` contains proportional rescaled
weights instead; feasible lengths remain available.

Full lifecycle timings, memory measurements, and limitations are recorded in
[`reports/implementation_results_2026_09_07.md`](../reports/implementation_results_2026_09_07.md).

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
start-order masses reported by their fixed-length backend. Logarithmic masses
preserve feasible lengths when ordinary float masses underflow; `length_weights`
then contains proportional rescaled weights. With `constraints=None`, the
per-length backends share one backward table through suffix views. Additional
constraints, including an explicit empty `ConstraintSet`, use the general
per-length path.

Learned START/END sentinels remain forbidden in ordinary unconstrained
fixed-length generation. First-hit generation allows a forbidden stop symbol
only at the explicit final stop position, so
`prepare_until_end_order_stack(...)` can emit END without changing the default
fixed-length behavior.

## Source Compression

Exact source minimization is optional and semantics-preserving. Use it when you
want a read-only quotient of the source graph:

```python
from vo_regular_bp import exact_context_graph_quotient_stats, minimize_context_graph

stats = exact_context_graph_quotient_stats(graph)
minimized = minimize_context_graph(graph)
```

For order-stack backends, pass `minimize_source_graphs=True` to preparation
functions. This minimizes the fixed-order source graphs before BP while keeping
the training/count model unchanged. BP remains exact with respect to the same
source model.

Approximate source-state merging is experimental and explicit:

```python
from vo_regular_bp.experimental import alergia_merge, alergia_metadata

merged = alergia_merge(
    graph,
    alpha=0.01,
    min_support=10,
    recursive=True,
    transition_projection=lambda state, symbol, edge: (
        symbol - state[-1]
        if state and isinstance(state[-1], int) and isinstance(symbol, int)
        else symbol
    ),
)
metadata = alergia_metadata(merged)
```

This ALERGIA-like merge compares continuation distributions with a Hoeffding
compatibility test and recursively checks successor states when requested. It
changes the source model, but the returned object is a normal `ContextGraph`, so
regular BP remains exact for the merged model. Approximate merging is not a
constrained-product quotient and is never enabled by default.

By default, compatibility compares raw symbols. Pass `symbol_projection` when a
client wants to compare emitted symbols through a feature map. Pass
`transition_projection(state, symbol, edge)` when similarity depends on the
source context and transition, such as relative motion. If both are supplied,
`transition_projection` takes precedence. The merged graph still emits concrete
symbols; projections only control the compatibility test and recursive
successor matching.

Within one `alergia_merge(...)` call, projected continuation counts, dominant
projected successor labels, and recursive pair-compatibility decisions are
cached. Active merge classes and their members are tracked incrementally instead
of being reconstructed by scanning all source states during every candidate
comparison. This is most useful when ALERGIA considers many candidate classes,
whether because a projection makes states compatible enough to require
recursive checks or because many pairs are rejected cheaply. It preserves the
same merge criterion and metadata as the uncached implementation.

For quick compression diagnostics, use:

```bash
python scripts/diagnose_source_merging.py data/bach_prelude_c_major_pitches.txt \
  --max-order 6 \
  --projection interval \
  --alpha 0.01 \
  --min-support 10
```

The script reports raw source size, exact-minimized size, ALERGIA-merged size,
edge counts, compression ratios, projection kind, and optional held-out average
log probability.

For Continuator-style order stacks, use the experimental wrapper before calling
the normal backend:

```python
from vo_regular_bp import ConstraintSet, prepare_constrained_order_stack
from vo_regular_bp.experimental import (
    alergia_merge_order_stack_model,
    alergia_metadata,
)

abstract_model = alergia_merge_order_stack_model(
    model,
    alpha=0.01,
    min_support=10,
    recursive=True,
    transition_projection=lambda state, symbol, edge: (
        symbol - state[-1]
        if state and isinstance(state[-1], int) and isinstance(symbol, int)
        else symbol
    ),
)

backend = prepare_constrained_order_stack(
    abstract_model,
    ConstraintSet(positional={7: final_symbols}),
    length=8,
    prefix=prefix,
)
metadata = alergia_metadata(abstract_model)
```

`alergia_merge_order_stack_model(...)` materializes and merges each fixed-order
source graph independently. It returns a model-like wrapper that exposes
`compile_graph(order)`, so existing positional and regular order-stack
preparation functions can use it without a special sampling path. Prefix
aliases from original contexts are preserved, which means ordinary prefixes
still resolve to the merged representative state. Trace objects and sampled
order labels keep their existing shape, but they describe the merged source
model rather than the literal training model.

For lower-level experiments, `alergia_merge_fixed_order_graph(...)` applies the
same operation to one materialized fixed-order graph. Supplying
`continuation_counts` is recommended when `min_support` should reflect training
support rather than normalized probabilities; `alergia_merge_order_stack_model`
does this automatically for ordinary `OrderStackModel` instances.

The per-order metadata reports raw and merged state/edge counts, class members,
destination conflicts, edge-order conflicts, projection kind, and merge time.
This is intended for abstraction experiments; it is not exact source
minimization and is not enabled by default.

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
