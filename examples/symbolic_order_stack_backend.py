"""Symbolic order-stack backend with final pitch class and MAXORDER constraints."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    LongestFeasiblePolicy,
    OrderStackModel,
    avoid_copied_ngrams,
    combine_constraints,
    final_pitch_class,
    prepare_constrained_order_stack,
)


training = (
    60,
    62,
    64,
    60,
    65,
    67,
    60,
    69,
    72,
    60,
    62,
    67,
    72,
    60,
)
horizon = 4
prefix = training[:3]
forbidden_length = 4

model = OrderStackModel.from_sequences([training], max_order=3)
constraints = combine_constraints(
    final_pitch_class(0, length=horizon),
    avoid_copied_ngrams(training, forbidden_length),
)

backend = prepare_constrained_order_stack(
    model,
    constraints,
    length=horizon,
    prefix=prefix,
    policy=LongestFeasiblePolicy(),
)

generated = backend.sample_with_orders(rng=7)
forbidden = {
    tuple(training[index : index + forbidden_length])
    for index in range(len(training) - forbidden_length + 1)
}

assert generated.sequence[-1] % 12 == 0
assert all(
    tuple(generated.sequence[index : index + forbidden_length]) not in forbidden
    for index in range(len(generated.sequence) - forbidden_length + 1)
)

print(generated.sequence)
print(generated.orders)
print(backend.diagnostics.as_dict())
