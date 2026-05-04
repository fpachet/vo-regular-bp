"""First-hit order-stack continuation examples."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    ConstraintSet,
    LongestFeasiblePolicy,
    OrderStackModel,
    prepare_until_end_order_stack,
    prepare_until_order_stack,
)


training = ("A", "B", "C", "A", "D", "C")
model = OrderStackModel.from_sequences([training], max_order=1)

until_c = prepare_until_order_stack(
    model,
    prefix=("A",),
    stop="C",
    min_length=1,
    max_length=3,
    constraints=ConstraintSet(positional={0: {"D"}}),
    policy=LongestFeasiblePolicy(),
)

sample = until_c.sample_with_orders(rng=0)
assert sample.sequence == ("D", "C")
assert sample.sequence[-1] == "C"
assert "C" not in sample.sequence[:-1]


end = "<END>"
end_model = OrderStackModel.from_sequences(
    [("A", "B", "C")],
    max_order=1,
    end_symbol=end,
)

until_end = prepare_until_end_order_stack(
    end_model,
    prefix=("A",),
    end_symbol=end,
    min_length=3,
    max_length=3,
    policy=LongestFeasiblePolicy(),
)

end_sample = until_end.sample(rng=0)
assert end_sample == ("B", "C", end)

print(sample.sequence)
print(sample.orders)
print(until_c.feasible_lengths)
print(until_c.diagnostics.as_dict())
print(end_sample)
