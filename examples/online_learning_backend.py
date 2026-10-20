"""Retain a symbolic model while learning phrases and preparing fresh inference."""
from vo_regular_bp import (
    ConstraintSet, LongestFeasiblePolicy, OrderStackModel,
    prepare_constrained_order_stack_plan, prepare_until_end_order_stack,
)


def main():
    phrase_memory = [(60, 64, 67, 60), (60, 65, 67, 60)]
    model = OrderStackModel.from_sequences(
        phrase_memory, max_order=4, start_symbol="START", end_symbol="END",
    )
    constraints = ConstraintSet(positional={8: {60}})
    plan = prepare_constrained_order_stack_plan(
        model, constraints, length=9, policy=LongestFeasiblePolicy(),
    )
    print("Before learning:", plan.for_prefix((60,)).sample(rng=42))

    new_phrase = (60, 69, 67, 60)
    phrase_memory.append(new_phrase)
    model = model.with_sequences([new_phrase])
    # Application plan/backend caches must now be cleared or keyed by the new
    # model's identity. Replacing this plan releases the old inference tables.
    plan = prepare_constrained_order_stack_plan(
        model, constraints, length=9, policy=LongestFeasiblePolicy(),
    )
    print("After learning:", plan.for_prefix((60,)).sample(rng=42))
    until = prepare_until_end_order_stack(
        model, prefix=(60,), end_symbol="END", max_length=16,
    )
    print("Through first END:", until.sample(rng=42))

    # Keep full phrase memory for persistence, removals, encoding/order changes,
    # and occasional independent consistency checks.
    fresh = OrderStackModel.from_sequences(
        phrase_memory, max_order=4, start_symbol="START", end_symbol="END",
    )
    assert model.counts == fresh.counts


if __name__ == "__main__":
    main()
