"""Stable public backend entry points.

This module is the library-facing layer.  It compiles generic constraint
specifications and delegates to the current BP engines.  Future optimized
engines should be wired behind these functions without changing the public
surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .constraints import ConstraintSet, compile_constraints
from .context import Symbol
from .order_stack_bp import (
    OrderPolicy,
    OrderStackModel,
    RegularOrderStackBPResult,
    OrderStackBPResult,
    run_order_stack_bp,
    run_order_stack_masked_dfa_bp,
)


def run_constrained_order_stack(
    model: OrderStackModel,
    constraints: ConstraintSet | None = None,
    *,
    length: int,
    prefix: Sequence[Symbol],
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
) -> OrderStackBPResult | RegularOrderStackBPResult:
    """Run constrained order-stack BP through the public constraint API.

    Purely positional constraints use the positional order-stack backend.  Any
    regular component uses the regular backend with positional constraints kept
    as masks instead of being folded into the DFA state.
    """

    compiled = compile_constraints(
        constraints,
        length=length,
        alphabet=tuple(model.alphabet if alphabet is None else alphabet),
        prefer_dense_forbidden=prefer_dense_forbidden,
    )
    if compiled.regular_acceptor is None:
        return run_order_stack_bp(
            model,
            length=length,
            prefix=prefix,
            constraints=compiled.positional,
            policy=policy,
        )
    return run_order_stack_masked_dfa_bp(
        model,
        compiled.regular_acceptor,
        length=length,
        prefix=prefix,
        constraints=compiled.positional,
        policy=policy,
    )
