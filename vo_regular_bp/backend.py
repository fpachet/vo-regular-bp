"""Stable public backend entry points.

This module is the library-facing layer.  It compiles generic constraint
specifications and delegates to the current BP engines.  Future optimized
engines should be wired behind these functions without changing the public
surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import random

from .constraints import ConstraintSet, compile_constraints
from .context import Symbol
from .order_stack_bp import (
    OrderPolicy,
    OrderSampleStep,
    OrderStackModel,
    RegularOrderStackBPResult,
    OrderStackBPResult,
    run_order_stack_bp,
    run_order_stack_masked_dfa_bp,
)


@dataclass(frozen=True)
class GeneratedSequence:
    """A generated sequence plus optional order diagnostics."""

    sequence: tuple[Symbol, ...]
    orders: tuple[int, ...] = ()


@dataclass(frozen=True)
class BackendDiagnostics:
    """Runtime-independent diagnostics exposed by prepared backends."""

    backend: str
    length: int
    max_order: int
    context_states: int
    context_edges: int
    regular_product_states: int | None = None
    regular_product_states_time_indexed: int | None = None
    regular_product_edges: int | None = None
    success_mass: float | None = None
    start_order_masses: tuple[tuple[int, float], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "length": self.length,
            "max_order": self.max_order,
            "context_states": self.context_states,
            "context_edges": self.context_edges,
            "regular_product_states": self.regular_product_states,
            "regular_product_states_time_indexed": self.regular_product_states_time_indexed,
            "regular_product_edges": self.regular_product_edges,
            "success_mass": self.success_mass,
            "start_order_masses": self.start_order_masses,
        }


@dataclass(frozen=True)
class ConstrainedOrderStackBackend:
    """Prepared reusable constrained order-stack sampler.

    External projects can keep this object around after compilation/BP and call
    the sampling methods repeatedly.  The concrete result remains available for
    advanced inspection, but regular users should prefer this stable wrapper.
    """

    result: OrderStackBPResult | RegularOrderStackBPResult

    @property
    def diagnostics(self) -> BackendDiagnostics:
        is_regular = isinstance(self.result, RegularOrderStackBPResult)
        return BackendDiagnostics(
            backend="order_stack_regular" if is_regular else "order_stack_positional",
            length=self.result.length,
            max_order=self.result.model.max_order,
            context_states=self.result.context_state_count,
            context_edges=self.result.context_edge_count,
            regular_product_states=self.result.product_state_count if is_regular else None,
            regular_product_states_time_indexed=(
                self.result.time_indexed_product_state_count if is_regular else None
            ),
            regular_product_edges=self.result.product_edge_count if is_regular else None,
            success_mass=self.result.success_mass,
            start_order_masses=self.result.start_order_masses(),
        )

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        return self.result.sample(rng=rng)

    def sample_many(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[Symbol, ...]]:
        return [
            sequence
            for sequence, _orders in self.result.sample_many_with_orders(count, rng=rng)
        ]

    def sample_with_orders(self, *, rng: random.Random | int | None = None) -> GeneratedSequence:
        sequence, orders = self.result.sample_with_orders(rng=rng)
        return GeneratedSequence(sequence=sequence, orders=orders)

    def sample_many_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[GeneratedSequence]:
        return [
            GeneratedSequence(sequence=sequence, orders=orders)
            for sequence, orders in self.result.sample_many_with_orders(count, rng=rng)
        ]

    def sample_with_trace(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[OrderSampleStep, ...]]:
        return self.result.sample_with_trace(rng=rng)


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


def prepare_constrained_order_stack(
    model: OrderStackModel,
    constraints: ConstraintSet | None = None,
    *,
    length: int,
    prefix: Sequence[Symbol],
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
) -> ConstrainedOrderStackBackend:
    """Compile constraints and return a reusable order-stack backend."""

    return ConstrainedOrderStackBackend(
        run_constrained_order_stack(
            model,
            constraints,
            length=length,
            prefix=prefix,
            policy=policy,
            alphabet=alphabet,
            prefer_dense_forbidden=prefer_dense_forbidden,
        )
    )


def prepare_constrained_order_stack_from_sequences(
    sequences: Iterable[Sequence[Symbol]],
    constraints: ConstraintSet | None = None,
    *,
    max_order: int,
    length: int,
    prefix: Sequence[Symbol],
    policy: OrderPolicy | None = None,
    start_symbol: Symbol | None = None,
    end_symbol: Symbol | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
) -> ConstrainedOrderStackBackend:
    """Build an order-stack model from sequences and prepare a backend."""

    model = OrderStackModel.from_sequences(
        sequences,
        max_order=max_order,
        start_symbol=start_symbol,
        end_symbol=end_symbol,
    )
    return prepare_constrained_order_stack(
        model,
        constraints,
        length=length,
        prefix=prefix,
        policy=policy,
        alphabet=alphabet,
        prefer_dense_forbidden=prefer_dense_forbidden,
    )
