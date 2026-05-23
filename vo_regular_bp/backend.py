"""Stable public backend entry points.

This module is the library-facing layer.  It compiles generic constraint
specifications and delegates to the current BP engines.  Future optimized
engines should be wired behind these functions without changing the public
surface.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
import random

from .acceptors import DFA
from .constraint_builders import combine_constraints
from .constraints import ConstraintSet, compile_constraints
from .context import Symbol
from .order_stack_bp import (
    OrderPolicy,
    OrderSampleStep,
    OrderStackBPPlan,
    OrderStackModel,
    RegularOrderStackBPResult,
    RegularOrderStackBPPlan,
    OrderStackBPResult,
    prepare_order_stack_bp,
    prepare_order_stack_masked_dfa_bp,
    run_order_stack_bp,
    run_order_stack_masked_dfa_bp,
)
from .positional_bp import AllowedForbiddenSymbols


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
    regular_transition_rows: int | None = None
    regular_transition_row_cache_hits: int | None = None
    regular_transition_row_cache_misses: int | None = None
    regular_accepted_transitions: int | None = None
    regular_beta_state_expansions: int | None = None
    regular_beta_cache_hits: int | None = None
    regular_beta_cache_misses: int | None = None
    regular_acceptor_symbol_transition_cache_hits: int | None = None
    regular_acceptor_symbol_transition_cache_misses: int | None = None
    duration_view_quotient_states: int | None = None
    duration_view_quotient_classes: int | None = None
    duration_view_quotient_state_reduction: float | None = None
    duration_view_quotient_edges: int | None = None
    duration_view_quotient_projected_edges: int | None = None
    duration_view_quotient_ignored_edges: int | None = None
    duration_view_quotient_quotient_edges: int | None = None
    duration_view_quotient_projected_edge_reduction: float | None = None
    duration_view_quotient_max_class_size: int | None = None
    duration_view_quotient_refinement_rounds: int | None = None
    duration_view_quotient_seconds: float | None = None
    duration_view_quotient_order_stats: tuple[dict[str, object], ...] = ()
    virtual_context_materialization_seconds: float | None = None
    virtual_context_materialization_calls: int | None = None
    virtual_context_materialization_cache_hits: int | None = None
    virtual_context_materialization_cache_misses: int | None = None
    augmented_count_calls: int | None = None
    augmented_count_cache_hits: int | None = None
    augmented_count_cache_misses: int | None = None
    virtual_outgoing_row_calls: int | None = None
    virtual_outgoing_row_cache_hits: int | None = None
    virtual_outgoing_row_cache_misses: int | None = None
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
            "regular_transition_rows": self.regular_transition_rows,
            "regular_transition_row_cache_hits": self.regular_transition_row_cache_hits,
            "regular_transition_row_cache_misses": self.regular_transition_row_cache_misses,
            "regular_accepted_transitions": self.regular_accepted_transitions,
            "regular_beta_state_expansions": self.regular_beta_state_expansions,
            "regular_beta_cache_hits": self.regular_beta_cache_hits,
            "regular_beta_cache_misses": self.regular_beta_cache_misses,
            "regular_acceptor_symbol_transition_cache_hits": (
                self.regular_acceptor_symbol_transition_cache_hits
            ),
            "regular_acceptor_symbol_transition_cache_misses": (
                self.regular_acceptor_symbol_transition_cache_misses
            ),
            "duration_view_quotient_states": self.duration_view_quotient_states,
            "duration_view_quotient_classes": self.duration_view_quotient_classes,
            "duration_view_quotient_state_reduction": (
                self.duration_view_quotient_state_reduction
            ),
            "duration_view_quotient_edges": self.duration_view_quotient_edges,
            "duration_view_quotient_projected_edges": (
                self.duration_view_quotient_projected_edges
            ),
            "duration_view_quotient_ignored_edges": (
                self.duration_view_quotient_ignored_edges
            ),
            "duration_view_quotient_quotient_edges": (
                self.duration_view_quotient_quotient_edges
            ),
            "duration_view_quotient_projected_edge_reduction": (
                self.duration_view_quotient_projected_edge_reduction
            ),
            "duration_view_quotient_max_class_size": (
                self.duration_view_quotient_max_class_size
            ),
            "duration_view_quotient_refinement_rounds": (
                self.duration_view_quotient_refinement_rounds
            ),
            "duration_view_quotient_seconds": self.duration_view_quotient_seconds,
            "duration_view_quotient_order_stats": (
                self.duration_view_quotient_order_stats
            ),
            "virtual_context_materialization_seconds": (
                self.virtual_context_materialization_seconds
            ),
            "virtual_context_materialization_calls": self.virtual_context_materialization_calls,
            "virtual_context_materialization_cache_hits": (
                self.virtual_context_materialization_cache_hits
            ),
            "virtual_context_materialization_cache_misses": (
                self.virtual_context_materialization_cache_misses
            ),
            "augmented_count_calls": self.augmented_count_calls,
            "augmented_count_cache_hits": self.augmented_count_cache_hits,
            "augmented_count_cache_misses": self.augmented_count_cache_misses,
            "virtual_outgoing_row_calls": self.virtual_outgoing_row_calls,
            "virtual_outgoing_row_cache_hits": self.virtual_outgoing_row_cache_hits,
            "virtual_outgoing_row_cache_misses": self.virtual_outgoing_row_cache_misses,
            "success_mass": self.success_mass,
            "start_order_masses": self.start_order_masses,
        }


@dataclass(frozen=True)
class UntilLengthDiagnostics:
    """Diagnostics for one feasible first-hit continuation length."""

    length: int
    weight: float
    backend: BackendDiagnostics

    def as_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "weight": self.weight,
            "backend": self.backend.as_dict(),
        }


@dataclass(frozen=True)
class UntilOrderStackDiagnostics:
    """Diagnostics exposed by variable-length first-hit backends."""

    backend: str
    min_length: int
    max_length: int
    feasible_lengths: tuple[int, ...]
    length_weights: tuple[tuple[int, float], ...]
    length_diagnostics: tuple[UntilLengthDiagnostics, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "feasible_lengths": self.feasible_lengths,
            "length_weights": self.length_weights,
            "length_diagnostics": tuple(
                diagnostic.as_dict() for diagnostic in self.length_diagnostics
            ),
        }


def _virtual_model_diagnostics(model: object) -> dict[str, object]:
    diagnostics = getattr(model, "virtual_diagnostics", None)
    if callable(diagnostics):
        return dict(diagnostics())
    return {}


def _virtual_graph_outgoing_diagnostics(
    graphs: object,
) -> dict[str, int]:
    values = tuple(graphs.values()) if isinstance(graphs, dict) else ()
    return {
        "virtual_outgoing_row_calls": sum(
            int(getattr(graph, "outgoing_row_calls", 0)) for graph in values
        ),
        "virtual_outgoing_row_cache_hits": sum(
            int(getattr(graph, "outgoing_row_cache_hits", 0))
            for graph in values
        ),
        "virtual_outgoing_row_cache_misses": sum(
            int(getattr(graph, "outgoing_row_cache_misses", 0))
            for graph in values
        ),
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
        virtual_diagnostics = _virtual_model_diagnostics(self.result.model)
        virtual_outgoing_diagnostics = _virtual_graph_outgoing_diagnostics(
            self.result.graphs,
        )
        duration_view = (
            self.result.duration_view_quotient_diagnostics
            if is_regular
            else None
        )
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
            regular_transition_rows=(
                self.result.regular_transition_row_count if is_regular else None
            ),
            regular_transition_row_cache_hits=(
                self.result.regular_transition_row_cache_hits if is_regular else None
            ),
            regular_transition_row_cache_misses=(
                self.result.regular_transition_row_cache_misses if is_regular else None
            ),
            regular_accepted_transitions=(
                self.result.regular_accepted_transition_count if is_regular else None
            ),
            regular_beta_state_expansions=(
                self.result.regular_beta_state_expansions if is_regular else None
            ),
            regular_beta_cache_hits=(
                self.result.regular_beta_cache_hits if is_regular else None
            ),
            regular_beta_cache_misses=(
                self.result.regular_beta_cache_misses if is_regular else None
            ),
            regular_acceptor_symbol_transition_cache_hits=(
                self.result.regular_acceptor_symbol_transition_cache_hits
                if is_regular
                else None
            ),
            regular_acceptor_symbol_transition_cache_misses=(
                self.result.regular_acceptor_symbol_transition_cache_misses
                if is_regular
                else None
            ),
            duration_view_quotient_states=(
                duration_view.states if duration_view is not None else None
            ),
            duration_view_quotient_classes=(
                duration_view.classes if duration_view is not None else None
            ),
            duration_view_quotient_state_reduction=(
                duration_view.state_reduction if duration_view is not None else None
            ),
            duration_view_quotient_edges=(
                duration_view.edges if duration_view is not None else None
            ),
            duration_view_quotient_projected_edges=(
                duration_view.projected_edges if duration_view is not None else None
            ),
            duration_view_quotient_ignored_edges=(
                duration_view.ignored_edges if duration_view is not None else None
            ),
            duration_view_quotient_quotient_edges=(
                duration_view.quotient_edges if duration_view is not None else None
            ),
            duration_view_quotient_projected_edge_reduction=(
                duration_view.projected_edge_reduction
                if duration_view is not None
                else None
            ),
            duration_view_quotient_max_class_size=(
                duration_view.max_class_size if duration_view is not None else None
            ),
            duration_view_quotient_refinement_rounds=(
                duration_view.refinement_rounds if duration_view is not None else None
            ),
            duration_view_quotient_seconds=(
                duration_view.seconds if duration_view is not None else None
            ),
            duration_view_quotient_order_stats=(
                tuple(
                    stats.as_dict()
                    for stats in duration_view.orders
                )
                if duration_view is not None
                else ()
            ),
            virtual_context_materialization_seconds=virtual_diagnostics.get(
                "virtual_context_materialization_seconds",
            ),
            virtual_context_materialization_calls=virtual_diagnostics.get(
                "virtual_context_materialization_calls",
            ),
            virtual_context_materialization_cache_hits=virtual_diagnostics.get(
                "virtual_context_materialization_cache_hits",
            ),
            virtual_context_materialization_cache_misses=virtual_diagnostics.get(
                "virtual_context_materialization_cache_misses",
            ),
            augmented_count_calls=virtual_diagnostics.get("augmented_count_calls"),
            augmented_count_cache_hits=virtual_diagnostics.get("augmented_count_cache_hits"),
            augmented_count_cache_misses=virtual_diagnostics.get(
                "augmented_count_cache_misses",
            ),
            virtual_outgoing_row_calls=virtual_outgoing_diagnostics[
                "virtual_outgoing_row_calls"
            ],
            virtual_outgoing_row_cache_hits=virtual_outgoing_diagnostics[
                "virtual_outgoing_row_cache_hits"
            ],
            virtual_outgoing_row_cache_misses=virtual_outgoing_diagnostics[
                "virtual_outgoing_row_cache_misses"
            ],
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


@dataclass(frozen=True)
class ConstrainedOrderStackPlan:
    """Prefix-independent constrained order-stack preparation.

    Keep this object around when the model, horizon, and constraints are fixed
    but callers need to sample from many different prefixes. ``for_prefix`` is
    cheap and returns the existing prefix-bound backend wrapper.
    """

    plan: OrderStackBPPlan | RegularOrderStackBPPlan

    @property
    def length(self) -> int:
        return self.plan.length

    @property
    def is_regular(self) -> bool:
        return isinstance(self.plan, RegularOrderStackBPPlan)

    def for_prefix(self, prefix: Sequence[Symbol]) -> ConstrainedOrderStackBackend:
        return ConstrainedOrderStackBackend(self.plan.for_prefix(prefix))

    def sample(
        self,
        *,
        prefix: Sequence[Symbol],
        rng: random.Random | int | None = None,
    ) -> tuple[Symbol, ...]:
        return self.for_prefix(prefix).sample(rng=rng)

    def sample_with_orders(
        self,
        *,
        prefix: Sequence[Symbol],
        rng: random.Random | int | None = None,
    ) -> GeneratedSequence:
        return self.for_prefix(prefix).sample_with_orders(rng=rng)


@dataclass(frozen=True)
class ConstrainedOrderStackSupportPlan:
    """Reusable hard-support plan for dynamic soft regular acceptors.

    The wrapped hard plan owns the compiled source/order-stack graphs and stable
    constraints. ``with_soft_acceptor`` creates a prefix-bound backend with fresh
    regular caches so changing soft weights or topology cannot reuse stale beta
    values.
    """

    plan: ConstrainedOrderStackPlan

    @property
    def length(self) -> int:
        return self.plan.length

    @property
    def is_regular(self) -> bool:
        return self.plan.is_regular

    def for_prefix(self, prefix: Sequence[Symbol]) -> ConstrainedOrderStackBackend:
        """Bind the hard-only support plan to a prefix."""

        return self.plan.for_prefix(prefix)

    def with_soft_acceptor(
        self,
        soft_acceptor: DFA | None,
        *,
        prefix: Sequence[Symbol],
        soft_start_acceptor_state: Hashable | None = None,
    ) -> ConstrainedOrderStackBackend:
        """Bind a dynamic soft acceptor on top of the cached hard support.

        Passing ``None`` samples from the hard support only. For a real soft
        acceptor, the returned backend has new regular transition-row and beta
        caches but reuses the source/order-stack graphs and stable masks.
        """

        if soft_acceptor is None:
            return self.for_prefix(prefix)

        lower_plan = self.plan.plan
        soft0 = (
            soft_acceptor.start_state
            if soft_start_acceptor_state is None
            else soft_start_acceptor_state
        )
        if isinstance(lower_plan, RegularOrderStackBPPlan):
            soft_plan = lower_plan.with_soft_acceptor(
                soft_acceptor,
                start_acceptor_state=(lower_plan.start_acceptor_state, soft0),
            )
        else:
            soft_plan = lower_plan.with_regular_acceptor(
                soft_acceptor,
                start_acceptor_state=soft0,
            )
        return ConstrainedOrderStackBackend(soft_plan.for_prefix(prefix))


@dataclass(frozen=True)
class UntilOrderStackBackend:
    """Prepared reusable first-hit order-stack sampler."""

    backends: tuple[ConstrainedOrderStackBackend, ...]
    length_weights: tuple[float, ...]
    min_length: int
    max_length: int

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("at least one feasible length backend is required")
        if len(self.backends) != len(self.length_weights):
            raise ValueError("backends and length_weights must have the same length")
        if any(weight < 0.0 for weight in self.length_weights):
            raise ValueError("length weights must be non-negative")
        if sum(self.length_weights) <= 0.0:
            raise ValueError("at least one length weight must be positive")

    @property
    def feasible_lengths(self) -> tuple[int, ...]:
        return tuple(backend.result.length for backend in self.backends)

    @property
    def diagnostics(self) -> UntilOrderStackDiagnostics:
        length_diagnostics = tuple(
            UntilLengthDiagnostics(
                length=backend.result.length,
                weight=weight,
                backend=backend.diagnostics,
            )
            for backend, weight in zip(self.backends, self.length_weights)
        )
        return UntilOrderStackDiagnostics(
            backend="order_stack_until",
            min_length=self.min_length,
            max_length=self.max_length,
            feasible_lengths=self.feasible_lengths,
            length_weights=tuple(
                (backend.result.length, weight)
                for backend, weight in zip(self.backends, self.length_weights)
            ),
            length_diagnostics=length_diagnostics,
        )

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        generator = _coerce_backend_rng(rng)
        return self._choose_backend(generator).sample(rng=generator)

    def sample_many(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[Symbol, ...]]:
        generator = _coerce_backend_rng(rng)
        return [self.sample(rng=generator) for _ in range(count)]

    def sample_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> GeneratedSequence:
        generator = _coerce_backend_rng(rng)
        return self._choose_backend(generator).sample_with_orders(rng=generator)

    def sample_many_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[GeneratedSequence]:
        generator = _coerce_backend_rng(rng)
        return [self.sample_with_orders(rng=generator) for _ in range(count)]

    def sample_with_trace(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[OrderSampleStep, ...]]:
        generator = _coerce_backend_rng(rng)
        return self._choose_backend(generator).sample_with_trace(rng=generator)

    def _choose_backend(self, rng: random.Random) -> ConstrainedOrderStackBackend:
        cumulative = _cumulative_weights(self.length_weights)
        index = bisect.bisect_left(cumulative, rng.random() * cumulative[-1])
        if index >= len(self.backends):
            index = len(self.backends) - 1
        return self.backends[index]


def run_constrained_order_stack(
    model: OrderStackModel,
    constraints: ConstraintSet | None = None,
    *,
    length: int,
    prefix: Sequence[Symbol],
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    minimize_source_graphs: bool = False,
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
            allowed_forbidden_symbols=allowed_forbidden_symbols,
            policy=policy,
            minimize_source_graphs=minimize_source_graphs,
        )
    return run_order_stack_masked_dfa_bp(
        model,
        compiled.regular_acceptor,
        length=length,
        prefix=prefix,
        constraints=compiled.positional,
        allowed_forbidden_symbols=allowed_forbidden_symbols,
        policy=policy,
        minimize_source_graphs=minimize_source_graphs,
    )


def prepare_constrained_order_stack_plan(
    model: OrderStackModel,
    constraints: ConstraintSet | None = None,
    *,
    length: int,
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    minimize_source_graphs: bool = False,
) -> ConstrainedOrderStackPlan:
    """Prepare reusable constrained order-stack BP without binding a prefix."""

    compiled = compile_constraints(
        constraints,
        length=length,
        alphabet=tuple(model.alphabet if alphabet is None else alphabet),
        prefer_dense_forbidden=prefer_dense_forbidden,
    )
    if compiled.regular_acceptor is None:
        return ConstrainedOrderStackPlan(
            prepare_order_stack_bp(
                model,
                length=length,
                constraints=compiled.positional,
                allowed_forbidden_symbols=allowed_forbidden_symbols,
                policy=policy,
                minimize_source_graphs=minimize_source_graphs,
            )
        )
    return ConstrainedOrderStackPlan(
        prepare_order_stack_masked_dfa_bp(
            model,
            compiled.regular_acceptor,
            length=length,
            constraints=compiled.positional,
            allowed_forbidden_symbols=allowed_forbidden_symbols,
            policy=policy,
            minimize_source_graphs=minimize_source_graphs,
        )
    )


def prepare_constrained_order_stack_support_plan(
    model: OrderStackModel,
    constraints: ConstraintSet | None = None,
    *,
    length: int,
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    minimize_source_graphs: bool = False,
) -> ConstrainedOrderStackSupportPlan:
    """Prepare hard support once for repeated dynamic soft regular weighting."""

    return ConstrainedOrderStackSupportPlan(
        prepare_constrained_order_stack_plan(
            model,
            constraints,
            length=length,
            policy=policy,
            alphabet=alphabet,
            prefer_dense_forbidden=prefer_dense_forbidden,
            allowed_forbidden_symbols=allowed_forbidden_symbols,
            minimize_source_graphs=minimize_source_graphs,
        )
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
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    minimize_source_graphs: bool = False,
) -> ConstrainedOrderStackBackend:
    """Compile constraints and return a reusable order-stack backend."""

    try:
        plan = prepare_constrained_order_stack_plan(
            model,
            constraints,
            length=length,
            policy=policy,
            alphabet=alphabet,
            prefer_dense_forbidden=prefer_dense_forbidden,
            allowed_forbidden_symbols=allowed_forbidden_symbols,
            minimize_source_graphs=minimize_source_graphs,
        )
    except ValueError as error:
        if "prefix-independent graph compilation" not in str(error):
            raise
        return ConstrainedOrderStackBackend(
            run_constrained_order_stack(
                model,
                constraints,
                length=length,
                prefix=prefix,
                policy=policy,
                alphabet=alphabet,
                prefer_dense_forbidden=prefer_dense_forbidden,
                allowed_forbidden_symbols=allowed_forbidden_symbols,
                minimize_source_graphs=minimize_source_graphs,
            )
        )
    return plan.for_prefix(prefix)


def prepare_constrained_order_stack_plan_from_sequences(
    sequences: Iterable[Sequence[Symbol]],
    constraints: ConstraintSet | None = None,
    *,
    max_order: int,
    length: int,
    policy: OrderPolicy | None = None,
    start_symbol: Symbol | None = None,
    end_symbol: Symbol | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    minimize_source_graphs: bool = False,
) -> ConstrainedOrderStackPlan:
    """Build an order-stack model from sequences and prepare a prefixless plan."""

    model = OrderStackModel.from_sequences(
        sequences,
        max_order=max_order,
        start_symbol=start_symbol,
        end_symbol=end_symbol,
    )
    return prepare_constrained_order_stack_plan(
        model,
        constraints,
        length=length,
        policy=policy,
        alphabet=alphabet,
        prefer_dense_forbidden=prefer_dense_forbidden,
        minimize_source_graphs=minimize_source_graphs,
    )


def prepare_constrained_order_stack_support_plan_from_sequences(
    sequences: Iterable[Sequence[Symbol]],
    constraints: ConstraintSet | None = None,
    *,
    max_order: int,
    length: int,
    policy: OrderPolicy | None = None,
    start_symbol: Symbol | None = None,
    end_symbol: Symbol | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    minimize_source_graphs: bool = False,
) -> ConstrainedOrderStackSupportPlan:
    """Build an order-stack model once and prepare reusable hard support."""

    return ConstrainedOrderStackSupportPlan(
        prepare_constrained_order_stack_plan_from_sequences(
            sequences,
            constraints,
            max_order=max_order,
            length=length,
            policy=policy,
            start_symbol=start_symbol,
            end_symbol=end_symbol,
            alphabet=alphabet,
            prefer_dense_forbidden=prefer_dense_forbidden,
            minimize_source_graphs=minimize_source_graphs,
        )
    )


def prepare_until_order_stack(
    model: OrderStackModel,
    *,
    prefix: Sequence[Symbol],
    stop: Symbol | Iterable[Symbol] | Callable[[Symbol], bool],
    min_length: int = 1,
    max_length: int = 64,
    constraints: ConstraintSet | None = None,
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    minimize_source_graphs: bool = False,
) -> UntilOrderStackBackend:
    """Prepare variable-length first-hit continuation for the order stack.

    The generated suffix has length in ``[min_length, max_length]``. The final
    emitted symbol satisfies ``stop`` and no earlier emitted symbol does.
    Lengths are sampled proportionally to the sum of each feasible backend's
    positive start-order masses, falling back to unit weight if only a success
    indicator is available.
    """

    if min_length < 1:
        raise ValueError("min_length must be at least 1")
    if max_length < min_length:
        raise ValueError("max_length must be greater than or equal to min_length")
    if not prefix:
        raise ValueError("order-stack BP requires a non-empty prefix")

    stop_predicate, stop_symbols = _coerce_stop_condition(
        stop,
        known_symbols=model.alphabet,
    )
    prepared: list[ConstrainedOrderStackBackend] = []
    weights: list[float] = []

    for length in range(min_length, max_length + 1):
        if not _constraints_compatible_with_length(constraints, length):
            continue
        first_hit = _first_hit_constraints(stop_predicate, length)
        combined = combine_constraints(constraints, first_hit)
        allowed_forbidden = _allowed_forbidden_stop_symbols(
            model,
            stop_predicate,
            stop_symbols,
            length,
        )
        backend = prepare_constrained_order_stack(
            model,
            combined,
            length=length,
            prefix=prefix,
            policy=policy,
            alphabet=alphabet,
            prefer_dense_forbidden=prefer_dense_forbidden,
            allowed_forbidden_symbols=allowed_forbidden,
            minimize_source_graphs=minimize_source_graphs,
        )
        weight = _length_weight(backend)
        if weight <= 0.0:
            continue
        prepared.append(backend)
        weights.append(weight)

    if not prepared:
        raise ValueError("No feasible first-hit continuation length satisfies the constraints.")

    return UntilOrderStackBackend(
        backends=tuple(prepared),
        length_weights=tuple(weights),
        min_length=min_length,
        max_length=max_length,
    )


def prepare_until_end_order_stack(
    model: OrderStackModel,
    *,
    prefix: Sequence[Symbol],
    end_symbol: Symbol,
    min_length: int = 1,
    max_length: int = 64,
    constraints: ConstraintSet | None = None,
    policy: OrderPolicy | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    minimize_source_graphs: bool = False,
) -> UntilOrderStackBackend:
    """Prepare first-hit continuation that stops at ``end_symbol``."""

    return prepare_until_order_stack(
        model,
        prefix=prefix,
        stop=end_symbol,
        min_length=min_length,
        max_length=max_length,
        constraints=constraints,
        policy=policy,
        alphabet=alphabet,
        prefer_dense_forbidden=prefer_dense_forbidden,
        minimize_source_graphs=minimize_source_graphs,
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
    minimize_source_graphs: bool = False,
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
        minimize_source_graphs=minimize_source_graphs,
    )


def _coerce_stop_condition(
    stop: Symbol | Iterable[Symbol] | Callable[[Symbol], bool],
    *,
    known_symbols: Iterable[Symbol] = (),
) -> tuple[Callable[[Symbol], bool], frozenset[Symbol] | None]:
    if callable(stop):
        return lambda symbol: bool(stop(symbol)), None
    known = frozenset(known_symbols)
    if isinstance(stop, (str, bytes)) or _is_known_symbol(stop, known):
        symbols = frozenset({stop})
    else:
        try:
            symbols = frozenset(stop)  # type: ignore[arg-type]
        except TypeError:
            symbols = frozenset({stop})  # type: ignore[list-item]
    if not symbols:
        raise ValueError("stop symbol set must not be empty")
    return lambda symbol: symbol in symbols, symbols


def _constraints_compatible_with_length(
    constraints: ConstraintSet | None,
    length: int,
) -> bool:
    if constraints is None:
        return True
    for position in constraints.positional:
        if position < 0:
            raise IndexError(f"constraint position {position} is outside length {length}")
        if position >= length:
            return False
    if constraints.meter is not None and len(constraints.meter.pattern) != length:
        return False
    cumulative = constraints.cumulative_meter
    if cumulative is not None and cumulative.length is not None and cumulative.length != length:
        return False
    return True


def _is_known_symbol(value: object, known_symbols: frozenset[Symbol]) -> bool:
    if not isinstance(value, Hashable):
        return False
    try:
        return value in known_symbols
    except TypeError:
        return False


def _first_hit_constraints(
    stop_predicate: Callable[[Symbol], bool],
    length: int,
) -> ConstraintSet:
    positional: dict[int, Callable[[Symbol], bool]] = {
        position: _negate_predicate(stop_predicate)
        for position in range(length - 1)
    }
    positional[length - 1] = stop_predicate
    return ConstraintSet(positional=positional)


def _negate_predicate(predicate: Callable[[Symbol], bool]) -> Callable[[Symbol], bool]:
    return lambda symbol: not predicate(symbol)


def _allowed_forbidden_stop_symbols(
    model: OrderStackModel,
    stop_predicate: Callable[[Symbol], bool],
    stop_symbols: frozenset[Symbol] | None,
    length: int,
) -> dict[int, frozenset[Symbol]]:
    if stop_symbols is None:
        allowed = frozenset(
            symbol
            for symbol in model.forbidden_symbols
            if _predicate_accepts(stop_predicate, symbol)
        )
    else:
        allowed = frozenset(symbol for symbol in stop_symbols if symbol in model.forbidden_symbols)
    return {length - 1: allowed} if allowed else {}


def _predicate_accepts(predicate: Callable[[Symbol], bool], symbol: Symbol) -> bool:
    try:
        return bool(predicate(symbol))
    except Exception:
        return False


def _length_weight(backend: ConstrainedOrderStackBackend) -> float:
    mass = sum(
        max(0.0, float(start_mass))
        for _order, start_mass in backend.result.start_order_masses()
    )
    if mass > 0.0:
        return mass
    return 1.0 if backend.result.success_mass > 0.0 else 0.0


def _coerce_backend_rng(rng: random.Random | int | None) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    return random.Random(rng)


def _cumulative_weights(weights: Sequence[float]) -> tuple[float, ...]:
    total = 0.0
    cumulative: list[float] = []
    for weight in weights:
        total += float(weight)
        cumulative.append(total)
    return tuple(cumulative)
