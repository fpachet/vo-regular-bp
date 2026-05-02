"""Exact constrained sampling for sparse variable-order context models."""

from .acceptors import (
    DFA,
    DenseForbiddenSubstringDFA,
    all_of,
    cumulative_meter_acceptor,
    dense_forbidden_substring_acceptor,
    forbidden_substring_acceptor,
    max_order_acceptor,
    meter_acceptor,
    positional_acceptor,
    true_acceptor,
)
from .brute_force import (
    brute_force_distribution,
    brute_force_partition_function,
    conditional_distribution,
)
from .backend import (
    BackendDiagnostics,
    ConstrainedOrderStackBackend,
    GeneratedSequence,
    prepare_constrained_order_stack,
    prepare_constrained_order_stack_from_sequences,
    run_constrained_order_stack,
)
from .constraints import (
    CompiledConstraints,
    ConstraintSet,
    CumulativeMeterConstraint,
    MeterConstraint,
    compile_constraints,
)
from .context import ContextGraph, Edge
from .metrics import empirical_distribution, total_variation
from .order_stack_bp import (
    LongestFeasiblePolicy,
    OrderStackBPResult,
    OrderStackModel,
    RegularOrderStackBPResult,
    SingletonAvoidingBackoffPolicy,
    run_order_stack_dfa_bp,
    run_order_stack_masked_dfa_bp,
    run_order_stack_bp,
)
from .positional_bp import LazyBackoffContextModel, PositionalBPResult, run_positional_bp
from .product_bp import ProductBPResult, run_bp, sample_exact

__all__ = [
    "ContextGraph",
    "BackendDiagnostics",
    "CompiledConstraints",
    "ConstrainedOrderStackBackend",
    "ConstraintSet",
    "CumulativeMeterConstraint",
    "DFA",
    "DenseForbiddenSubstringDFA",
    "Edge",
    "GeneratedSequence",
    "LazyBackoffContextModel",
    "LongestFeasiblePolicy",
    "MeterConstraint",
    "OrderStackBPResult",
    "OrderStackModel",
    "PositionalBPResult",
    "ProductBPResult",
    "RegularOrderStackBPResult",
    "SingletonAvoidingBackoffPolicy",
    "all_of",
    "brute_force_distribution",
    "brute_force_partition_function",
    "conditional_distribution",
    "compile_constraints",
    "cumulative_meter_acceptor",
    "empirical_distribution",
    "dense_forbidden_substring_acceptor",
    "forbidden_substring_acceptor",
    "max_order_acceptor",
    "meter_acceptor",
    "positional_acceptor",
    "prepare_constrained_order_stack",
    "prepare_constrained_order_stack_from_sequences",
    "run_constrained_order_stack",
    "run_order_stack_dfa_bp",
    "run_order_stack_masked_dfa_bp",
    "run_order_stack_bp",
    "run_positional_bp",
    "run_bp",
    "sample_exact",
    "total_variation",
    "true_acceptor",
]
