"""Exact constrained sampling for sparse variable-order context models.

The top-level package exports the stable library surface: generic product BP,
Continuator-style order-stack backends, event adapters, constraint builders, and
diagnostic helpers.  Lower-level modules remain importable for experiments, but
external projects should prefer the names exported here.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _metadata_version

try:
    __version__ = _metadata_version("vo-regular-bp")
except _PackageNotFoundError:
    __version__ = "0+unknown"

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
from .adapters import (
    EventCodec,
    EventOrderStackBackend,
    GeneratedEvents,
    infer_symbol_to_event,
    prepare_constrained_order_stack_from_events,
)
from .augmentation import (
    SymbolTransform,
    VirtualAugmentedOrderStackModel,
    integer_shift_transform,
    integer_shift_transforms,
    materialize_transformed_sequences,
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
    UntilLengthDiagnostics,
    UntilOrderStackBackend,
    UntilOrderStackDiagnostics,
    prepare_constrained_order_stack,
    prepare_constrained_order_stack_from_sequences,
    prepare_until_end_order_stack,
    prepare_until_order_stack,
    run_constrained_order_stack,
)
from .constraint_builders import (
    at_position,
    avoid_copied_ngrams,
    combine_constraints,
    cumulative_meter,
    final_pitch_class,
    final_symbol,
    final_symbols,
    meter_pattern,
    padded_duration_total,
)
from .continuator import (
    duration_total_constraint,
    final_pitch_class_constraint,
    meter_cycle_constraint,
    padded_duration_total_constraint,
    prepare_continuation_backend,
)
from .constraints import (
    CompiledConstraints,
    ConstraintSet,
    CumulativeMeterConstraint,
    MeterConstraint,
    compile_constraints,
)
from .context import Context, ContextGraph, Edge, Symbol
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
from .orbit_diagnostics import (
    ForbiddenPatternOrbitStats,
    RegularProductOrbitStats,
    RegularRowSignatureStats,
    canonical_integer_shift_key,
    forbidden_pattern_orbit_stats,
    regular_product_orbit_stats,
    regular_row_signature_stats,
)
from .padding import append_padding
from .positional_bp import LazyBackoffContextModel, PositionalBPResult, run_positional_bp
from .product_bp import ProductBPResult, run_bp, sample_exact

__all__ = [
    "__version__",
    # Core symbolic graph and DFA API.
    "Context",
    "ContextGraph",
    "DFA",
    "DenseForbiddenSubstringDFA",
    "Edge",
    "Symbol",
    "all_of",
    "cumulative_meter_acceptor",
    "dense_forbidden_substring_acceptor",
    "forbidden_substring_acceptor",
    "max_order_acceptor",
    "meter_acceptor",
    "positional_acceptor",
    "true_acceptor",
    # General exact product BP.
    "LazyBackoffContextModel",
    "PositionalBPResult",
    "ProductBPResult",
    "run_bp",
    "run_positional_bp",
    "sample_exact",
    # Library-facing order-stack backend.
    "BackendDiagnostics",
    "ConstrainedOrderStackBackend",
    "GeneratedSequence",
    "LongestFeasiblePolicy",
    "OrderStackBPResult",
    "OrderStackModel",
    "RegularOrderStackBPResult",
    "SingletonAvoidingBackoffPolicy",
    "UntilLengthDiagnostics",
    "UntilOrderStackBackend",
    "UntilOrderStackDiagnostics",
    "prepare_constrained_order_stack",
    "prepare_constrained_order_stack_from_sequences",
    "prepare_until_end_order_stack",
    "prepare_until_order_stack",
    "run_constrained_order_stack",
    "run_order_stack_bp",
    "run_order_stack_dfa_bp",
    "run_order_stack_masked_dfa_bp",
    # Transformation-orbit diagnostics.
    "ForbiddenPatternOrbitStats",
    "RegularProductOrbitStats",
    "RegularRowSignatureStats",
    "canonical_integer_shift_key",
    "forbidden_pattern_orbit_stats",
    "regular_product_orbit_stats",
    "regular_row_signature_stats",
    # Virtual data augmentation.
    "SymbolTransform",
    "VirtualAugmentedOrderStackModel",
    "integer_shift_transform",
    "integer_shift_transforms",
    "materialize_transformed_sequences",
    # Fixed-horizon padding helpers.
    "append_padding",
    # Constraint specifications and builders.
    "CompiledConstraints",
    "ConstraintSet",
    "CumulativeMeterConstraint",
    "MeterConstraint",
    "at_position",
    "avoid_copied_ngrams",
    "combine_constraints",
    "compile_constraints",
    "cumulative_meter",
    "final_pitch_class",
    "final_symbol",
    "final_symbols",
    "meter_pattern",
    "padded_duration_total",
    # Event and Continuator-style adapters.
    "EventCodec",
    "EventOrderStackBackend",
    "GeneratedEvents",
    "duration_total_constraint",
    "final_pitch_class_constraint",
    "infer_symbol_to_event",
    "meter_cycle_constraint",
    "padded_duration_total_constraint",
    "prepare_constrained_order_stack_from_events",
    "prepare_continuation_backend",
    # Analysis/test helpers that are useful for exactness checks.
    "brute_force_distribution",
    "brute_force_partition_function",
    "conditional_distribution",
    "empirical_distribution",
    "total_variation",
]
