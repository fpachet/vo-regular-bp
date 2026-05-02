from math import isclose

from scripts.eval_tiny_exactness import (
    analyze_experiment,
    build_forbidden_substring_experiment,
    build_paper_integer_experiment,
)


def has_forbidden_22(sequence: tuple[int, ...]) -> bool:
    return any(left == 2 and right == 2 for left, right in zip(sequence, sequence[1:]))


def test_paper_integer_future_positional_constraint():
    result = analyze_experiment(
        build_paper_integer_experiment(),
        sample_count=5_000,
        seed=123,
    )

    assert isclose(result.z_bp, 11 / 21, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(result.z_brute, result.z_bp, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(result.exact[(2, 4)], 10 / 11, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(result.exact[(3, 4)], 1 / 11, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(result.bp_distribution[(2, 4)], 10 / 11, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(result.bp_distribution[(3, 4)], 1 / 11, rel_tol=1e-12, abs_tol=1e-12)
    assert all(sequence[1] == 4 for sequence in result.samples)


def test_forbidden_substring_exactness_and_sampling():
    result = analyze_experiment(
        build_forbidden_substring_experiment(),
        sample_count=30_000,
        seed=456,
    )

    assert isclose(result.z_bp, result.z_brute, rel_tol=1e-12, abs_tol=1e-12)
    assert result.z_bp > 0.0
    assert result.tv_exact_bp < 1e-12
    assert result.tv_exact_empirical < 0.05
    assert all(not has_forbidden_22(sequence) for sequence in result.samples)
