from math import sqrt

import numpy as np
import pandas as pd
import pytest

from housing_dss.decision import DEFAULT_CRITERIA, DEFAULT_MATRIX, ahp_weights, rank_properties


def properties() -> pd.DataFrame:
    return pd.DataFrame({
        "price_rub": [100.0, 200.0, 100.0],
        "district_score": [5.0, 5.0, 10.0],
        "travel_minutes": [10.0, 20.0, 10.0],
        "source_url": ["first", "second", "third"],
    })


def test_consistent_matrix_has_known_principal_eigenvector() -> None:
    result = ahp_weights([[1, 2, 6], [1 / 2, 1, 3], [1 / 6, 1 / 3, 1]])
    np.testing.assert_allclose(result.weights, [0.6, 0.3, 0.1])
    assert result.lambda_max == pytest.approx(3)
    assert result.consistency_index == pytest.approx(0, abs=1e-12)
    assert result.consistency_ratio == pytest.approx(0, abs=1e-12)
    assert result.is_consistent


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [([[1]], [1]), ([[1, 4], [1 / 4, 1]], [0.8, 0.2])],
)
def test_small_matrices_have_zero_consistency_ratio(matrix, expected) -> None:
    result = ahp_weights(matrix)
    np.testing.assert_allclose(result.weights, expected)
    assert result.consistency_index == 0
    assert result.consistency_ratio == 0


def test_default_matrix_is_consistent_and_inconsistent_cycle_is_detected() -> None:
    default = ahp_weights(DEFAULT_MATRIX)
    assert default.weights.sum() == pytest.approx(1)
    assert np.all(default.weights > 0)
    assert default.consistency_ratio < 0.1
    inconsistent = ahp_weights([[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]])
    assert inconsistent.consistency_ratio > 0.1
    assert not inconsistent.is_consistent


@pytest.mark.parametrize("matrix", [
    [], [[1, 2, 3]], [[2]], [[1, 2], [1 / 3, 1]],
    [[1, 0], [0, 1]], [[1, -2], [-1 / 2, 1]],
    [[1, np.nan], [np.nan, 1]], [[1, np.inf], [0, 1]],
    [[1, 1.2], [1 / 1.2, 1]], [["invalid"]], np.ones((16, 16)),
])
def test_invalid_comparison_matrices_are_rejected(matrix) -> None:
    with pytest.raises(ValueError):
        ahp_weights(matrix)


def test_manual_ahp_topsis_and_pareto_example() -> None:
    original = properties()
    ranked = rank_properties(original, [0.5, 0.3, 0.2])
    np.testing.assert_allclose(ranked["price_rub_priority"], [0.4, 0.2, 0.4])
    np.testing.assert_allclose(ranked["district_score_priority"], [0.25, 0.25, 0.5])
    np.testing.assert_allclose(ranked["travel_minutes_priority"], [0.4, 0.2, 0.4])
    np.testing.assert_allclose(ranked["ahp_score"], [0.355, 0.215, 0.43])
    np.testing.assert_allclose(ranked["topsis_score"], [sqrt(29) / (sqrt(29) + 3), 0, 1])
    assert ranked["pareto"].tolist() == [False, False, True]
    assert ranked["ahp_rank"].tolist() == [2, 3, 1]
    assert ranked["topsis_rank"].tolist() == [2, 3, 1]
    assert ranked["source_url"].tolist() == original["source_url"].tolist()
    assert "ahp_score" not in original.columns


def test_alternative_priorities_match_ratio_matrix_eigenvectors() -> None:
    frame = properties()
    ranked = rank_properties(frame, [1, 1, 1])
    for criterion in DEFAULT_CRITERIA:
        values = frame[criterion.key].to_numpy()
        ratios = values if criterion.direction == "max" else 1 / values
        matrix = ratios[:, None] / ratios[None, :]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvector = np.abs(eigenvectors[:, np.argmax(eigenvalues.real)].real)
        np.testing.assert_allclose(ranked[f"{criterion.key}_priority"], eigenvector / eigenvector.sum())


@pytest.mark.parametrize("count", [1, 2, 4])
def test_identical_alternatives_and_constant_columns_are_neutral(count) -> None:
    frame = pd.concat([properties().iloc[[0]]] * count, ignore_index=True)
    ranked = rank_properties(frame, [0.5, 0.3, 0.2])
    np.testing.assert_allclose(ranked["ahp_score"], np.full(count, 1 / count))
    np.testing.assert_allclose(ranked["topsis_score"], np.full(count, 0.5))
    assert ranked["pareto"].all()
    assert ranked["ahp_rank"].eq(1).all()
    assert ranked["topsis_rank"].eq(1).all()


def test_constant_criterion_and_zero_weight_do_not_produce_nan() -> None:
    frame = properties()
    frame["district_score"] = 5
    ranked = rank_properties(frame, [1, 0, 0])
    assert np.isfinite(ranked[["ahp_score", "topsis_score"]]).all().all()
    assert ranked["ahp_rank"].tolist() == [1, 2, 1]
    assert ranked["topsis_rank"].tolist() == [1, 2, 1]


def test_pareto_keeps_duplicates_and_tradeoffs_even_with_zero_weights() -> None:
    frame = pd.DataFrame({
        "price_rub": [100, 100, 200, 300],
        "district_score": [5, 5, 10, 4],
        "travel_minutes": [10, 10, 5, 20],
    }, index=[7, 7, 3, 1])
    ranked = rank_properties(frame, [1, 0, 0])
    assert ranked.index.tolist() == [7, 7, 3, 1]
    assert ranked["pareto"].tolist() == [True, True, True, False]
    assert len(ranked) == len(frame)
    assert ranked["ahp_rank"].tolist() == [1, 1, 2, 3]


def test_weight_scale_and_column_units_do_not_change_scores() -> None:
    frame = properties()
    expected = rank_properties(frame, [0.5, 0.3, 0.2])
    frame["price_rub"] *= 1e9
    frame["travel_minutes"] *= 60
    actual = rank_properties(frame, [5, 3, 2])
    np.testing.assert_allclose(actual["ahp_score"], expected["ahp_score"])
    np.testing.assert_allclose(actual["topsis_score"], expected["topsis_score"])


@pytest.mark.parametrize("weights", [[0, 0, 0], [-1, 1, 1], [1, 2], [1, 2, np.nan], [[1, 2, 3]]])
def test_invalid_weights_are_rejected(weights) -> None:
    with pytest.raises(ValueError):
        rank_properties(properties(), weights)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, "invalid"])
def test_incomplete_or_nonpositive_criterion_is_rejected(value) -> None:
    frame = properties().astype(object)
    frame.loc[0, "travel_minutes"] = value
    with pytest.raises(ValueError):
        rank_properties(frame, [1, 1, 1])


def test_empty_filtered_sample_has_complete_empty_result_columns() -> None:
    result = rank_properties(properties().iloc[:0], [1, 1, 1])
    assert result.empty
    assert {"ahp_score", "topsis_score", "pareto", "ahp_rank", "topsis_rank"}.issubset(result.columns)
