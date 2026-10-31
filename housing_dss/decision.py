"""Explainable multicriteria ranking for a prepared housing sample.

Alternative priorities in AHP come from consistent ratio matrices: positive
values for benefit criteria and their reciprocals for cost criteria. Their
principal eigenvectors are exactly the normalized values used below.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    direction: str


DEFAULT_CRITERIA = (
    Criterion("price_rub", "Цена", "min"),
    Criterion("district_score", "Предпочтение района", "max"),
    Criterion("travel_minutes", "Время в пути", "min"),
)
DEFAULT_MATRIX = np.array(
    [[1.0, 3.0, 5.0], [1 / 3, 1.0, 2.0], [1 / 5, 1 / 2, 1.0]],
    dtype=float,
)
RANDOM_INDEX = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}
CONSISTENCY_THRESHOLD = 0.10


@dataclass(frozen=True)
class AHPResult:
    weights: np.ndarray
    lambda_max: float
    consistency_index: float
    consistency_ratio: float

    @property
    def is_consistent(self) -> bool:
        return self.consistency_ratio <= CONSISTENCY_THRESHOLD


def ahp_weights(matrix: Sequence[Sequence[float]] | np.ndarray) -> AHPResult:
    """Validate a Saaty matrix and compute its principal eigenvector.

    Only 1..9 and their reciprocals are accepted. CI and CR are zero for
    dimensions one and two; the tabulated random index supports sizes 1..15.
    """
    try:
        values = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Матрица сравнений должна содержать числа.") from error
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Матрица сравнений должна быть квадратной.")
    size = values.shape[0]
    if size not in RANDOM_INDEX:
        raise ValueError("Размер матрицы должен быть от 1 до 15.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Матрица должна содержать положительные конечные числа.")
    if not np.allclose(np.diag(values), 1.0, rtol=0, atol=1e-9):
        raise ValueError("На диагонали матрицы должны стоять единицы.")
    if not np.allclose(values * values.T, 1.0, rtol=1e-8, atol=1e-9):
        raise ValueError("Парные оценки должны быть взаимно обратными.")
    scale = np.concatenate((np.arange(1, 10, dtype=float), 1 / np.arange(2, 10)))
    if not np.isclose(values[..., None], scale, rtol=1e-8, atol=1e-9).any(axis=-1).all():
        raise ValueError("Используйте шкалу Саати: 1–9 и обратные значения.")
    eigenvalues, eigenvectors = np.linalg.eig(values)
    position = int(np.argmax(eigenvalues.real))
    principal = np.abs(eigenvectors[:, position].real)
    weights = principal / principal.sum()
    lambda_max = float(eigenvalues[position].real)
    index = max(0.0, (lambda_max - size) / (size - 1)) if size > 2 else 0.0
    ratio = index / RANDOM_INDEX[size] if size > 2 else 0.0
    return AHPResult(weights, lambda_max, index, ratio)


def _normalized_weights(weights: Sequence[float], size: int) -> np.ndarray:
    try:
        values = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Веса должны быть числами.") from error
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError("Количество конечных весов должно совпадать с критериями.")
    if (values < 0).any() or not (values > 0).any():
        raise ValueError("Веса должны быть неотрицательными и иметь положительную сумму.")
    scaled = values / values.max()
    return scaled / scaled.sum()


def _dense_ranks(scores: np.ndarray) -> np.ndarray:
    """Give mathematically equal scores the same rank, preserving input order."""
    ranks = np.empty(len(scores), dtype=int)
    previous = None
    current_rank = 0
    for position in np.argsort(-scores, kind="stable"):
        value = scores[position]
        if previous is None or not np.isclose(value, previous, rtol=1e-10, atol=1e-12):
            current_rank += 1
            previous = value
        ranks[position] = current_rank
    return ranks


def rank_properties(
    frame: pd.DataFrame,
    weights: Sequence[float],
    criteria: Sequence[Criterion] = DEFAULT_CRITERIA,
) -> pd.DataFrame:
    """Add AHP, TOPSIS and Pareto results without sorting or dropping rows.

    All criterion values must be positive and complete. Cost directions are
    minimized and benefit directions maximized. A TOPSIS tie with no distance
    to either ideal receives 0.5, including one alternative. Ranks are dense.
    Pareto dominance uses every original criterion, independently of weights.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Ожидается таблица pandas DataFrame.")
    criteria = tuple(criteria)
    keys = [criterion.key for criterion in criteria]
    if not criteria or len(set(keys)) != len(keys):
        raise ValueError("Нужен непустой список критериев с уникальными названиями.")
    if any(criterion.direction not in {"min", "max"} for criterion in criteria):
        raise ValueError("Направление критерия должно быть min или max.")
    if frame.columns.duplicated().any() or not set(keys).issubset(frame.columns):
        raise ValueError("Таблица должна содержать все критерии без повторяющихся столбцов.")
    normalized_weights = _normalized_weights(weights, len(criteria))
    try:
        values = frame[keys].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError("Критерии должны содержать положительные числа.") from error
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Критерии должны содержать положительные конечные числа без пропусков.")
    result = frame.copy(deep=True)
    if not len(values):
        for key in keys:
            result[f"{key}_priority"] = pd.Series(index=result.index, dtype=float)
        for key in ("ahp_score", "topsis_score"):
            result[key] = pd.Series(index=result.index, dtype=float)
        result["pareto"] = pd.Series(index=result.index, dtype=bool)
        for key in ("ahp_rank", "topsis_rank"):
            result[key] = pd.Series(index=result.index, dtype=int)
        return result
    benefits = np.array([criterion.direction == "max" for criterion in criteria])
    # Column scaling prevents overflow without changing any of the rankings.
    scaled = values / values.max(axis=0)
    ratios = np.where(benefits, scaled, values.min(axis=0) / values)
    priorities = ratios / ratios.sum(axis=0)
    ahp_scores = priorities @ normalized_weights
    weighted = scaled / np.linalg.norm(scaled, axis=0) * normalized_weights
    positive_ideal = np.where(benefits, weighted.max(axis=0), weighted.min(axis=0))
    negative_ideal = np.where(benefits, weighted.min(axis=0), weighted.max(axis=0))
    positive_distance = np.linalg.norm(weighted - positive_ideal, axis=1)
    negative_distance = np.linalg.norm(weighted - negative_ideal, axis=1)
    total_distance = positive_distance + negative_distance
    topsis_scores = np.divide(
        negative_distance, total_distance,
        out=np.full(len(values), 0.5), where=total_distance > 0,
    )
    oriented = np.where(benefits, values, -values)
    pareto = np.ones(len(values), dtype=bool)
    for position, candidate in enumerate(oriented):
        dominates = np.all(oriented >= candidate, axis=1) & np.any(oriented > candidate, axis=1)
        pareto[position] = not dominates.any()
    for index, key in enumerate(keys):
        result[f"{key}_priority"] = priorities[:, index]
    result["ahp_score"] = ahp_scores
    result["topsis_score"] = topsis_scores
    result["pareto"] = pareto
    result["ahp_rank"] = _dense_ranks(ahp_scores)
    result["topsis_rank"] = _dense_ranks(topsis_scores)
    return result
