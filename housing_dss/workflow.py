"""Shared filtering and mortgage summaries used by the UI and integration tests."""

from decimal import Decimal, ROUND_HALF_UP
from math import isfinite
from typing import Sequence

import pandas as pd

from housing_dss.mortgage import calculate_mortgage


TYPE_OPTIONS = ("Квартира и дом", "Квартира", "Дом")
MODE_OPTIONS = {"Любой способ": None, "Пешком": "walking", "На транспорте": "transport"}
SUMMARY_COLUMNS = (
    "downpayment_rub", "loan_rub", "annuity_payment", "annuity_interest",
    "differentiated_first_payment", "differentiated_last_payment",
    "differentiated_interest", "total_purchase_annuity", "total_purchase_differentiated",
)


def filter_properties(
    frame: pd.DataFrame,
    property_type: str = TYPE_OPTIONS[0],
    max_price: float = 50_000_000,
    min_area: float = 0,
    districts: Sequence[str] = (),
    transport_mode: str | None = None,
    max_travel: float = 60,
) -> pd.DataFrame:
    """Select feasible alternatives with complete observed transport data."""
    if property_type not in TYPE_OPTIONS or transport_mode not in {None, "walking", "transport"}:
        raise ValueError("Неизвестный тип объекта или способ передвижения.")
    if not all(isfinite(value) for value in (max_price, min_area, max_travel)):
        raise ValueError("Параметры фильтра должны быть конечными числами.")
    if max_price <= 0 or min_area < 0 or max_travel <= 0:
        raise ValueError("Цена и время должны быть положительными, площадь — неотрицательной.")
    mask = (
        (frame["price_rub"] <= max_price)
        & (frame["area_m2"] >= min_area)
        & frame["travel_minutes"].notna()
        & (frame["travel_minutes"] <= max_travel)
    )
    if property_type != TYPE_OPTIONS[0]:
        mask &= frame["property_type"].eq(property_type)
    if districts:
        mask &= frame["district"].isin(districts)
    if transport_mode is not None:
        mask &= frame["transport_mode"].eq(transport_mode)
    return frame.loc[mask].copy()


def add_mortgage_summaries(
    frame: pd.DataFrame, downpayment_percent: float, annual_rate: float, months: int,
) -> pd.DataFrame:
    """Add both schemes using the same down payment fraction for each property."""
    if not isfinite(downpayment_percent) or not 0 <= downpayment_percent <= 100:
        raise ValueError("Первоначальный взнос должен быть от 0 до 100 процентов.")
    result = frame.copy()
    summaries = []
    for price in result["price_rub"]:
        amount = Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        deposit = (amount * Decimal(str(downpayment_percent)) / 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        loan = amount - deposit
        annuity = calculate_mortgage(loan, annual_rate, months, "annuity")
        differentiated = calculate_mortgage(loan, annual_rate, months, "differentiated")
        summaries.append([
            float(deposit), float(loan), float(annuity.first_payment),
            float(annuity.total_interest), float(differentiated.first_payment),
            float(differentiated.last_payment), float(differentiated.total_interest),
            float(deposit + annuity.total_paid), float(deposit + differentiated.total_paid),
        ])
    for index, column in enumerate(SUMMARY_COLUMNS):
        result[column] = [row[index] for row in summaries]
    return result
