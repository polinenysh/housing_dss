from decimal import Decimal

import numpy as np
import pytest

from housing_dss.mortgage import calculate_mortgage


@pytest.mark.parametrize("scheme", ["annuity", "differentiated"])
def test_one_month_manual_case(scheme) -> None:
    result = calculate_mortgage(120000, 12, 1, scheme)
    assert result.first_payment == Decimal("121200.00")
    assert result.last_payment == result.first_payment
    assert result.total_interest == Decimal("1200.00")
    assert result.total_paid == Decimal("121200.00")
    assert result.schedule.iloc[-1]["balance"] == 0


def test_differentiated_manual_case() -> None:
    result = calculate_mortgage(120000, 12, 12, "differentiated")
    assert result.first_payment == Decimal("11200.00")
    assert result.last_payment == Decimal("10100.00")
    assert result.total_interest == Decimal("7800.00")
    assert result.total_paid == Decimal("127800.00")
    np.testing.assert_allclose(result.schedule["principal"], 10000)
    np.testing.assert_allclose(result.schedule["interest"], np.arange(1200, 0, -100))


def test_annuity_payment_matches_closed_form_and_last_payment_correction() -> None:
    principal, rate, months = 120000, 0.01, 12
    theoretical = principal * rate / (1 - (1 + rate) ** (-months))
    result = calculate_mortgage(principal, 12, months)
    assert float(result.first_payment) == pytest.approx(theoretical, abs=0.005)
    assert result.first_payment == Decimal("10661.85")
    assert result.schedule.iloc[:-1]["payment"].eq(float(result.first_payment)).all()
    assert result.schedule.iloc[-1]["balance"] == 0
    assert abs(result.last_payment - result.first_payment) < Decimal("0.10")


@pytest.mark.parametrize("scheme", ["annuity", "differentiated"])
def test_zero_interest_and_rounding_of_last_payment(scheme) -> None:
    result = calculate_mortgage(100, 0, 3, scheme)
    assert result.total_interest == Decimal("0.00")
    assert result.total_paid == Decimal("100.00")
    assert result.schedule["payment"].sum() == pytest.approx(100)
    assert result.schedule.iloc[-1]["balance"] == 0
    assert result.first_payment == Decimal("33.33")
    if scheme == "annuity":
        assert result.last_payment == Decimal("33.34")


@pytest.mark.parametrize("scheme", ["annuity", "differentiated"])
def test_zero_loan_produces_zero_schedule(scheme) -> None:
    result = calculate_mortgage(0, 12, 24, scheme)
    assert len(result.schedule) == 24
    assert result.schedule.drop(columns="month").eq(0).all().all()
    assert result.total_paid == result.total_interest == Decimal("0.00")
    assert result.first_payment == result.last_payment == Decimal("0.00")


@pytest.mark.parametrize("scheme", ["annuity", "differentiated"])
@pytest.mark.parametrize(("principal", "annual_rate", "months"), [
    (10000000, 18.5, 240), (Decimal("12345.67"), Decimal("7.95"), 37),
    (Decimal("0.01"), 0, 12), (1, 0.001, 1200),
])
def test_money_conservation_and_schedule_invariants(principal, annual_rate, months, scheme) -> None:
    result = calculate_mortgage(principal, annual_rate, months, scheme)
    schedule = result.schedule
    rounded_principal = Decimal(str(principal)).quantize(Decimal("0.01"))
    assert len(schedule) == months
    assert schedule["month"].tolist() == list(range(1, months + 1))
    assert schedule[["payment", "principal", "interest", "balance"]].ge(0).all().all()
    assert schedule["balance"].is_monotonic_decreasing
    assert schedule.iloc[-1]["balance"] == 0
    np.testing.assert_allclose(schedule["payment"], schedule["principal"] + schedule["interest"], atol=1e-8)
    assert schedule["principal"].sum() == pytest.approx(float(rounded_principal))
    assert result.total_paid == rounded_principal + result.total_interest
    assert schedule["interest"].sum() == pytest.approx(float(result.total_interest))
    assert schedule["payment"].sum() == pytest.approx(float(result.total_paid))
    for column in ("payment", "principal", "interest", "balance"):
        np.testing.assert_allclose(schedule[column] * 100, np.round(schedule[column] * 100), atol=1e-5)


def test_differentiated_interest_is_lower_for_the_same_loan() -> None:
    annuity = calculate_mortgage(10000000, 18, 240, "annuity")
    differentiated = calculate_mortgage(10000000, 18, 240, "differentiated")
    assert differentiated.total_interest < annuity.total_interest
    assert differentiated.first_payment > annuity.first_payment
    assert differentiated.last_payment < annuity.last_payment


def test_tiny_positive_rate_remains_numerically_stable() -> None:
    result = calculate_mortgage(100, Decimal("1e-100"), 3)
    assert result.total_interest == Decimal("0.00")
    assert result.total_paid == Decimal("100.00")
    assert result.schedule.iloc[-1]["balance"] == 0


@pytest.mark.parametrize(("principal", "rate", "months", "scheme"), [
    (-1, 10, 12, "annuity"), (1, -1, 12, "annuity"),
    (np.nan, 10, 12, "annuity"), (1, np.inf, 12, "annuity"),
    (True, 10, 12, "annuity"), (1, False, 12, "annuity"),
    (1, 10, 0, "annuity"), (1, 10, 1201, "annuity"),
    (1, 10, 1.5, "annuity"), (1, 10, True, "annuity"),
    (1, 10, 12, "unknown"), ("invalid", 10, 12, "annuity"),
])
def test_invalid_inputs_are_rejected(principal, rate, months, scheme) -> None:
    with pytest.raises(ValueError):
        calculate_mortgage(principal, rate, months, scheme)
