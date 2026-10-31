"""Fixed-rate monthly mortgage schedules with reproducible kopeck rounding.

The model includes principal and interest only: no insurance, fees, subsidies,
early repayment or variable rates. annual_rate is expressed in percent.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from numbers import Integral

import pandas as pd


CENT = Decimal("0.01")
ZERO = Decimal("0.00")
MAX_MONTHS = 1200
SCHEMES = ("annuity", "differentiated")


@dataclass(frozen=True)
class MortgageResult:
    schedule: pd.DataFrame
    total_interest: Decimal
    total_paid: Decimal
    first_payment: Decimal
    last_payment: Decimal


def _decimal_nonnegative(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name}: ожидается конечное неотрицательное число.")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name}: ожидается число.") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name}: ожидается конечное неотрицательное число.")
    return number


def calculate_mortgage(
    principal: Decimal | float | int,
    annual_rate: Decimal | float | int,
    months: int,
    scheme: str = "annuity",
) -> MortgageResult:
    """Calculate a schedule for a loan amount in rubles and annual percent.

    Monthly rate = annual_rate / 100 / 12. Money is rounded half-up to kopecks
    at every payment. The last payment repays the actual remaining balance.
    A zero loan returns a zero schedule; a zero rate works for both schemes.
    The supported term is 1..1200 months.
    """
    loan = _decimal_nonnegative(principal, "Сумма кредита")
    rate_percent = _decimal_nonnegative(annual_rate, "Годовая ставка")
    if isinstance(months, bool) or not isinstance(months, Integral) or not 1 <= months <= MAX_MONTHS:
        raise ValueError(f"Срок должен быть целым числом от 1 до {MAX_MONTHS} месяцев.")
    months = int(months)
    if scheme not in SCHEMES:
        raise ValueError("Схема должна быть annuity или differentiated.")
    with localcontext() as context:
        # Retain a very small positive rate in (1 + r) and enough integer
        # digits to round a large principal to kopecks without loss.
        context.prec = max(
            40,
            len(loan.as_tuple().digits) + max(0, loan.adjusted()) + 20,
            max(0, -rate_percent.adjusted()) + 20,
        )
        loan = loan.quantize(CENT, rounding=ROUND_HALF_UP)
        monthly_rate = rate_percent / Decimal(1200)
        if monthly_rate == 0:
            annuity_payment = (loan / months).quantize(CENT, rounding=ROUND_HALF_UP)
        else:
            growth = (1 + monthly_rate) ** months
            annuity_payment = (loan * monthly_rate * growth / (growth - 1)).quantize(
                CENT, rounding=ROUND_HALF_UP,
            )
        balance = loan
        total_interest = ZERO
        total_paid = ZERO
        payments = []
        rows = []
        for month in range(1, months + 1):
            interest = (balance * monthly_rate).quantize(CENT, rounding=ROUND_HALF_UP)
            if month == months:
                principal_payment = balance
            elif scheme == "differentiated":
                repaid_target = (loan * month / months).quantize(CENT, rounding=ROUND_HALF_UP)
                principal_payment = min(balance, repaid_target - (loan - balance))
            else:
                principal_payment = min(balance, max(ZERO, annuity_payment - interest))
            payment = principal_payment + interest
            balance -= principal_payment
            total_interest += interest
            total_paid += payment
            payments.append(payment)
            rows.append({
                "month": month,
                "payment": float(payment),
                "principal": float(principal_payment),
                "interest": float(interest),
                "balance": float(balance),
            })
    return MortgageResult(
        schedule=pd.DataFrame(rows),
        total_interest=total_interest,
        total_paid=total_paid,
        first_payment=payments[0],
        last_payment=payments[-1],
    )
