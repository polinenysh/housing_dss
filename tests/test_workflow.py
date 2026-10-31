"""Integration checks against the real prepared sample."""

import pytest

from housing_dss.data import load_properties
from housing_dss.decision import DEFAULT_MATRIX, ahp_weights, rank_properties
from housing_dss.workflow import add_mortgage_summaries, filter_properties


@pytest.mark.parametrize("property_type", ["Квартира и дом", "Квартира", "Дом"])
def test_type_filter_keeps_the_requested_objects(property_type):
    source = load_properties()
    expected_mask = (
        source["price_rub"].le(50_000_000)
        & source["travel_minutes"].le(60)
        & source[["metro", "travel_minutes", "transport_mode"]].notna().all(axis=1)
    )
    if property_type != "Квартира и дом":
        expected_mask &= source["property_type"].eq(property_type)
    selected = filter_properties(source, property_type=property_type)
    assert selected["listing_id"].tolist() == source.loc[expected_mask, "listing_id"].tolist()


@pytest.mark.parametrize("mode", ["walking", "transport"])
def test_transport_filter_and_budget_are_applied_together(mode):
    selected = filter_properties(load_properties(), max_price=25_000_000, min_area=35, transport_mode=mode)
    assert not selected.empty
    assert selected["transport_mode"].eq(mode).all()
    assert selected["price_rub"].le(25_000_000).all()
    assert selected["area_m2"].ge(35).all()


def test_empty_result_is_a_normal_filter_outcome():
    assert filter_properties(load_properties(), max_price=1).empty


def test_missing_transport_is_excluded_without_an_invented_value():
    source = load_properties()
    baseline = filter_properties(source)
    removed = baseline.index[0]
    source.loc[removed, ["metro", "travel_minutes", "transport_mode"]] = None
    expected_ids = baseline.drop(index=removed)["listing_id"].tolist()
    assert filter_properties(source)["listing_id"].tolist() == expected_ids


@pytest.mark.parametrize("deposit", [0, 20, 100])
def test_complete_pipeline_preserves_ids_and_purchase_cost_identity(deposit):
    source = filter_properties(load_properties())
    source["district_score"] = 5
    ranked = rank_properties(source, ahp_weights(DEFAULT_MATRIX).weights)
    result = add_mortgage_summaries(ranked, deposit, 12, 240)
    assert result["listing_id"].tolist() == source["listing_id"].tolist()
    assert result["total_purchase_annuity"].to_numpy() == pytest.approx(
        result["price_rub"].to_numpy() + result["annuity_interest"].to_numpy(), abs=0.01,
    )
    assert result["total_purchase_differentiated"].to_numpy() == pytest.approx(
        result["price_rub"].to_numpy() + result["differentiated_interest"].to_numpy(), abs=0.01,
    )
    if deposit == 100:
        assert result["loan_rub"].eq(0).all()
        assert result["annuity_interest"].eq(0).all()


def test_zero_interest_has_no_overpayment_in_either_scheme():
    result = add_mortgage_summaries(load_properties().head(1), 20, 0, 12)
    assert result["annuity_interest"].eq(0).all()
    assert result["differentiated_interest"].eq(0).all()


@pytest.mark.parametrize("deposit", [-1, 101, float("nan")])
def test_invalid_deposit_is_rejected(deposit):
    with pytest.raises(ValueError):
        add_mortgage_summaries(load_properties().head(1), deposit, 12, 12)
