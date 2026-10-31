"""Use Streamlit's official in-process testing API, without a browser subprocess."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import housing_dss.data as housing_data
from housing_dss.data import load_properties


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app():
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


@pytest.fixture(scope="module")
def eligible_properties():
    source = load_properties()
    mask = (
        source["price_rub"].le(50_000_000)
        & source["travel_minutes"].le(60)
        & source[["metro", "travel_minutes", "transport_mode"]].notna().all(axis=1)
    )
    return source.loc[mask].copy()


def test_initial_screen_shows_filtered_sample_and_mortgage(app, eligible_properties):
    assert not app.exception
    assert app.number_input(key="rate").value == 14.0
    assert app.metric[0].value == str(len(eligible_properties))
    assert len(app.selectbox(key="mortgage_listing").options) == len(eligible_properties)


@pytest.mark.parametrize("property_type", ["Квартира", "Дом"])
def test_property_type_switch_changes_results_without_errors(app, eligible_properties, property_type):
    app.radio(key="property_type").set_value(property_type).run()
    assert not app.exception
    expected = eligible_properties["property_type"].eq(property_type).sum()
    assert app.metric[0].value == str(expected)


def test_empty_filter_does_not_crash_or_show_a_stale_mortgage(app):
    app.number_input(key="max_price").set_value(1.0).run()
    assert not app.exception
    assert not app.selectbox
    assert any("объектов нет" in item.value for item in app.warning)


def test_single_property_has_both_rankings_and_pareto(app, eligible_properties, monkeypatch):
    monkeypatch.setattr(
        housing_data, "load_properties",
        lambda *_args, **_kwargs: eligible_properties.head(1).copy(),
    )
    app.run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.metric[1].value == "1"


def test_inconsistent_judgments_show_a_clear_warning(app):
    app.select_slider(key="pair_0_1").set_value(9).run()
    app.select_slider(key="pair_0_2").set_value(1 / 9).run()
    app.select_slider(key="pair_1_2").set_value(9).run()
    assert not app.exception
    assert any("CR > 0,10" in item.value for item in app.warning)


@pytest.mark.parametrize("mode,code", [("Пешком", "walking"), ("На транспорте", "transport")])
def test_transport_mode_keeps_matching_objects(app, eligible_properties, mode, code):
    app.radio(key="transport").set_value(mode).run()
    assert not app.exception
    expected = eligible_properties["transport_mode"].eq(code).sum()
    assert app.metric[0].value == str(expected)


def test_all_cash_and_zero_rate_are_valid_user_scenarios(app):
    app.slider(key="deposit").set_value(100).run()
    app.number_input(key="rate").set_value(0.0).run()
    assert not app.exception
    assert any(metric.label == "Сумма кредита" and metric.value == "0 ₽" for metric in app.metric)
