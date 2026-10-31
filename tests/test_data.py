"""Tests for importing real listings and safely exporting user-provided text."""

import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from housing_dss.data import (
    DATA_PATH,
    MAX_ROWS,
    REQUIRED_COLUMNS,
    export_csv,
    load_properties,
    validate_properties,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "listing_id": "123456789", "property_type": "Квартира",
        "price_rub": 18000000, "area_m2": 55.5, "rooms": 2,
        "location": "Москва, улица Примерная, 1", "district": "Сокольники",
        "metro": "Сокольники", "travel_minutes": 7, "transport_mode": "walking",
        "source_url": "https://www.cian.ru/sale/flat/123456789/",
        "observed_at": "2026-10-28", "verification_level": "indexed_catalog",
        "evidence_url": "https://www.cian.ru/kupit-prodat-kvartiru/",
        "evidence_note": "Контрольная строка теста, не объявление выборки.",
    }])


def test_normalization_does_not_modify_input():
    frame = sample_frame()
    frame["rooms"] = "2"
    original = frame.copy(deep=True)
    result = validate_properties(frame)
    assert_frame_equal(frame, original)
    assert str(result["rooms"].dtype) == "Int64"
    assert result.at[0, "price_rub"] == 18000000
    assert result.at[0, "listing_id"] == "123456789"


@pytest.mark.parametrize("field,value", [
    ("listing_id", "abc"), ("listing_id", "123.0"),
    ("property_type", "Апартаменты"), ("price_rub", 0),
    ("price_rub", -1), ("price_rub", np.inf), ("price_rub", "не число"),
    ("price_rub", np.nan), ("area_m2", 0), ("area_m2", "-inf"),
    ("rooms", -1), ("rooms", 1.5), ("rooms", "три"), ("rooms", np.inf),
    ("rooms", "9999999999999999999999999"),
    ("location", "Московская область, Мытищи"), ("location", ""),
    ("district", "  "), ("metro", ""), ("travel_minutes", 0),
    ("travel_minutes", np.inf), ("transport_mode", "car"),
    ("source_url", "https://www.cian.ru/sale/flat/123456788/"),
    ("source_url", "https://www.cian.ru/sale/suburban/123456789/"),
    ("source_url", "http://www.cian.ru/sale/flat/123456789/"),
    ("source_url", "https://www.cian.ru.evil.test/sale/flat/123456789/"),
    ("source_url", "https://www.cian.ru/sale/flat/123456789/?x=1"),
    ("observed_at", "2026-02-30"), ("observed_at", "2026-1-1"),
    ("verification_level", "verified"),
    ("evidence_url", "http://www.cian.ru/"),
    ("evidence_url", "https://www.cian.ru.evil.test/"),
    ("evidence_url", "https://user@www.cian.ru/"),
    ("evidence_note", "x" * 5001),
])
def test_invalid_fields_are_rejected(field, value):
    frame = sample_frame()
    frame[field] = value
    with pytest.raises(ValueError):
        validate_properties(frame)


@pytest.mark.parametrize("field", ["listing_id", "district", "source_url", "evidence_url"])
def test_required_columns_are_checked(field):
    with pytest.raises(ValueError, match="столбцы"):
        validate_properties(sample_frame().drop(columns=field))


def test_duplicate_ids_and_excess_rows_are_rejected():
    with pytest.raises(ValueError, match="повторяться"):
        validate_properties(pd.concat([sample_frame()] * 2, ignore_index=True))
    with pytest.raises(ValueError, match="1000"):
        validate_properties(pd.concat([sample_frame()] * (MAX_ROWS + 1), ignore_index=True))


@pytest.mark.parametrize("rooms", [None, "", 0])
def test_optional_rooms_and_transport(rooms):
    frame = sample_frame()
    frame["rooms"] = rooms
    frame[["metro", "travel_minutes", "transport_mode"]] = ""
    result = validate_properties(frame)
    assert result["travel_minutes"].isna().all()
    if rooms == 0:
        assert result.at[0, "rooms"] == 0
    else:
        assert result["rooms"].isna().all()


def test_house_url_matches_house_type():
    frame = sample_frame()
    frame["property_type"] = "Дом"
    frame["source_url"] = "https://www.cian.ru/sale/suburban/123456789/"
    result = validate_properties(frame)
    assert result.at[0, "property_type"] == "Дом"


def test_utf8_bom_import_preserves_leading_zero_ids():
    frame = sample_frame()
    frame["listing_id"] = "00123456789"
    frame["source_url"] = "https://www.cian.ru/sale/flat/00123456789/"
    contents = frame.to_csv(index=False).encode("utf-8-sig")
    result = load_properties(BytesIO(contents))
    assert result.at[0, "listing_id"] == "00123456789"
    assert result.at[0, "district"] == "Сокольники"


def test_export_escapes_text_formulas_and_preserves_numbers():
    frame = pd.DataFrame({
        "text": ["=1+1", "+SUM(A1)", "-cmd", "@formula", "  =SUM(A1)", "Москва"],
        "number": [-3, 1, 2, 3, 4, 5],
    })
    original = frame.copy(deep=True)
    encoded = export_csv(frame)
    assert encoded.startswith(b"\xef\xbb\xbf")
    exported = pd.read_csv(BytesIO(encoded), keep_default_na=False)
    assert exported["text"].tolist() == [
        "'=1+1", "'+SUM(A1)", "'-cmd", "'@formula", "'  =SUM(A1)", "Москва",
    ]
    assert exported["number"].tolist() == [-3, 1, 2, 3, 4, 5]
    assert_frame_equal(frame, original)


def test_export_escapes_formula_header():
    exported = pd.read_csv(BytesIO(export_csv(pd.DataFrame({"=formula": [1]}))))
    assert exported.columns.tolist() == ["'=formula"]


def test_import_does_not_silently_truncate_nul_in_number():
    frame = sample_frame()
    frame["price_rub"] = "123\x00garbage"
    contents = frame.to_csv(index=False).encode("utf-8-sig")
    with pytest.raises(ValueError, match="price_rub"):
        load_properties(BytesIO(contents))


def test_url_import_is_rejected_without_fetching():
    with pytest.raises(ValueError, match="URL"):
        load_properties("https://www.cian.ru/sample.csv")


def test_prepared_real_csv_satisfies_contract():
    result = load_properties(DATA_PATH)
    assert len(result) >= 100
    assert set(result["property_type"]) == {"Квартира", "Дом"}
    assert result["listing_id"].is_unique
    assert result["travel_minutes"].notna().any()
    metadata = json.loads(DATA_PATH.with_name("source_observations.json").read_text(encoding="utf-8"))
    assert metadata["record_count"] == len(result)
    assert metadata["observation_dates"] == sorted(result["observed_at"].unique())
    assert metadata["observed_at"] == max(metadata["observation_dates"])
    observations = validate_properties(pd.DataFrame(metadata["observations"]))
    assert_frame_equal(
        result.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True),
        observations.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True),
    )
