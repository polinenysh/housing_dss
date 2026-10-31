"""Read and validate the project's prepared Cian sample without network access."""

from datetime import date
import re
from pathlib import Path
from typing import BinaryIO, TextIO
from urllib.parse import urlsplit

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cian_moscow.csv"
MAX_ROWS = 1000
MAX_TEXT_LENGTH = 5000
REQUIRED_COLUMNS = (
    "listing_id", "property_type", "price_rub", "area_m2", "rooms",
    "location", "district", "metro", "travel_minutes", "transport_mode",
    "source_url", "observed_at", "verification_level", "evidence_url",
    "evidence_note",
)
_TEXT_COLUMNS = (
    "listing_id", "property_type", "location", "district", "metro",
    "transport_mode", "source_url", "observed_at", "verification_level",
    "evidence_url", "evidence_note",
)
_OPTIONAL_TEXT_COLUMNS = {"metro", "transport_mode"}
_VERIFICATION_LEVELS = {"live_catalog", "indexed_catalog", "search_index"}


def load_properties(
    path_or_buffer: str | Path | BinaryIO | TextIO = DATA_PATH,
) -> pd.DataFrame:
    """Load at most 1000 rows; preserve IDs and missing text from UTF-8 CSV."""
    if isinstance(path_or_buffer, str) and "://" in path_or_buffer:
        raise ValueError("Загрузка по URL не поддерживается: выберите локальный CSV.")
    frame = pd.read_csv(
        path_or_buffer,
        encoding="utf-8-sig",
        dtype="string",
        engine="python",
        keep_default_na=False,
        nrows=MAX_ROWS + 1,
    )
    return validate_properties(frame)


def _numeric_column(
    frame: pd.DataFrame, column: str, *, nullable: bool = False,
) -> pd.Series:
    original = frame[column].replace(r"^\s*$", pd.NA, regex=True)
    numeric = pd.to_numeric(original, errors="coerce").astype("Float64")
    invalid = original.notna() & numeric.isna()
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    invalid |= numeric.notna() & ~finite
    invalid |= numeric.notna() & (numeric <= 0)
    if not nullable:
        invalid |= numeric.isna()
    if invalid.any():
        raise ValueError(f"Поле {column}: ожидается положительное конечное число.")
    return numeric


def _valid_evidence_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and parsed.netloc in {"cian.ru", "www.cian.ru"}
            and not any(character.isspace() for character in value)
        )
    except ValueError:
        return False


def validate_properties(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy, or raise ValueError for an invalid sample."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Ожидается таблица pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Названия столбцов не должны повторяться.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(missing)}.")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"В выборке должно быть от 1 до {MAX_ROWS} объектов.")
    result = frame.copy(deep=True)
    for column in result.columns:
        if result[column].map(
            lambda value: isinstance(value, str) and len(value) > MAX_TEXT_LENGTH
        ).any():
            raise ValueError(f"Поле {column} длиннее {MAX_TEXT_LENGTH} символов.")
    for column in _TEXT_COLUMNS:
        result[column] = result[column].astype("string").str.strip()
        result[column] = result[column].replace("", pd.NA)
        if column not in _OPTIONAL_TEXT_COLUMNS and result[column].isna().any():
            raise ValueError(f"Поле {column} обязательно для каждого объекта.")
    if not result["listing_id"].str.fullmatch(r"[0-9]+").all():
        raise ValueError("listing_id должен содержать только цифры.")
    if result["listing_id"].duplicated().any():
        raise ValueError("listing_id не должен повторяться.")
    if not result["property_type"].isin({"Квартира", "Дом"}).all():
        raise ValueError("property_type: разрешены только Квартира и Дом.")
    if not result["location"].str.startswith("Москва").all():
        raise ValueError("location должен начинаться с Москва.")
    for column in ("price_rub", "area_m2"):
        result[column] = _numeric_column(result, column).astype(float)
    result["travel_minutes"] = _numeric_column(result, "travel_minutes", nullable=True)
    rooms = result["rooms"].replace(r"^\s*$", pd.NA, regex=True)
    room_numbers = pd.to_numeric(rooms, errors="coerce").astype("Float64")
    invalid_rooms = rooms.notna() & room_numbers.isna()
    finite_rooms = np.isfinite(room_numbers.to_numpy(dtype=float, na_value=np.nan))
    invalid_rooms |= room_numbers.notna() & (
        ~finite_rooms | (room_numbers < 0) | (room_numbers % 1 != 0)
        | (room_numbers >= 2**63)
    )
    if invalid_rooms.any():
        raise ValueError("rooms: ожидается целое неотрицательное число или пустое поле.")
    result["rooms"] = room_numbers.astype("Int64")
    transport = result[["metro", "travel_minutes", "transport_mode"]].notna()
    if not (transport.all(axis=1) | ~transport.any(axis=1)).all():
        raise ValueError("metro, travel_minutes и transport_mode заполните вместе или оставьте пустыми.")
    if not result["transport_mode"].dropna().isin({"walking", "transport"}).all():
        raise ValueError("transport_mode: разрешены walking, transport или пустое поле.")
    if not result["verification_level"].isin(_VERIFICATION_LEVELS).all():
        raise ValueError("Неизвестный verification_level.")
    for row in result.itertuples(index=False):
        category = "flat" if row.property_type == "Квартира" else "suburban"
        expected = f"https://www.cian.ru/sale/{category}/{row.listing_id}/"
        if row.source_url != expected:
            raise ValueError("source_url должен точно соответствовать типу и listing_id на Циан.")
        if not _valid_evidence_url(row.evidence_url):
            raise ValueError("evidence_url должен быть HTTPS-ссылкой на cian.ru.")
        try:
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", row.observed_at) is None:
                raise ValueError
            date.fromisoformat(row.observed_at)
        except ValueError as error:
            raise ValueError("observed_at: ожидается существующая дата YYYY-MM-DD.") from error
    return result


def _safe_csv_cell(value: object) -> object:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_csv(frame: pd.DataFrame) -> bytes:
    """Export UTF-8 BOM CSV; text formulas are escaped, numeric cells preserved."""
    safe = frame.copy(deep=True)
    for column in safe.columns:
        safe[column] = safe[column].map(_safe_csv_cell)
    safe.columns = [_safe_csv_cell(column) for column in safe.columns]
    return safe.to_csv(index=False).encode("utf-8-sig")
