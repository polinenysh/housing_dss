"""Run with: python -m streamlit run app.py."""

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from housing_dss.data import export_csv, load_properties
from housing_dss.decision import DEFAULT_CRITERIA, DEFAULT_MATRIX, ahp_weights, rank_properties
from housing_dss.mortgage import calculate_mortgage
from housing_dss.workflow import MODE_OPTIONS, TYPE_OPTIONS, add_mortgage_summaries, filter_properties


PROJECT_ROOT = Path(__file__).resolve().parent
CRITERION_LABELS = [criterion.label for criterion in DEFAULT_CRITERIA]
SAATY_SCALE = [1 / value for value in range(9, 1, -1)] + list(range(1, 10))
SCHEME_NAMES = {"annuity": "Аннуитет", "differentiated": "Дифференцированный"}
MODE_NAMES = {"walking": "Пешком", "transport": "На транспорте"}
DISPLAY_NAMES = {
    "listing_id": "ID Циан", "property_type": "Тип", "district": "Район / местность",
    "price_rub": "Цена, ₽", "area_m2": "Площадь, м²", "rooms": "Комнаты",
    "metro": "Станция", "travel_minutes": "До станции, мин",
    "transport_mode": "Способ", "ahp_rank": "Место AHP", "topsis_rank": "Место TOPSIS",
    "ahp_score": "Оценка AHP", "topsis_score": "Оценка TOPSIS", "pareto": "Парето",
    "annuity_payment": "Аннуитет, ₽/мес", "annuity_interest": "Проценты аннуитет, ₽",
    "differentiated_first_payment": "Первый дифф., ₽", "differentiated_last_payment": "Последний дифф., ₽",
    "differentiated_interest": "Проценты дифф., ₽", "source_url": "Объявление",
    "evidence_url": "Источник полей", "location": "Адрес", "observed_at": "Дата сбора",
    "evidence_note": "Примечание к источнику", "district_score": "Предпочтение района",
}


def rubles(value: object) -> str:
    return f"{float(value):,.0f} ₽".replace(",", " ")


def scale_label(value: float) -> str:
    return str(int(value)) if value >= 1 else f"1/{round(1 / value)}"


def display_table(frame: pd.DataFrame, columns: list[str]) -> None:
    shown = frame[columns].copy()
    if "transport_mode" in shown:
        shown["transport_mode"] = shown["transport_mode"].map(MODE_NAMES)
    shown = shown.rename(columns=DISPLAY_NAMES)
    config = {
        "Объявление": st.column_config.LinkColumn("Объявление", display_text="Циан"),
        "Источник полей": st.column_config.LinkColumn("Источник полей", display_text="Каталог"),
    }
    for column in columns:
        if column in frame and pd.api.types.is_numeric_dtype(frame[column]) and column != "pareto":
            label = DISPLAY_NAMES.get(column, column)
            pattern = "%.4f" if column.endswith("_score") else ("%.1f" if column == "area_m2" else "%.0f")
            config[label] = st.column_config.NumberColumn(label, format=pattern)
    st.dataframe(shown, hide_index=True, width="stretch", column_config=config)


def preference_controls(frame: pd.DataFrame) -> tuple[object, np.ndarray, dict[str, float]]:
    st.subheader("Важность критериев")
    st.write("Сравните критерии попарно. Значение больше 1 означает, что левый критерий важнее. "
             "Значение 1/3 означает, что правый критерий важнее в 3 раза.")
    matrix = np.eye(3)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        ratio = st.select_slider(
            f"{CRITERION_LABELS[left]} / {CRITERION_LABELS[right]}",
            options=SAATY_SCALE, value=float(DEFAULT_MATRIX[left, right]),
            format_func=scale_label, key=f"pair_{left}_{right}",
        )
        matrix[left, right], matrix[right, left] = ratio, 1 / ratio
    result = ahp_weights(matrix)
    weights = pd.DataFrame({"Критерий": CRITERION_LABELS, "Вес": result.weights})
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(pd.DataFrame(matrix, index=CRITERION_LABELS, columns=CRITERION_LABELS).round(4), width="stretch")
    with col2:
        st.dataframe(weights, hide_index=True, width="stretch", column_config={"Вес": st.column_config.NumberColumn(format="%.4f")})
    st.caption(f"λmax = {result.lambda_max:.4f} · CI = {result.consistency_index:.4f} · "
               f"CR = {result.consistency_ratio:.4f}. Для трёх критериев RI = 0,58.")
    if result.is_consistent:
        st.success("Суждения согласованы: CR ≤ 0,10.")
    else:
        st.warning("CR > 0,10: сравнения противоречат друг другу. Рейтинги предварительные. "
                   "Уточните попарные оценки перед выбором объекта.")
    st.subheader("Ваше отношение к районам")
    st.write("Задайте предпочтение от 1 до 9. По умолчанию везде 5.")
    district_frame = pd.DataFrame({
        "district": sorted(frame["district"].unique()), "district_score": 5,
    })
    edited = st.data_editor(
        district_frame, key="district_preferences", hide_index=True, width="stretch",
        disabled=["district"], num_rows="fixed",
        column_config={
            "district": st.column_config.TextColumn("Район / местность"),
            "district_score": st.column_config.NumberColumn("Предпочтение 1–9", min_value=1, max_value=9, step=1, required=True),
        },
    )
    return result, matrix, dict(zip(edited["district"], edited["district_score"], strict=True))


def ranking_view(ranked: pd.DataFrame, weights: np.ndarray, consistent: bool) -> None:
    st.subheader("Сравнение объектов")
    if not consistent:
        st.warning("Предварительный результат: матрица AHP требует уточнения на вкладке предпочтений.")
    sort_method = st.radio("Порядок рейтинга", ["AHP", "TOPSIS"], horizontal=True, key="ranking_method")
    order = "ahp_rank" if sort_method == "AHP" else "topsis_rank"
    ordered = ranked.sort_values([order, "listing_id"], kind="stable")
    top = ordered.iloc[0]
    st.write(f"**Первый объект по {sort_method}: {top['property_type'].lower()}, {top['area_m2']:g} м², "
             f"{top['district']}, {rubles(top['price_rub'])}.**")
    st.caption("При одинаковом месте несколько объектов равноценны по выбранному методу. "
               "Площадь задаёт ограничение поиска и не входит в три критерия рейтинга.")
    display_table(ordered, [
        "ahp_rank", "topsis_rank", "listing_id", "property_type", "district", "price_rub", "area_m2",
        "metro", "travel_minutes", "transport_mode", "ahp_score", "topsis_score", "pareto",
        "annuity_payment", "annuity_interest", "differentiated_first_payment", "differentiated_interest", "source_url",
    ])
    st.subheader("Почему объект получил такую оценку AHP")
    contributions = ranked.sort_values(["ahp_rank", "listing_id"], kind="stable").head(8)
    records = []
    for row in contributions.itertuples(index=False):
        for criterion, weight in zip(DEFAULT_CRITERIA, weights, strict=True):
            records.append({"Объект": str(row.listing_id), "Критерий": criterion.label,
                            "Вклад": getattr(row, f"{criterion.key}_priority") * weight})
    figure = px.bar(pd.DataFrame(records), x="Вклад", y="Объект", color="Критерий", orientation="h",
                    color_discrete_sequence=["#275DAD", "#1E9C91", "#E4A63A"])
    figure.update_layout(yaxis={"type": "category", "autorange": "reversed"}, legend_title_text="", margin={"l": 0, "r": 0, "t": 15, "b": 0})
    st.plotly_chart(figure, width="stretch")
    st.subheader("Компромисс между ценой и транспортом")
    chart = ranked.assign(price_mln=ranked["price_rub"] / 1_000_000,
                          pareto_label=ranked["pareto"].map({True: "Парето", False: "Доминируемый"}),
                          mode_label=ranked["transport_mode"].map(MODE_NAMES))
    figure = px.scatter(chart, x="price_mln", y="travel_minutes", color="pareto_label", symbol="property_type",
                        size="area_m2", size_max=24, hover_name="listing_id",
                        hover_data=["district", "mode_label", "district_score"],
                        labels={"price_mln": "Цена, млн ₽", "travel_minutes": "До станции, мин",
                                "pareto_label": "Статус", "property_type": "Тип", "area_m2": "Площадь, м²"},
                        color_discrete_map={"Парето": "#1E9C91", "Доминируемый": "#A4B1C3"})
    st.plotly_chart(figure, width="stretch")
    st.caption("Парето учитывает все три критерия, включая предпочтение района. На графике показаны две оси. "
               "Статус Парето не удаляет объект из AHP или TOPSIS.")


def mortgage_view(ranked: pd.DataFrame, rate: float, months: int) -> None:
    st.subheader("Два способа погашения")
    lookup = ranked.set_index("listing_id")
    identifier = st.selectbox(
        "Объект для расчёта", list(ranked.sort_values(["ahp_rank", "listing_id"])["listing_id"]),
        key="mortgage_listing", format_func=lambda key: (
            f"{key} · {lookup.loc[key, 'property_type']} · {lookup.loc[key, 'district']} · "
            f"{rubles(lookup.loc[key, 'price_rub'])}"
        ),
    )
    row = lookup.loc[identifier]
    col1, col2, col3 = st.columns(3)
    col1.metric("Цена объекта", rubles(row["price_rub"]))
    col2.metric("Первоначальный взнос", rubles(row["downpayment_rub"]))
    col3.metric("Сумма кредита", rubles(row["loan_rub"]))
    st.link_button("Объявление на Циан", row["source_url"])
    schedules = []
    summaries = []
    for scheme, name in SCHEME_NAMES.items():
        result = calculate_mortgage(row["loan_rub"], rate, months, scheme)
        schedules.append(result.schedule.assign(scheme=name))
        summaries.append({
            "Схема": name, "Первый платёж, ₽": float(result.first_payment), "Последний платёж, ₽": float(result.last_payment),
            "Проценты, ₽": float(result.total_interest), "Выплаты банку, ₽": float(result.total_paid),
            "Покупка со взносом, ₽": row["downpayment_rub"] + float(result.total_paid),
        })
    st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
    st.caption("Проценты — переплата сверх суммы кредита. Стоимость покупки со взносом = первоначальный взнос + выплаты банку.")
    schedule = pd.concat(schedules, ignore_index=True)
    figure = px.line(schedule, x="month", y="payment", color="scheme",
                     labels={"month": "Месяц", "payment": "Платёж, ₽", "scheme": "Схема"},
                     color_discrete_sequence=["#275DAD", "#1E9C91"])
    st.plotly_chart(figure, width="stretch")
    figure = px.line(schedule, x="month", y="balance", color="scheme",
                     labels={"month": "Месяц", "balance": "Остаток долга, ₽", "scheme": "Схема"},
                     color_discrete_sequence=["#275DAD", "#1E9C91"])
    st.plotly_chart(figure, width="stretch")
    with st.expander("Помесячные платежи"):
        st.dataframe(schedule.rename(columns={
            "month": "Месяц", "payment": "Платёж, ₽", "principal": "Основной долг, ₽",
            "interest": "Проценты, ₽", "balance": "Остаток, ₽", "scheme": "Схема",
        }), hide_index=True, width="stretch")
    st.download_button("Скачать оба графика CSV", export_csv(schedule), f"mortgage_{identifier}.csv", "text/csv")
    st.info("Модель использует фиксированную ставку и равные месячные периоды. Страхование, комиссии, "
            "налоги, ремонт и досрочное погашение не включены. Платежи округляются до копеек. "
            "Последний платёж закрывает остаток долга.")


def data_and_methods_view(frame: pd.DataFrame) -> None:
    st.subheader("Подготовленная выборка Циан")
    display_table(frame, [
        "listing_id", "property_type", "price_rub", "area_m2", "rooms", "location", "metro", "travel_minutes",
        "transport_mode", "source_url", "evidence_url", "observed_at", "evidence_note",
    ])
    st.download_button("Скачать исходную выборку CSV", export_csv(frame), "cian_moscow.csv", "text/csv")
    st.caption("0 комнат означает студию. Пустое число комнат у дома означает отсутствие данных. "
               "Пустой транспорт исключает объект из рейтинга. Район / местность — группа для настройки предпочтений.")
    st.subheader("Расчёты")
    st.markdown("**AHP.** Из попарной матрицы важности критериев вычисляется главный собственный вектор. "
                "Его нормированные компоненты дают веса. Согласованность проверяется по CI и CR.")
    st.latex(r"CI = \frac{\lambda_{max}-n}{n-1},\qquad CR = \frac{CI}{RI}")
    st.markdown("Для цены и времени локальный приоритет пропорционален обратному значению, "
                "для предпочтения района — прямому. Это собственные векторы согласованных матриц "
                "отношений показателей. Итог AHP равен взвешенной сумме приоритетов.")
    st.latex(r"p_{ij}=\frac{1/x_{ij}}{\sum_k 1/x_{kj}}\;(\min),\qquad p_{ij}=\frac{x_{ij}}{\sum_k x_{kj}}\;(\max),\qquad S_i=\sum_j w_jp_{ij}")
    st.markdown("**TOPSIS.** Столбцы нормируются по евклидовой норме и умножаются на веса AHP. "
                "Для каждого объекта определяются расстояния до лучшего и худшего идеалов. "
                "Чем выше относительная близость, тем выше рейтинг.")
    st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_k x_{kj}^2}},\qquad C_i=\frac{D_i^-}{D_i^++D_i^-}")
    st.markdown("При совпадении всех значений TOPSIS равен 0,5. Ранги обоих методов плотные: "
                "одинаковая оценка получает одинаковое место. AHP и TOPSIS могут давать разный порядок.")
    st.markdown("**Парето.** Объект доминируется, если другой объект не хуже по всем трём критериям "
                "и строго лучше хотя бы по одному. Проверка не зависит от весов.")
    st.markdown("**Ипотека.** Для суммы кредита P, ставки за месяц r и срока N:")
    st.latex(r"A=P\frac{r(1+r)^N}{(1+r)^N-1},\qquad D_t=\frac{P}{N}+\left(P-\frac{P(t-1)}{N}\right)r,\qquad r=\frac{\text{годовая ставка в процентах}}{1200}")
    st.caption("При нулевой ставке платёж равен P/N. Формулы описывают суммы до округления. "
               "Дифференцированный график распределяет копейки по накопленному погашению основного долга.")


def main() -> None:
    st.set_page_config(page_title="Выбор жилья в Москве", page_icon="🏠", layout="wide")
    st.title("Выбор жилья в Москве")
    with st.sidebar:
        st.header("Поиск жилья")
        upload = st.file_uploader("Своя выборка CSV", type="csv", help="Формат такой же, как у исходной выборки. До 1000 объектов и 5 МБ.")
    try:
        if upload is not None and upload.size > 5_000_000:
            raise ValueError("CSV должен занимать не более 5 МБ.")
        frame = load_properties(BytesIO(upload.getvalue()) if upload is not None else PROJECT_ROOT / "data" / "cian_moscow.csv")
    except (ValueError, OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"Не удалось прочитать выборку: {error}")
        st.stop()
    with st.sidebar:
        property_type = st.radio("Тип жилья", TYPE_OPTIONS, key="property_type")
        max_price = st.number_input("Максимальная цена, млн ₽", min_value=0.1, max_value=1000.0, value=50.0, step=0.5, key="max_price")
        min_area = st.number_input("Минимальная площадь, м²", min_value=0.0, max_value=5000.0, value=0.0, step=5.0, key="min_area")
        districts = st.multiselect("Район / местность", sorted(frame["district"].unique()), key="districts", help="Пустой список означает все районы.")
        transport = st.radio("Способ до станции", list(MODE_OPTIONS), key="transport")
        max_travel = st.slider("Максимум до станции, мин", 1, 90, 60, key="max_travel")
        st.header("Ипотечный сценарий")
        deposit = st.slider("Первоначальный взнос, %", 0, 100, 20, key="deposit")
        rate = st.number_input("Годовая ставка, %", min_value=0.0, max_value=60.0, value=14.0, step=0.1, key="rate")
        years = st.slider("Срок, лет", 1, 40, 20, key="years")
    ranking_tab, preferences_tab, mortgage_tab, data_tab = st.tabs(["Рейтинг", "Предпочтения AHP", "Ипотека", "Данные и методы"])
    with preferences_tab:
        ahp, matrix, preferences = preference_controls(frame)
    with data_tab:
        data_and_methods_view(frame)
    selected = filter_properties(frame, property_type, max_price * 1_000_000, min_area, districts, MODE_OPTIONS[transport], max_travel)
    if selected.empty:
        with ranking_tab:
            st.warning("Под эти условия объектов нет. Увеличьте бюджет, уменьшите площадь или расширьте транспортный фильтр.")
        with mortgage_tab:
            st.info("Для расчёта выберите условия, при которых найдётся хотя бы один объект.")
        return
    selected["district_score"] = selected["district"].map(preferences)
    ranked = rank_properties(selected, ahp.weights)
    ranked = add_mortgage_summaries(ranked, deposit, rate, years * 12)
    with ranking_tab:
        c1, c2, c3 = st.columns(3)
        c1.metric("Объектов после фильтра", len(ranked))
        c2.metric("Объектов Парето", int(ranked["pareto"].sum()))
        c3.metric("CR матрицы", f"{ahp.consistency_ratio:.4f}")
        if ranked["transport_mode"].nunique() > 1:
            st.warning("В выборке есть пешие пути и поездки транспортом. Сравнивается указанное в карточке время "
                       "до станции. Для однородного сравнения выберите один способ в фильтре.")
        if selected["district_score"].nunique() == 1:
            st.caption("Оценки районов одинаковы. Изменить их можно на вкладке «Предпочтения AHP».")
        ranking_view(ranked, ahp.weights, ahp.is_consistent)
        exported = ranked.copy()
        exported["annual_rate_percent"] = rate
        exported["term_months"] = years * 12
        exported["downpayment_percent"] = deposit
        exported["consistency_ratio"] = ahp.consistency_ratio
        for criterion, weight in zip(DEFAULT_CRITERIA, ahp.weights, strict=True):
            exported[f"weight_{criterion.key}"] = weight
        st.download_button("Скачать рейтинг и расчёты CSV", export_csv(exported), "housing_ranking.csv", "text/csv")
        settings = {"pairwise_matrix": matrix.tolist(), "weights": ahp.weights.tolist(), "district_preferences": preferences,
                    "mortgage": {"downpayment_percent": deposit, "annual_rate_percent": rate, "months": years * 12},
                    "filters": {"type": property_type, "max_price_rub": max_price * 1_000_000, "min_area_m2": min_area,
                                "districts": districts, "transport_mode": MODE_OPTIONS[transport], "max_travel_minutes": max_travel},
                    "source_observed_at": sorted(frame["observed_at"].unique())}
        st.download_button("Скачать параметры сценария JSON", json.dumps(settings, ensure_ascii=False, indent=2).encode("utf-8"), "scenario.json", "application/json")
    with mortgage_tab:
        mortgage_view(ranked, rate, years * 12)


if __name__ == "__main__":
    main()
