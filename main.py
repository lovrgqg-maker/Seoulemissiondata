import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="서울시 TMS 대기오염물질 일별 배출량 대시보드",
    page_icon="📈",
    layout="wide",
)

st.title("서울시 TMS 설치 사업장 대기오염물질 일별 배출량 대시보드")


# -----------------------------
# Utilities
# -----------------------------
DEFAULT_ENCODING_CANDIDATES = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]


def infer_pollutant_name(filename: str) -> str:
    name = filename or ""
    # 자주 쓰는 키워드 기반 추정
    if re.search(r"먼지|PM|분진", name, re.IGNORECASE):
        return "먼지"
    if re.search(r"질소산화물|NOx|NOX", name, re.IGNORECASE):
        return "질소산화물"
    if re.search(r"황산화물|SOx|SOX", name, re.IGNORECASE):
        return "황산화물"
    # 괄호/확장자 제거
    name = re.sub(r"\.[^.]+$", "", name)
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name if name else "업로드 데이터"


def read_csv_robust(file) -> pd.DataFrame:
    # file can be a path(str) or UploadedFile
    last_err = None
    for enc in DEFAULT_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(file, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def detect_date_col(df: pd.DataFrame) -> str:
    # Prefer typical Korean column names
    candidates = [c for c in df.columns if str(c).strip() in ["년월일", "일자", "날짜", "date", "Date"]]
    if candidates:
        return candidates[0]
    # Fallback: find column containing date-like pattern
    for c in df.columns:
        if "일" in str(c) or "date" in str(c).lower():
            return c
    # final fallback: first column
    return df.columns[0]


def to_long(df: pd.DataFrame, pollutant: str, source: str) -> pd.DataFrame:
    df = df.copy()
    date_col = detect_date_col(df)

    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Drop rows with invalid date
    df = df.dropna(subset=[date_col])

    # Ensure numeric in value columns
    value_cols = [c for c in df.columns if c != date_col]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long_df = df.melt(
        id_vars=[date_col],
        value_vars=value_cols,
        var_name="사업장",
        value_name="배출량",
    ).rename(columns={date_col: "일자"})

    long_df["오염물질"] = pollutant
    long_df["데이터출처"] = source

    return long_df


def iqr_outlier_count(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1 = np.percentile(s, 25)
    q3 = np.percentile(s, 75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


def format_number(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


# -----------------------------
# Load default datasets (include these CSVs in your repo)
# -----------------------------
@st.cache_data
def load_default_data() -> pd.DataFrame:
    # IMPORTANT: Streamlit Cloud 배포 시 아래 3개 파일을 repo에 포함시키세요.
    default_files = [
        ("먼지 배출량(2019년 샘플).csv", "먼지"),
        ("질소산화물 배출량(2019년 샘플).csv", "질소산화물"),
        ("황산화물 배출량(2019년 샘플).csv", "황산화물"),
    ]

    frames = []
    for path, pol in default_files:
        try:
            df = read_csv_robust(path)
            frames.append(to_long(df, pol, source="기본탑재"))
        except Exception:
            # 기본 파일이 repo에 없으면 조용히 스킵 (배포 전에는 반드시 포함 권장)
            continue

    if not frames:
        return pd.DataFrame(columns=["일자", "사업장", "배출량", "오염물질", "데이터출처"])
    return pd.concat(frames, ignore_index=True)


if "data_long" not in st.session_state:
    st.session_state["data_long"] = load_default_data()

if "upload_counter" not in st.session_state:
    st.session_state["upload_counter"] = 0


# -----------------------------
# Sidebar: upload & filters
# -----------------------------
st.sidebar.header("데이터 관리")

uploaded_files = st.sidebar.file_uploader(
    "같은 형식의 CSV를 추가 업로드하세요 (복수 선택 가능)",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_frames = []
    for uf in uploaded_files:
        try:
            df_up = read_csv_robust(uf)
            inferred = infer_pollutant_name(uf.name)
            pol_name = st.sidebar.text_input(
                f"오염물질명 확인/수정: {uf.name}",
                value=inferred,
                key=f"pol_{uf.name}",
            )
            st.session_state["upload_counter"] += 1
            source_name = f"업로드({st.session_state['upload_counter']})"
            new_frames.append(to_long(df_up, pol_name, source=source_name))
        except Exception as e:
            st.sidebar.error(f"업로드 실패: {uf.name} ({e})")

    if new_frames:
        st.session_state["data_long"] = pd.concat(
            [st.session_state["data_long"], *new_frames],
            ignore_index=True,
        )
        st.sidebar.success("업로드 데이터가 반영되었습니다.")

data_long = st.session_state["data_long"].copy()

if data_long.empty:
    st.warning("표시할 데이터가 없습니다. 기본탑재 CSV를 repo에 포함하거나 CSV를 업로드하세요.")
    st.stop()

# Basic cleanup
data_long["배출량"] = pd.to_numeric(data_long["배출량"], errors="coerce")

st.sidebar.header("분석 조건")

pollutants = sorted(data_long["오염물질"].dropna().unique().tolist())
pollutant_sel = st.sidebar.selectbox("오염물질", pollutants, index=0)

scope_sel = st.sidebar.radio("집계 단위", ["전체 합계", "사업장별"], index=0)

filtered_pol = data_long[data_long["오염물질"] == pollutant_sel].copy()

# Date range
min_date = filtered_pol["일자"].min()
max_date = filtered_pol["일자"].max()
date_range = st.sidebar.date_input(
    "기간",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = min_date, max_date

filtered_pol = filtered_pol[(filtered_pol["일자"] >= start_date) & (filtered_pol["일자"] <= end_date)]

# Facility selection
facilities = sorted(filtered_pol["사업장"].dropna().unique().tolist())
facility_sel = None
if scope_sel == "사업장별":
    facility_sel = st.sidebar.selectbox("사업장", facilities, index=0)

# Selected date: default latest date in filtered data
available_dates = sorted(filtered_pol["일자"].dt.date.unique().tolist())
default_selected_date = max(available_dates)
selected_date = st.sidebar.date_input("비교 기준일(기본=최근)", value=default_selected_date)
selected_date_ts = pd.to_datetime(selected_date)


# -----------------------------
# Prepare series for chart/metrics
# -----------------------------
def build_daily_series(df: pd.DataFrame, scope: str, facility: Optional[str]) -> pd.DataFrame:
    d = df.copy()
    if scope == "사업장별" and facility:
        d = d[d["사업장"] == facility]
    daily = (
        d.groupby("일자", as_index=False)["배출량"]
        .sum()
        .sort_values("일자")
    )
    return daily


daily_series = build_daily_series(filtered_pol, scope_sel, facility_sel)

if daily_series.empty:
    st.warning("선택 조건에서 데이터가 없습니다. 기간/사업장 조건을 조정하세요.")
    st.stop()

# Reference values
latest_date_ts = daily_series["일자"].max()
# If selected date isn't in series, fallback to latest
if selected_date_ts not in set(daily_series["일자"]):
    selected_date_ts = latest_date_ts

selected_value = float(daily_series.loc[daily_series["일자"] == selected_date_ts, "배출량"].iloc[0])
avg_value = float(daily_series["배출량"].mean())
diff = selected_value - avg_value
pct = (diff / avg_value * 100.0) if avg_value != 0 else np.nan

# Simple trend (7-day moving average last value vs overall avg)
daily_series["MA7"] = daily_series["배출량"].rolling(7, min_periods=1).mean()
ma7_last = float(daily_series.loc[daily_series["일자"] == selected_date_ts, "MA7"].iloc[0])


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["대시보드", "데이터 품질"])


with tab1:
    # Header context
    left, right = st.columns([2, 1])
    with left:
        scope_label = "전체 합계" if scope_sel == "전체 합계" else f"사업장: {facility_sel}"
        st.subheader(f"{pollutant_sel} · {scope_label}")

    with right:
        st.caption("비교 기준일")
        st.markdown(f"**{selected_date_ts.date()}**")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("기준일 배출량", format_number(selected_value))
    m2.metric("기간 평균", format_number(avg_value))
    m3.metric("차이(기준일-평균)", format_number(diff))
    m4.metric("증감률", "-" if np.isnan(pct) else f"{pct:,.2f}%")

    # Interpretation
    interp = "평균 수준"
    if not np.isnan(pct):
        if pct >= 10:
            interp = "평균 대비 높은 수준(상향)"
        elif pct <= -10:
            interp = "평균 대비 낮은 수준(하향)"
        else:
            interp = "평균과 유사한 수준"
    st.info(f"해석(간단): 선택일({selected_date_ts.date()})은 **{interp}**입니다. (기준: 평균 대비 ±10%)")

    # Plotly chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_series["일자"],
            y=daily_series["배출량"],
            mode="lines",
            name="일별 배출량",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_series["일자"],
            y=daily_series["MA7"],
            mode="lines",
            name="7일 이동평균",
        )
    )

    # 평균선
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        annotation_text="기간 평균",
        annotation_position="top left",
    )

    # 선택일 표시
    fig.add_vline(
        x=selected_date_ts,
        line_dash="dot",
        annotation_text=f"선택일 {selected_date_ts.date()}",
        annotation_position="top right",
    )

    fig.update_layout(
        height=520,
        xaxis_title="일자",
        yaxis_title="배출량(집계)",
        legend_title="",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Optional: Top facilities on selected date (only meaningful when scope is total)
    if scope_sel == "전체 합계":
        st.subheader("선택일 사업장별 기여도(상위 10)")
        day_detail = filtered_pol[filtered_pol["일자"] == selected_date_ts].copy()
        day_detail = day_detail.groupby("사업장", as_index=False)["배출량"].sum().sort_values("배출량", ascending=False)
        top10 = day_detail.head(10)

        if not top10.empty:
            fig2 = px.bar(top10, x="사업장", y="배출량")
            fig2.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("선택일에 사업장별 데이터가 없습니다.")


with tab2:
    st.subheader("결측치 및 이상치 점검(간단)")

    # Pivot back to wide for easier column checks
    date_col = "일자"
    wide = filtered_pol.pivot_table(index=date_col, columns="사업장", values="배출량", aggfunc="sum").sort_index()

    # Missing summary
    missing_counts = wide.isna().sum().sort_values(ascending=False)
    missing_df = missing_counts.reset_index()
    missing_df.columns = ["사업장", "결측치 개수"]
    st.caption("사업장별 결측치 개수(기간 내)")
    st.dataframe(missing_df, use_container_width=True, height=300)

    # Outlier summary (IQR rule)
    outlier_counts = pd.Series({col: iqr_outlier_count(wide[col]) for col in wide.columns}).sort_values(ascending=False)
    outlier_df = outlier_counts.reset_index()
    outlier_df.columns = ["사업장", "이상치(IQR) 개수"]
    st.caption("사업장별 이상치 개수(IQR 기준)")
    st.dataframe(outlier_df, use_container_width=True, height=300)

    st.caption("참고")
    st.write(
        "- 이상치(IQR)는 탐지용 지표이며, 실제 이상 여부는 공정/계절/운전조건/측정오류 등을 함께 확인해야 합니다.\n"
        "- 결측치는 0 배출과 구분되며, 업로드 파일의 공백/문자/누락 등으로 발생할 수 있습니다."
    )
