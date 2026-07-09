"""Streamlit UI for the Mexico Toys retail analytics assistant."""

from __future__ import annotations

from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chatbot_engine import (  # noqa: E402
    _historical_cols,
    build_dashboard_summary,
    clean_table,
    confidence_label,
    dashboard_reference_table,
    find_column,
    format_millions,
    format_number,
    get_model_score,
    get_chatbot_response,
    prepare_numeric_series,
)
from app.data_loader import load_all_data  # noqa: E402

try:
    from src.config import VISUALS_DIR  # noqa: E402
except Exception:  # pragma: no cover - keep the UI usable if config import fails
    VISUALS_DIR = PROJECT_ROOT / "visuals"


APP_TITLE = "Mexico Toys Retail Analytics Assistant"
APP_SUBTITLE = (
    "Ask about historical sales, products, stores, revenue and gross profit forecasts, "
    "profit opportunities, and model performance."
)

# Crucial 5 specific questions categorized for readability
EXAMPLE_QUESTIONS = [
    "How accurate is the 30-day revenue forecast?",
    "What is the projected revenue for the next 30 days?",
    "Which products are forecasted to have the highest profit?",
    "What were our top-selling products last month?",
    "Which store locations generated the most revenue historically?",
]

REQUESTED_VISUALS_DIR = PROJECT_ROOT / "data" / "outputs" / "visuals"
FORECAST_VISUALS = [
    {
        "label": "Revenue Forecast",
        "candidates": [
            REQUESTED_VISUALS_DIR / "revenue" / "revenue_forecast.png",
            VISUALS_DIR / "revenue" / "revenue_forecast.png",
        ],
    },
    {
        "label": "Gross Profit Forecast",
        "candidates": [
            REQUESTED_VISUALS_DIR / "profit" / "profit_forecast.png",
            VISUALS_DIR / "profit" / "profit_forecast.png",
        ],
    },
]


def escape_html(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def message_to_html(value: Any) -> str:
    text = escape_html(value)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text.replace("\n", "<br>")


def chat_timestamp() -> str:
    return datetime.now().strftime("%H:%M")


def dataset_frame(data: dict[str, dict[str, object]], key: str) -> pd.DataFrame:
    df = data.get(key, {}).get("df")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def inject_theme_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --orange: #E65A00;
            --orange-dark: #B94700;
            --dark-green: #0B3D20;
            --light-green: #EDF5EF;
            --background: #FAF7F2;
            --white: #FFFFFF;
            --text: #1E1E1E;
            --muted: #6B7280;
            --border: #E5E7EB;
            --shadow: 0 12px 28px rgba(11, 61, 32, 0.08);
        }

        #MainMenu, header, footer {
            visibility: hidden;
        }

        .stApp {
            background: var(--background);
            color: var(--text);
        }

        .stApp, .stMarkdown, .stMarkdown p, .stMarkdown span, label, p, div {
            color: var(--text);
            letter-spacing: 0;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 1.35rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--dark-green);
        }

        .sidebar-status-card {
            background: var(--light-green);
            border: 1px solid rgba(11, 61, 32, 0.14);
            border-radius: 8px;
            margin: 0.3rem 0 0.8rem 0;
            padding: 0.75rem 0.85rem;
        }

        .sidebar-status-title {
            color: var(--dark-green);
            font-size: 0.95rem;
            font-weight: 850;
            line-height: 1.25;
        }

        .sidebar-status-detail {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.28rem;
        }

        h1, h2, h3 {
            color: var(--dark-green);
            letter-spacing: 0;
        }

        .hero-panel {
            margin-bottom: 1rem;
        }

        .hero-title {
            color: var(--dark-green);
            font-size: 2rem;
            font-weight: 850;
            line-height: 1.15;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin: 0.35rem 0 0 0;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }

        .hero-badge {
            background: var(--white);
            border: 1px solid rgba(230, 90, 0, 0.22);
            border-radius: 999px;
            color: var(--dark-green);
            display: inline-flex;
            font-size: 0.76rem;
            font-weight: 750;
            padding: 0.24rem 0.65rem;
        }

        .section-label {
            color: var(--orange);
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            margin: 1.05rem 0 0.45rem 0;
            text-transform: uppercase;
        }

        .kpi-card {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 8px;
            border-top: 4px solid var(--orange);
            box-shadow: var(--shadow);
            min-height: 120px;
            padding: 0.9rem;
        }

        .section-card {
            background: var(--white);
            border: 1px solid var(--border);
            border-left: 4px solid var(--orange);
            border-radius: 8px;
            box-shadow: var(--shadow);
            margin: 0.35rem 0 0.85rem 0;
            padding: 0.9rem;
        }

        .section-card.green {
            border-left-color: var(--dark-green);
        }

        .section-card-title {
            color: var(--dark-green);
            font-size: 0.95rem;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .kpi-value {
            color: var(--dark-green);
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.15;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }

        .source-chip, .status-chip, .confidence-chip {
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 700;
            margin-top: 0.5rem;
            padding: 0.18rem 0.55rem;
            width: fit-content;
        }

        .source-chip {
            background: var(--light-green);
            border: 1px solid rgba(11, 61, 32, 0.16);
            color: var(--dark-green);
        }

        .status-chip {
            background: var(--light-green);
            border: 1px solid rgba(11, 61, 32, 0.16);
            color: var(--dark-green);
            margin: 0.15rem 0 0.25rem 0;
        }

        .status-chip.missing {
            background: #FFF4ED;
            border-color: rgba(230, 90, 0, 0.24);
            color: #9A3412;
        }

        .status-chip.disabled {
            background: #FFF7ED;
            border-color: rgba(230, 90, 0, 0.22);
            color: #8A3B00;
        }

        .notice-card {
            background: #FFF7ED;
            border: 1px solid rgba(230, 90, 0, 0.22);
            border-left: 4px solid var(--orange);
            border-radius: 8px;
            box-shadow: var(--shadow);
            margin: 0.4rem 0 1rem 0;
            padding: 0.95rem 1rem;
        }

        .chat-shell {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
            margin-top: 0.8rem;
            padding: 1rem;
        }

        .chat-header {
            align-items: center;
            border-bottom: 1px solid #F0E7DC;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.85rem;
            padding-bottom: 0.75rem;
        }

        .chat-name {
            color: var(--dark-green);
            font-size: 1rem;
            font-weight: 850;
        }

        .chat-subtitle {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.1rem;
        }

        .message-row {
            display: flex;
            margin: 0.7rem 0;
            width: 100%;
        }

        .message-row.user {
            justify-content: flex-end;
        }

        .message-row.assistant {
            justify-content: flex-start;
        }

        .message-stack {
            max-width: 78%;
        }

        .message-row.user .message-stack {
            max-width: 68%;
        }

        .bubble {
            border-radius: 8px;
            font-size: 0.94rem;
            line-height: 1.48;
            padding: 0.72rem 0.85rem;
        }

        .bubble.user {
            background: var(--orange);
            color: #FFFFFF;
        }

        .bubble.user strong,
        .bubble.user code {
            color: #FFFFFF;
        }

        .bubble.assistant {
            background: var(--light-green);
            border: 1px solid rgba(11, 61, 32, 0.14);
            color: var(--dark-green);
        }

        .bubble.assistant strong {
            color: var(--dark-green);
        }

        .bubble code {
            background: rgba(11, 61, 32, 0.08);
            border-radius: 6px;
            padding: 0.08rem 0.28rem;
        }

        .message-meta {
            color: #8A948D;
            font-size: 0.72rem;
            margin-top: 0.25rem;
        }

        .message-row.user .message-meta {
            text-align: right;
        }

        .answer-meta {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 0.35rem 0 0.8rem 0;
            padding: 0.68rem 0.75rem;
        }

        .confidence-track {
            background: #F3F4F6;
            border: 1px solid var(--border);
            border-radius: 999px;
            height: 0.5rem;
            margin-top: 0.55rem;
            overflow: hidden;
        }

        .confidence-fill {
            background: var(--confidence-color);
            border-radius: 999px;
            height: 100%;
            width: var(--confidence-width);
        }

        .guide-shell {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
            margin-top: 1rem;
            padding: 1rem;
        }

        .guide-bar {
            border: 1px solid var(--border);
            border-radius: 8px;
            display: flex;
            min-height: 64px;
            overflow: hidden;
            width: 100%;
        }

        .guide-segment {
            color: #FFFFFF;
            display: flex;
            flex-direction: column;
            font-size: 0.74rem;
            justify-content: center;
            line-height: 1.15;
            min-width: 80px;
            padding: 0.5rem;
        }

        .guide-segment strong {
            color: #FFFFFF;
            font-size: 0.82rem;
        }

        .confidence-chip {
            background: var(--confidence-bg);
            border: 1px solid var(--confidence-border);
            color: var(--confidence-color);
            margin-right: 0.45rem;
        }

        .muted-text {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.45;
        }

        .stButton > button {
            background: #FFFFFF;
            border: 1px solid #F3C7AA;
            border-radius: 8px;
            color: var(--text);
            min-height: 2.2rem;
            white-space: normal;
        }

        .stButton > button p {
            color: var(--text);
        }

        .stButton > button:hover {
            background: #FFF4ED;
            border-color: var(--orange);
            color: var(--orange-dark);
        }

        .stButton > button:hover p {
            color: var(--orange-dark);
        }

        [data-testid="stChatInput"] {
            background: transparent;
        }

        [data-testid="stChatInput"] textarea {
            border: 1px solid #F3C7AA;
            border-radius: 8px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stTabs"] button {
            color: var(--dark-green);
        }

        @media (max-width: 760px) {
            .hero-title {
                font-size: 1.55rem;
            }

            .message-stack,
            .message-row.user .message-stack {
                max-width: 94%;
            }

            .guide-bar {
                flex-direction: column;
            }

            .guide-segment {
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_kpi_value(label: str, value: Any) -> str:
    if value is None:
        return "Unavailable"
    if "Revenue" in label or "Profit" in label:
        return f"${format_millions(value)}"
    if "Units" in label:
        return format_millions(value)
    return format_number(value)


def confidence_colors(score: int | float) -> tuple[str, str, str]:
    if score < 30:
        return "#B42318", "#FDECEC", "#F5B8B0"
    if score < 60:
        return "#C2410C", "#FFF4ED", "#FDBA74"
    if score < 80:
        return "#A16207", "#FEF9C3", "#FACC15"
    if score < 90:
        return "#15803D", "#ECFDF3", "#86EFAC"
    return "#0B3D20", "#EDF5EF", "#75C08A"


def source_chip(source: str, missing: bool = False) -> str:
    klass = "status-chip missing" if missing else "source-chip"
    return f'<span class="{klass}">{escape_html(source)}</span>'


def render_header() -> None:
    badges = "".join(
        f'<span class="hero-badge">{escape_html(label)}</span>'
        for label in ["Historical Analytics", "30-Day Forecasting", "Local Routing", "Model Monitoring"]
    )
    st.markdown(
        f"""
        <div class="hero-panel">
            <h1 class="hero-title">{escape_html(APP_TITLE)}</h1>
            <p class="hero-subtitle">{escape_html(APP_SUBTITLE)}</p>
            <div class="badge-row">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_technical_architecture() -> None:
    with st.expander("ℹ️ Technical Architecture & Methodology", expanded=False):
        st.markdown(
            """
            **Core Engine**: Holt-Winters Additive Exponential Smoothing (`statsmodels`).

            **Seasonality**: Captures weekly patterns (`L=7`).

            **Horizon**: Strictly 30-day rolling forecast to prevent error compounding.

            **Inference**: Rule-based local routing engine (`$0` API cost, zero mathematical hallucinations).
            """
        )


def render_kpi_cards(data: dict[str, dict[str, object]]) -> None:
    summary = build_dashboard_summary(data)
    order = ["revenue", "profit", "units", "stores", "products", "stock"]
    st.markdown('<div class="section-label">Executive KPIs</div>', unsafe_allow_html=True)
    for row in [order[:3], order[3:]]:
        columns = st.columns(len(row))
        for column, key in zip(columns, row):
            metric = summary.get(key, {"label": key.title(), "value": None, "source": "Unavailable"})
            label = str(metric["label"])
            value = metric.get("value")
            source = str(metric.get("source", "Unavailable"))
            with column:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{escape_html(label)}</div>
                        <div class="kpi-value">{escape_html(format_kpi_value(label, value))}</div>
                        {source_chip(source, missing=value is None)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def status_badge(available: bool, disabled: bool = False) -> str:
    if disabled:
        label = "Disabled"
        klass = "status-chip disabled"
    else:
        label = "Loaded" if available else "Missing"
        klass = "status-chip" if available else "status-chip missing"
    return f'<span class="{klass}">{label}</span>'


def render_sidebar(data: dict[str, dict[str, object]]) -> dict[str, bool]:
    # 1. Clean Title setup
    st.sidebar.markdown("## 🇲🇽 Mexico Toys Sales Bot")
    st.sidebar.divider()

    # 2. Refactored Minimalist System Check Section
    st.sidebar.markdown("### ⚙️ System Check")
    forecast_active = any(
        bool(data.get(key, {}).get("available"))
        for key in ["revenue_forecast", "profit_forecast"]
    )
    forecast_status = "🟢 Forecasts: Active" if forecast_active else "🔴 Forecasts: Unavailable"
    status_detail = (
        "ML Forecast datasets successfully loaded."
        if forecast_active
        else "Required forecasting outputs are offline."
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status-card">
            <div class="sidebar-status-title">{escape_html(forecast_status)}</div>
            <div class="sidebar-status-detail">{escape_html(status_detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean Expander showing Schema of Historical Data
    columns = data.get("historical_sales", {}).get("column_names", [])
    with st.sidebar.expander("📊 Historical Data Columns", expanded=False):
        if columns:
            st.markdown("\n".join(f"- `{column}`" for column in columns))
        else:
            st.caption("No historical columns loaded.")

    st.sidebar.divider()

    # 3. Clean Click-Routing Section for Suggested Questions
    st.sidebar.markdown("### 💡 Suggested Questions")
    
    # Render exactly your 5 desired questions
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if st.sidebar.button(question, key=f"sidebar-suggested-question-{index}", use_container_width=True):
            submit_question(question, data)
            st.rerun()

    # Safely return downstream layout properties back to main()
    return {
        "show_preview": False,
        "show_dashboard": False,
        "use_dashboard_reference": True,
        "show_debug": False,
    }


def detected_columns_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    col_map = _historical_cols(df)
    labels = {
        "revenue": "Revenue",
        "profit": "Gross profit",
        "units": "Units",
        "product": "Product",
        "store": "Store",
        "location": "Store location",
        "category": "Category",
        "date": "Date/month",
        "stock": "Stock on hand",
        "out_of_stock": "Out of stock",
    }
    rows = []
    for key, label in labels.items():
        col, match = col_map.get(key, (None, "missing"))
        rows.append({"field": label, "detected_column": col or "Not found", "match": match})
    return pd.DataFrame(rows)


def render_optional_panels(data: dict[str, dict[str, object]], options: dict[str, bool]) -> None:
    historical_df = dataset_frame(data, "historical_sales")

    if options["show_preview"]:
        with st.expander("Historical Data Preview", expanded=True):
            if historical_df.empty:
                st.warning("No historical sales data is loaded.")
            else:
                st.caption("First 20 cleaned rows from the loaded historical sales dataset.")
                st.dataframe(clean_table(historical_df, max_rows=20), use_container_width=True, hide_index=True)
                detected = detected_columns_table(historical_df)
                if not detected.empty:
                    st.caption("Detected analytics columns")
                    st.dataframe(detected, use_container_width=True, hide_index=True)

    if options["show_dashboard"]:
        with st.expander("Power BI Reference KPIs", expanded=True):
            st.caption(
                "Reference values only. Historical questions are calculated from the loaded CSV whenever the required columns exist."
            )
            st.dataframe(dashboard_reference_table(), use_container_width=True, hide_index=True)
    elif not options.get("use_dashboard_reference", True):
        st.info("Dashboard reference fallback display is turned off. Loaded CSV calculations are still used first.")

    if options["show_debug"]:
        with st.expander("File Debug JSON", expanded=False):
            st.json(
                {
                    key: {
                        "available": bool(item.get("available")),
                        "rows": int(item.get("rows", 0) or 0),
                        "columns": int(item.get("columns", 0) or 0),
                        "path": str(item.get("path", "")),
                        "warning": item.get("warning"),
                        "disabled": bool(item.get("disabled")),
                        "display_status": item.get("display_status"),
                    }
                    for key, item in data.items()
                }
            )


def initialize_chat() -> None:
    if "messages" in st.session_state:
        return
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi. I use the local historical sales CSV for actuals, forecast CSVs for future views, "
                "the model report for accuracy, and Power BI KPIs only as reference context. Product demand "
                "and inventory restocking are disabled because the demand model did not meet reliability standards."
            ),
            "source": "assistant help",
            "confidence": 90,
            "confidence_reason": "This is a local assistant orientation message.",
            "model_performance_score": None,
            "model_performance_label": None,
            "table": None,
            "chart": None,
            "table_rows": 10,
            "timestamp": chat_timestamp(),
        }
    ]


def submit_question(question: str, data: dict[str, dict[str, object]]) -> None:
    text = question.strip()
    if not text:
        return
    st.session_state["messages"].append({"role": "user", "content": text, "timestamp": chat_timestamp()})
    result = get_chatbot_response(text, data)
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "source": result.source,
            "confidence": result.confidence,
            "confidence_reason": result.confidence_reason,
            "model_performance_score": result.model_performance_score,
            "model_performance_label": result.model_performance_label,
            "table": result.table,
            "chart": result.chart,
            "table_rows": result.table_rows,
            "timestamp": chat_timestamp(),
        }
    )


def render_confidence_meta(message: dict[str, Any]) -> None:
    score = int(max(0, min(100, round(message.get("confidence") or 30))))
    label = confidence_label(score)
    color, bg, border = confidence_colors(score)
    source = str(message.get("source") or "unknown")
    reason = str(message.get("confidence_reason") or "No confidence reason supplied.")
    model_score = message.get("model_performance_score")
    model_label = message.get("model_performance_label")
    model_html = ""
    if model_score is not None:
        model_html = (
            f'<div class="muted-text" style="margin-top:0.35rem;">'
            f'ML Model Performance Score: {float(model_score):.2f}%'
            f'{f" ({escape_html(model_label)})" if model_label else ""}</div>'
        )
    st.markdown(
        f"""
        <div class="answer-meta">
            <span class="confidence-chip" style="--confidence-color:{color}; --confidence-bg:{bg}; --confidence-border:{border};">
                Confidence {score}% - {escape_html(label)}
            </span>
            <span class="source-chip">Source: {escape_html(source)}</span>
            <div class="confidence-track">
                <div class="confidence-fill" style="--confidence-width:{score}%; --confidence-color:{color};"></div>
            </div>
            <div class="muted-text" style="margin-top:0.4rem;">{escape_html(reason)}</div>
            {model_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart(chart: dict[str, Any] | None) -> None:
    if not chart:
        return
    df = chart.get("df")
    x_col = chart.get("x")
    y_col = chart.get("y")
    if not isinstance(df, pd.DataFrame) or df.empty or not x_col or not y_col:
        return
    if x_col not in df.columns or y_col not in df.columns:
        return
    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = prepare_numeric_series(plot_df, y_col).fillna(0)
    plot_df = plot_df.dropna(subset=[x_col])
    if plot_df.empty:
        return
    st.caption(str(chart.get("title", "")))
    indexed = plot_df.set_index(x_col)
    if chart.get("type") == "line":
        st.line_chart(indexed[[y_col]], use_container_width=True)
    else:
        st.bar_chart(indexed[[y_col]], use_container_width=True)


def render_prompt_buttons(data: dict[str, dict[str, object]], key_prefix: str = "main") -> None:
    st.markdown('<div class="section-label">Example Questions</div>', unsafe_allow_html=True)
    for row_index, row in enumerate([EXAMPLE_QUESTIONS[:4], EXAMPLE_QUESTIONS[4:8], EXAMPLE_QUESTIONS[8:]]):
        if not row:
            continue
        columns = st.columns(len(row))
        for column, question in zip(columns, row):
            with column:
                if st.button(question, key=f"{key_prefix}-prompt-{row_index}-{question}", use_container_width=True):
                    submit_question(question, data)
                    st.rerun()


def render_confidence_guide() -> None:
    st.markdown(
        """
        <div class="guide-shell">
            <div class="section-card-title">Confidence Guide</div>
            <div class="guide-bar">
                <div class="guide-segment" style="width:29%; background:#B42318;"><strong>0-29%</strong><span>Do not trust</span></div>
                <div class="guide-segment" style="width:30%; background:#C2410C;"><strong>30-59%</strong><span>Low confidence</span></div>
                <div class="guide-segment" style="width:20%; background:#D99A00;"><strong>60-79%</strong><span>Medium confidence</span></div>
                <div class="guide-segment" style="width:10%; background:#15803D;"><strong>80-89%</strong><span>High confidence</span></div>
                <div class="guide-segment" style="width:11%; background:#0B3D20;"><strong>90-100%</strong><span>Very high confidence</span></div>
            </div>
            <p class="muted-text" style="margin:0.8rem 0 0 0;">
                Confidence is based on source quality, column detection, data availability, whether calculations came directly from the loaded dataset, and model error where applicable.
            </p>
            <p class="muted-text" style="margin:0.35rem 0 0 0;">
                Below 30%: do not trust this answer. 90% or above: generally safe to trust for normal business decisions, assuming the source data is correct.
            </p>
            <p class="muted-text" style="margin:0.35rem 0 0 0;">
                Demand/restock forecast outputs are hidden because the demand model performance was weak. This prevents low-confidence recommendations from being shown as business advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    for message in st.session_state["messages"]:
        role = str(message.get("role", "assistant"))
        timestamp = escape_html(message.get("timestamp") or "")
        content = message_to_html(message.get("content", ""))
        if role == "user":
            st.markdown(
                f"""
                <div class="message-row user">
                    <div class="message-stack">
                        <div class="bubble user">{content}</div>
                        <div class="message-meta">{timestamp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        st.markdown(
            f"""
            <div class="message-row assistant">
                <div class="message-stack">
                    <div class="bubble assistant">{content}</div>
                    <div class="message-meta">{timestamp}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_confidence_meta(message)
        render_chart(message.get("chart"))
        table = message.get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(
                clean_table(table, max_rows=int(message.get("table_rows", 10))),
                use_container_width=True,
                hide_index=True,
            )


def render_chat_tab(data: dict[str, dict[str, object]]) -> None:
    st.markdown('<div class="section-label">Analytics Chat</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chat-shell">
            <div class="chat-header">
                <div>
                    <div class="chat-name">Retail Analytics Assistant</div>
                    <div class="chat-subtitle">Historical actuals first, forecasts only for future-looking questions</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_prompt_buttons(data, key_prefix="chat")
    render_chat_history()
    render_confidence_guide()
    question = st.chat_input("Ask about sales, products, stores, revenue/profit forecasts, or model performance")
    if question:
        submit_question(question, data)
        st.rerun()


def grouped_summary(
    df: pd.DataFrame,
    group_col: str | None,
    metric_cols: list[str | None],
    sort_col: str | None,
    limit: int = 10,
) -> pd.DataFrame:
    if df.empty or not group_col or group_col not in df.columns:
        return pd.DataFrame()
    metrics = [col for col in metric_cols if col and col in df.columns]
    if not metrics:
        return pd.DataFrame()
    working = df[[group_col] + metrics].copy()
    for metric in metrics:
        working[metric] = prepare_numeric_series(working, metric).fillna(0)
    order_col = sort_col if sort_col in metrics else metrics[0]
    table = working.groupby(group_col, as_index=False)[metrics].sum().sort_values(order_col, ascending=False)
    return clean_table(table, max_rows=limit)


def monthly_revenue_table(df: pd.DataFrame, date_col: str | None, revenue_col: str | None) -> pd.DataFrame:
    if df.empty or not date_col or not revenue_col or date_col not in df.columns or revenue_col not in df.columns:
        return pd.DataFrame()
    working = df[[date_col, revenue_col]].copy()
    working[revenue_col] = prepare_numeric_series(working, revenue_col).fillna(0)
    dates = pd.to_datetime(working[date_col], errors="coerce")
    if dates.notna().any():
        working["month"] = dates.dt.to_period("M").astype(str)
        table = working.groupby("month", as_index=False)[revenue_col].sum().sort_values("month")
    else:
        month_order = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
        working["month"] = working[date_col].astype(str).str[:3]
        working["_month_order"] = working["month"].map(month_order).fillna(99)
        table = working.groupby(["month", "_month_order"], as_index=False)[revenue_col].sum().sort_values("_month_order")
        table = table.drop(columns=["_month_order"], errors="ignore")
    return clean_table(table, max_rows=25)


def render_table_or_info(table: pd.DataFrame, message: str, rows: int = 10) -> None:
    if table.empty:
        st.info(message)
    else:
        st.dataframe(clean_table(table, max_rows=rows), use_container_width=True, hide_index=True)


def render_historical_tab(data: dict[str, dict[str, object]]) -> None:
    historical = data.get("historical_sales", {})
    df = dataset_frame(data, "historical_sales")

    with st.expander("Dataset Overview", expanded=True):
        if bool(historical.get("available")):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Rows", f"{int(historical.get('rows', 0) or 0):,}")
            col_b.metric("Columns", f"{int(historical.get('columns', 0) or 0):,}")
            col_c.metric("Source", Path(str(historical.get("path"))).name)
            st.caption(str(historical.get("path")))
        else:
            st.warning(historical.get("warning") or "No historical dataset loaded.")

    if df.empty:
        return

    col_map = _historical_cols(df)
    revenue_col = col_map["revenue"][0]
    profit_col = col_map["profit"][0]
    units_col = col_map["units"][0]
    product_col = col_map["product"][0]
    store_col = col_map["store"][0]
    category_col = col_map["category"][0]
    date_col = col_map["date"][0]

    with st.expander("Detected Columns", expanded=True):
        st.dataframe(detected_columns_table(df), use_container_width=True, hide_index=True)

    with st.expander("Historical Preview", expanded=True):
        st.dataframe(clean_table(df, max_rows=25), use_container_width=True, hide_index=True)

    with st.expander("Top Products", expanded=True):
        table = grouped_summary(df, product_col, [revenue_col, profit_col, units_col], revenue_col, limit=10)
        render_table_or_info(table, "Product and revenue/profit/unit columns were not available.")
        if not table.empty and product_col in table.columns and revenue_col in table.columns:
            st.bar_chart(table.set_index(product_col)[[revenue_col]], use_container_width=True)

    with st.expander("Top Stores", expanded=False):
        table = grouped_summary(df, store_col, [revenue_col, profit_col, units_col], revenue_col, limit=10)
        render_table_or_info(table, "Store and revenue/profit/unit columns were not available.")
        if not table.empty and store_col in table.columns and revenue_col in table.columns:
            st.bar_chart(table.set_index(store_col)[[revenue_col]], use_container_width=True)

    with st.expander("Category Summary", expanded=False):
        table = grouped_summary(df, category_col, [revenue_col, profit_col, units_col], revenue_col, limit=10)
        render_table_or_info(table, "Category and metric columns were not available.")
        if not table.empty and category_col in table.columns and revenue_col in table.columns:
            st.bar_chart(table.set_index(category_col)[[revenue_col]], use_container_width=True)

    with st.expander("Monthly Revenue", expanded=False):
        table = monthly_revenue_table(df, date_col, revenue_col)
        render_table_or_info(table, "Date/month and revenue columns were not available.", rows=25)
        if not table.empty and revenue_col in table.columns:
            st.line_chart(table.set_index("month")[[revenue_col]], use_container_width=True)


def response_message(result: Any) -> dict[str, Any]:
    return {
        "source": result.source,
        "confidence": result.confidence,
        "confidence_reason": result.confidence_reason,
        "model_performance_score": result.model_performance_score,
        "model_performance_label": result.model_performance_label,
    }


def render_forecast_section(
    title: str,
    key: str,
    question: str,
    data: dict[str, dict[str, object]],
    show_raw_previews: bool,
) -> None:
    dataset = data.get(key, {})
    result = get_chatbot_response(question, data)
    with st.expander(title, expanded=key in {"revenue_forecast", "profit_forecast"}):
        available = bool(dataset.get("available"))
        st.markdown(
            f"""
            <div class="section-card {'green' if available else ''}">
                <div class="section-card-title">{escape_html(title)}</div>
                <div class="muted-text">{escape_html(str(dataset.get("path", "")))}</div>
                {status_badge(available)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(result.answer)
        render_confidence_meta(response_message(result))
        render_chart(result.chart)
        if result.table is not None and not result.table.empty:
            st.dataframe(clean_table(result.table, max_rows=result.table_rows), use_container_width=True, hide_index=True)
        if show_raw_previews:
            preview = dataset_frame(data, key)
            if not preview.empty:
                st.caption("Raw preview limited to 25 rows.")
                st.dataframe(clean_table(preview, max_rows=25), use_container_width=True, hide_index=True)


def render_forecasts_tab(data: dict[str, dict[str, object]], show_raw_previews: bool) -> None:
    st.markdown(
        """
        <div class="notice-card">
            <div class="section-card-title">Demand and Restock Forecasting Disabled</div>
            <div class="muted-text">
                Demand and restock forecasting are currently disabled because the demand model did not meet reliability standards.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sections = [
        ("Revenue Forecast", "revenue_forecast", "What is the revenue forecast for the next 30 days?"),
        ("Gross Profit Forecast", "profit_forecast", "What is the profit forecast for the next 30 days?"),
        ("Profit Opportunities", "top_profit_products", "What are the top profit products next 30 days?"),
    ]
    for title, key, question in sections:
        render_forecast_section(title, key, question, data, show_raw_previews)


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def render_forecast_visuals_tab(data: dict[str, dict[str, object]], show_raw_previews: bool) -> None:
    st.markdown('<div class="section-label">Forecast Visuals</div>', unsafe_allow_html=True)
    columns = st.columns(len(FORECAST_VISUALS))
    for column, visual in zip(columns, FORECAST_VISUALS):
        label = str(visual["label"])
        candidates = [Path(path) for path in visual["candidates"]]
        path = first_existing_path(candidates)
        with column:
            st.markdown(f"#### {label}")
            if path:
                st.image(str(path), caption=path.name, use_container_width=True)
            else:
                checked_paths = "\n".join(f"- `{candidate}`" for candidate in candidates)
                st.warning(f"{label} image is unavailable. Checked:\n{checked_paths}")

    with st.expander("Forecast Data Details", expanded=False):
        render_forecasts_tab(data, show_raw_previews=show_raw_previews)


def render_model_score_card(title: str, score: dict[str, Any] | None) -> None:
    if not score:
        value = "Unavailable"
        detail = "Model evaluation missing"
        source = "Unavailable"
        missing = True
    else:
        value = f"{float(score['score']):.2f}%"
        detail = f"MAPE {float(score['mape']):.2f}% - {score['label']}"
        source = "model_evaluation_summary.csv"
        missing = False
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape_html(title)}</div>
            <div class="kpi-value">{escape_html(value)}</div>
            <div class="muted-text">{escape_html(detail)}</div>
            {source_chip(source, missing=missing)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_performance_table(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame()
    table = clean_table(model_df, max_rows=len(model_df))
    mape_col = find_column(table, ["mape"])
    if mape_col:
        scores = 100 - pd.to_numeric(table[mape_col], errors="coerce")
        table["model_score"] = scores.clip(lower=0, upper=100).round(2)
        table["score_label"] = table["model_score"].apply(lambda value: "Unavailable" if pd.isna(value) else str(model_score_label_for_display(float(value))))
    target_col = find_column(table, ["target", "model_area"])
    if target_col:
        demand_mask = table[target_col].astype(str).str.lower().str.contains("demand|unit|aggregate_units", regex=True)
        table["user_facing_status"] = "Active for forecast summaries"
        table.loc[demand_mask, "user_facing_status"] = "Disabled for chatbot decisions"
        table["recommendation_note"] = "Used for revenue/profit forecast answers when available"
        table.loc[demand_mask, "recommendation_note"] = "Weak / needs improvement; not used in user-facing recommendations"
    return table


def model_score_label_for_display(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Moderate"
    if score >= 30:
        return "Weak"
    return "Poor"


def render_model_performance_tab(data: dict[str, dict[str, object]]) -> None:
    model_df = dataset_frame(data, "model_evaluation")
    with st.expander("Model Score Cards", expanded=True):
        columns = st.columns(2)
        with columns[0]:
            render_model_score_card("Revenue Model Score", get_model_score(model_df, "revenue"))
        with columns[1]:
            render_model_score_card("Gross Profit Model Score", get_model_score(model_df, "profit"))

    with st.expander("Disabled Models - Not Used", expanded=True):
        columns = st.columns([1, 2])
        with columns[0]:
            render_model_score_card("Demand Model Score", get_model_score(model_df, "demand"))
        with columns[1]:
            st.markdown(
                """
                <div class="notice-card" style="margin-top:0;">
                    <div class="section-card-title">Demand Model Usage Status</div>
                    <div class="muted-text">
                        Disabled for chatbot decisions. Weak / needs improvement. Not used in user-facing recommendations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Evaluation Metrics", expanded=True):
        if model_df.empty:
            st.warning("Model evaluation CSV is not loaded.")
        else:
            st.dataframe(model_performance_table(model_df), use_container_width=True, hide_index=True)

    with st.expander("Interpretation", expanded=True):
        result = get_chatbot_response("How good is the model?", data)
        st.markdown(result.answer)
        render_confidence_meta(response_message(result))
        if result.table is not None and not result.table.empty:
            st.dataframe(clean_table(result.table, max_rows=result.table_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_theme_css()
    initialize_chat()
    data = load_all_data()
    options = render_sidebar(data)

    render_header()
    render_technical_architecture()
    render_kpi_cards(data)
    render_optional_panels(data, options)
    
    # Render Dashboard Tabs
    chat_tab, visuals_tab, historical_tab, model_tab = st.tabs([
        "💬 Chat Assistant", 
        "📈 Forecast Visuals",
        "📊 Historical Performance",
        "🧠 Model Performance"
    ])

    with chat_tab:
        render_chat_tab(data)
    with visuals_tab:
        render_forecast_visuals_tab(data, show_raw_previews=options["show_preview"])
    with historical_tab:
        render_historical_tab(data)
    with model_tab:
        render_model_performance_tab(data)


if __name__ == "__main__":
    main()