
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALERTS_FILE = PROJECT_ROOT / "artifacts" / "alerts.jsonl"

st.set_page_config(
    page_title="AI-NIDS | Security Operations Center",
    page_icon="🛡️",
    layout="wide",
)

# Dashboard styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }
    [data-testid="stMetric"] {
        background-color: #111c30;
        border: 1px solid #26354d;
        padding: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=5)
def load_alerts(file_path, modified_time):
    records = []

    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


st.title("🛡️ AI-NIDS Security Operations Center")
st.caption("Network intrusion detection • CICIDS2017 flow replay • Model predictions")

# Sidebar
st.sidebar.header("Dashboard Controls")
st.sidebar.caption("Data source: artifacts/alerts.jsonl")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=False)

if auto_refresh:
    st.sidebar.info(
        "To refresh automatically, install streamlit-autorefresh "
        "or refresh this browser page manually."
    )

# Load alert log
if ALERTS_FILE.exists():
    alerts_df = load_alerts(
        str(ALERTS_FILE),
        ALERTS_FILE.stat().st_mtime,
    )
else:
    alerts_df = pd.DataFrame()

if alerts_df.empty:
    st.warning("No alerts found yet. Run flow replay to generate predictions.")
    st.code(
        ' $env:PYTHONPATH = "src"\n'
        'python -m nids.ingestion.flow_replay '
        '--csv "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" '
        "--limit 20",
        language="powershell",
    )
    st.stop()

# Normalize fields for display
if "predicted_class" not in alerts_df.columns:
    st.error("Alert log is missing the predicted_class field.")
    st.stop()

alerts_df["predicted_class"] = (
    alerts_df["predicted_class"].fillna("Unknown").astype(str)
)

if "confidence" in alerts_df.columns:
    alerts_df["confidence"] = pd.to_numeric(
        alerts_df["confidence"], errors="coerce"
    )
else:
    alerts_df["confidence"] = 0.0

if "timestamp" in alerts_df.columns:
    alerts_df["timestamp"] = pd.to_datetime(
        alerts_df["timestamp"], errors="coerce", utc=True
    )

# Metrics
total_alerts = len(alerts_df)
unique_attacks = alerts_df["predicted_class"].nunique()
average_confidence = alerts_df["confidence"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Recorded Alerts", f"{total_alerts:,}")
col2.metric("Predicted Categories", f"{unique_attacks:,}")
col3.metric(
    "Average Model Confidence",
    f"{average_confidence:.1%}" if pd.notna(average_confidence) else "N/A",
)

st.divider()

# Filters
st.subheader("🔎 Filter Alerts")

filter_col1, filter_col2 = st.columns(2)

categories = sorted(alerts_df["predicted_class"].unique().tolist())

with filter_col1:
    selected_categories = st.multiselect(
        "Predicted category",
        options=categories,
        default=categories,
    )

with filter_col2:
    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    ) / 100

filtered_df = alerts_df[
    alerts_df["predicted_class"].isin(selected_categories)
    & (alerts_df["confidence"].fillna(0) >= min_confidence)
].copy()

# Attack distribution
st.subheader("📊 Predicted Category Distribution")

if filtered_df.empty:
    st.info("No alerts match the selected filters.")
else:
    distribution = (
        filtered_df["predicted_class"]
        .value_counts()
        .rename_axis("Predicted category")
        .reset_index(name="Count")
    )

    chart_col, table_col = st.columns([3, 2])

    with chart_col:
        st.bar_chart(
            distribution.set_index("Predicted category"),
            y="Count",
        )

    with table_col:
        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True,
        )

# Detailed alert table
st.subheader("🚨 Alert Records")

display_columns = [
    column
    for column in [
        "timestamp",
        "predicted_class",
        "confidence",
        "source_file",
        "source_row",
    ]
    if column in filtered_df.columns
]

display_df = filtered_df[display_columns].copy()

if "confidence" in display_df.columns:
    display_df["confidence"] = display_df["confidence"].map(
        lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
    )

st.dataframe(
    display_df.sort_values(
        "timestamp", ascending=False, na_position="last"
    ) if "timestamp" in display_df.columns else display_df,
    use_container_width=True,
    hide_index=True,
)

# Probability details
st.subheader("🧠 Prediction Probability Details")

if "probabilities" in filtered_df.columns:
    selected_index = st.selectbox(
        "Select an alert record",
        options=filtered_df.index.tolist(),
        format_func=lambda idx: (
            f"Row {idx}: "
            f"{filtered_df.loc[idx, 'predicted_class']} "
            f"({filtered_df.loc[idx, 'confidence']:.1%})"
        ),
    )

    probabilities = filtered_df.loc[selected_index, "probabilities"]

    if isinstance(probabilities, dict) and probabilities:
        probability_df = pd.DataFrame(
            probabilities.items(),
            columns=["Class", "Probability"],
        )
        probability_df["Probability"] = pd.to_numeric(
            probability_df["Probability"], errors="coerce"
        )
        probability_df = probability_df.dropna().sort_values(
            "Probability", ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Class"),
            y="Probability",
        )
    else:
        st.info("No probability breakdown is available for this record.")

st.caption(
    "Prototype dashboard: predictions are model outputs, not confirmed "
    "security incidents. This view displays recorded alerts only."
)
