from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALERTS_FILE = PROJECT_ROOT / "artifacts" / "alerts.jsonl"
FAVICON_FILE = PROJECT_ROOT / "assets" / "nids-shield.svg"
GITHUB_URL = "https://github.com/Abhi-3690/AI-NIDS"

st.set_page_config(
    page_title="AI-NIDS | Security Operations Center",
    page_icon=str(FAVICON_FILE) if FAVICON_FILE.is_file() else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# White claymorphism theme. Motion is subtle and respects reduced-motion settings.
st.markdown(
    """
    <style>
    :root {
      color-scheme: light;
      --clay-bg: #f4f7fb;
      --clay-surface: #f8faff;
      --clay-ink: #17243b;
      --clay-muted: #53627a;
      --clay-accent: #315bd6;
      --clay-border: #e2e9f4;
      --clay-shadow-dark: #d8e0ec;
      --clay-shadow-light: #ffffff;
    }
    html, body, [data-testid="stAppViewContainer"] {
      background: var(--clay-bg);
      color: var(--clay-ink);
      max-width: 100%;
      overflow-x: hidden !important;
    }
    [data-testid="stHeader"] {
      background: rgba(244, 247, 251, .92);
    }
    [data-testid="stMainBlockContainer"] {
      max-width: 1440px;
      padding-top: 1.6rem;
      padding-bottom: 2rem;
      padding-left: clamp(1rem, 3vw, 2.8rem);
      padding-right: clamp(1rem, 3vw, 2.8rem);
    }
    h1, h2, h3, p, label, span, div {
      overflow-wrap: anywhere;
    }
    h1, h2, h3 { color: var(--clay-ink); letter-spacing: -0.025em; }
    h1 { font-size: clamp(1.8rem, 4vw, 2.8rem) !important; }
    h2 { font-size: clamp(1.25rem, 2.4vw, 1.75rem) !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--clay-muted); }

    [data-testid="stMetric"],
    [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stDataFrame"],
    div[data-testid="stPlotlyChart"],
    div[data-testid="stAlert"] {
      background: var(--clay-surface);
      border: 1px solid rgba(219, 228, 241, .95);
      border-radius: 22px;
      box-shadow: 8px 8px 18px var(--clay-shadow-dark),
                  -8px -8px 18px var(--clay-shadow-light);
    }
    [data-testid="stMetric"] {
      padding: 1.1rem 1.2rem;
      transition: transform .22s ease, box-shadow .22s ease;
      min-width: 0;
    }
    [data-testid="stMetric"]:hover {
      transform: translateY(-3px) perspective(900px) rotateX(1deg);
      box-shadow: 10px 12px 22px var(--clay-shadow-dark),
                  -7px -7px 17px var(--clay-shadow-light);
    }
    [data-testid="stMetricLabel"] p { color: var(--clay-muted) !important; }
    [data-testid="stMetricValue"] { color: var(--clay-ink) !important; }

    .hero-card {
      position: relative;
      overflow: hidden;
      border: 1px solid #e2e9f4;
      border-radius: 28px;
      padding: clamp(1.2rem, 3vw, 2rem);
      margin: .3rem 0 1.4rem 0;
      background: linear-gradient(135deg, #ffffff 0%, #f0f5ff 100%);
      box-shadow: 12px 12px 26px #d8e0ec, -12px -12px 26px #ffffff;
      animation: reveal-up .55s ease both;
    }
    .hero-eyebrow {
      display: inline-block;
      padding: .35rem .7rem;
      border-radius: 999px;
      background: #e7eeff;
      color: #2448ad;
      font-size: .78rem;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .hero-title {
      margin: .8rem 0 .35rem 0;
      font-size: clamp(1.8rem, 4.5vw, 3rem);
      line-height: 1.12;
      font-weight: 800;
      color: #17243b;
    }
    .hero-copy { margin: 0; max-width: 70ch; color: #53627a; line-height: 1.65; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      border-radius: 999px;
      padding: .4rem .75rem;
      background: #e4f7ec;
      color: #17643a;
      font-size: .85rem;
      font-weight: 700;
    }
    .status-dot {
      width: .5rem; height: .5rem; border-radius: 50%;
      background: #22a45a; display: inline-block;
    }
    div.stButton > button, div.stDownloadButton > button {
      border: 1px solid #dce5f4;
      border-radius: 14px;
      color: #1c3c88;
      background: linear-gradient(145deg, #ffffff, #edf2fb);
      box-shadow: 5px 5px 10px #dce3ed, -4px -4px 9px #ffffff;
      font-weight: 650;
      min-height: 2.7rem;
      transition: transform .18s ease, box-shadow .18s ease;
      width: 100%;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 7px 8px 13px #d5deeb, -4px -4px 10px #ffffff;
      border-color: #b8c9f4;
      color: #183e9c;
    }
    div.stButton > button:focus-visible,
    div.stDownloadButton > button:focus-visible,
    input:focus-visible, button:focus-visible {
      outline: 3px solid #315bd6 !important;
      outline-offset: 3px !important;
    }
    [data-testid="stSidebar"] {
      background: #eef3fa;
      border-right: 1px solid #e0e7f1;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stExpander"] {
      border: 1px solid #e0e7f1;
      border-radius: 16px;
      background: #f8faff;
    }
    .section-kicker {
      margin: .15rem 0 .3rem 0;
      color: #315bd6;
      font-size: .78rem;
      font-weight: 750;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .app-footer {
      border-top: 1px solid #dfe7f2;
      margin-top: 2rem;
      padding-top: 1rem;
      color: #5d6b80;
      font-size: .85rem;
      line-height: 1.65;
    }
    .app-footer a { color: #244dbb; font-weight: 650; text-decoration-thickness: 1px; }
    a { color: #244dbb; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { max-width: 100%; }
    @keyframes reveal-up {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 720px) {
      [data-testid="stMainBlockContainer"] {
        padding: 1rem .75rem 1.5rem .75rem;
      }
      .hero-card { border-radius: 20px; padding: 1.1rem; }
      [data-testid="stMetric"] { padding: .8rem; border-radius: 16px; }
      [data-testid="stSidebar"] { min-width: 0; }
      [data-testid="stDataFrame"] { overflow-x: auto; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
        scroll-behavior: auto !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=5, show_spinner=False)
def load_alerts(file_path: str, modified_time: float, file_size: int) -> tuple[pd.DataFrame, int]:
    """Read newline-delimited JSON safely; return valid records and malformed-line count."""
    records: list[dict[str, Any]] = []
    malformed = 0
    path = Path(file_path)

    if not path.is_file():
        return pd.DataFrame(), malformed

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                    else:
                        malformed += 1
                except json.JSONDecodeError:
                    malformed += 1
    except OSError as exc:
        st.error(f"Could not read the local alert log: {exc}")
        return pd.DataFrame(), malformed

    return pd.DataFrame(records), malformed


def as_probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 <= number <= 1.0:
        return 0.0
    return number


def render_dashboard(auto_refresh: bool) -> None:
    """Render data inside a fragment so optional refresh doesn't reset sidebar controls."""
    if ALERTS_FILE.is_file():
        try:
            stat = ALERTS_FILE.stat()
            alerts_df, malformed_lines = load_alerts(
                str(ALERTS_FILE), stat.st_mtime, stat.st_size
            )
        except OSError as exc:
            st.error(f"Unable to inspect the alert log: {exc}")
            alerts_df, malformed_lines = pd.DataFrame(), 0
    else:
        alerts_df, malformed_lines = pd.DataFrame(), 0

    if malformed_lines:
        st.warning(
            f"{malformed_lines} malformed log line(s) were skipped. "
            "Check the replay process if this keeps happening."
        )

    if alerts_df.empty:
        st.info("No alert records are available yet.")
        st.markdown("Run a local CSV replay to generate model predictions:")
        st.code(
            '$env:PYTHONPATH = "src"\\n'
            'python -m nids.ingestion.flow_replay '
            '--csv "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" '
            "--limit 50",
            language="powershell",
        )
        st.caption("No traffic is captured by this dashboard; it reads the local alert log.")
        return

    if "predicted_class" not in alerts_df.columns:
        st.error("The alert log does not contain the required `predicted_class` field.")
        return

    alerts_df["predicted_class"] = (
        alerts_df["predicted_class"].fillna("Unknown").astype(str)
    )
    if "confidence" not in alerts_df:
        alerts_df["confidence"] = 0.0
    alerts_df["confidence"] = pd.to_numeric(alerts_df["confidence"], errors="coerce")
    alerts_df["confidence"] = alerts_df["confidence"].map(as_probability)

    if "timestamp" in alerts_df:
        alerts_df["timestamp"] = pd.to_datetime(
            alerts_df["timestamp"], errors="coerce", utc=True
        )

    total_alerts = len(alerts_df)
    category_count = alerts_df["predicted_class"].nunique()
    mean_confidence = alerts_df["confidence"].mean()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Recorded alert rows", f"{total_alerts:,}")
    metric_cols[1].metric("Predicted categories", f"{category_count:,}")
    metric_cols[2].metric(
        "Mean model confidence",
        f"{mean_confidence:.1%}" if pd.notna(mean_confidence) else "N/A",
        help="Mean confidence across recorded alert rows, not a model accuracy score.",
    )

    st.markdown('<p class="section-kicker">Explore the event log</p>', unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    categories = sorted(alerts_df["predicted_class"].unique().tolist())

    with left:
        selected_categories = st.multiselect(
            "Predicted class",
            options=categories,
            default=categories,
            help="Choose which predicted classes to display.",
        )
    with right:
        min_confidence = st.slider(
            "Minimum confidence",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%",
        ) / 100.0

    filtered_df = alerts_df[
        alerts_df["predicted_class"].isin(selected_categories)
        & (alerts_df["confidence"] >= min_confidence)
    ].copy()

    if filtered_df.empty:
        st.info("No recorded alerts match these filters. Adjust the class or confidence filter.")
    else:
        st.markdown('<p class="section-kicker">Prediction distribution</p>', unsafe_allow_html=True)
        distribution = (
            filtered_df["predicted_class"]
            .value_counts()
            .rename_axis("Predicted class")
            .reset_index(name="Rows")
        )
        chart_col, summary_col = st.columns([1.4, 1])
        with chart_col:
            st.bar_chart(distribution.set_index("Predicted class"), y="Rows", height=280)
        with summary_col:
            st.dataframe(distribution, width="stretch", hide_index=True)

        st.markdown('<p class="section-kicker">Alert records</p>', unsafe_allow_html=True)
        display_columns = [
            name for name in
            ["timestamp", "predicted_class", "confidence", "source_file", "source_row"]
            if name in filtered_df.columns
        ]
        table = filtered_df[display_columns].copy()
        if "timestamp" in table.columns:
            table = table.sort_values("timestamp", ascending=False, na_position="last")
            table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        if "confidence" in table.columns:
            table["confidence"] = table["confidence"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
            )
        st.dataframe(table, width="stretch", hide_index=True)

        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered alerts (CSV)",
            data=csv_bytes,
            file_name="ai_nids_filtered_alerts.csv",
            mime="text/csv",
            help="Downloads only the filtered records shown above.",
        )

        if "probabilities" in filtered_df.columns:
            st.markdown('<p class="section-kicker">Model probability breakdown</p>', unsafe_allow_html=True)
            options = filtered_df.index.tolist()

            def alert_label(index: int) -> str:
                timestamp = filtered_df.loc[index].get("timestamp")
                timestamp_text = (
                    timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                    if pd.notna(timestamp) else f"record {index}"
                )
                return (
                    f"{timestamp_text} · "
                    f"{filtered_df.loc[index, 'predicted_class']} · "
                    f"{filtered_df.loc[index, 'confidence']:.1%}"
                )

            selected_index = st.selectbox(
                "Select a record",
                options=options,
                format_func=alert_label,
            )
            raw_probabilities = filtered_df.loc[selected_index, "probabilities"]
            if isinstance(raw_probabilities, str):
                try:
                    raw_probabilities = json.loads(raw_probabilities)
                except json.JSONDecodeError:
                    raw_probabilities = {}
            if isinstance(raw_probabilities, dict) and raw_probabilities:
                probability_df = pd.DataFrame(
                    [
                        {"Class": str(label), "Probability": as_probability(prob)}
                        for label, prob in raw_probabilities.items()
                    ]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(
                    probability_df.set_index("Class"),
                    y="Probability",
                    height=320,
                )
            else:
                st.info("No valid probability breakdown is available for this record.")

    st.caption(
        "Interpretation note: these are model predictions. They are not confirmed incidents, "
        "and confidence is not equivalent to accuracy."
    )


st.markdown(
    """
    <section class="hero-card" aria-label="AI-NIDS dashboard introduction">
      <span class="hero-eyebrow">Network security · Local prototype</span>
      <div class="hero-title">AI-NIDS Security Operations Center</div>
      <p class="hero-copy">
        Review model-generated alerts from CICIDS2017 flow replay in a calm,
        accessible workspace. The dashboard reads local JSONL records and does
        not capture live network traffic.
      </p>
      <div style="margin-top:1rem">
        <span class="status-pill"><span class="status-dot"></span>Local alert-log view</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ AI-NIDS")
    st.caption("Security Operations Center")
    st.divider()
    auto_refresh = st.toggle(
        "Auto-refresh every 10 seconds",
        value=False,
        help="Re-read the local alert log periodically while this page is open.",
    )
    if st.button("Refresh now", width="stretch"):
        load_alerts.clear()
        st.toast("Alert log cache cleared.", icon="🔄")
        st.rerun()
    st.markdown("---")
    st.markdown(f"[View source repository ↗]({GITHUB_URL})")
    st.caption("No account or personal information is requested by this dashboard.")

# Use a fragment to refresh the data region without resetting the rest of the page.
dashboard_fragment = st.fragment(
    run_every="10s" if auto_refresh else None
)(render_dashboard)
with st.spinner("Loading local alert records…"):
    dashboard_fragment(auto_refresh)

with st.expander("Privacy, usage and limitations"):
    st.markdown(
        """
        - **Data source:** this app reads `artifacts/alerts.jsonl` from the project directory.
        - **Local processing:** this prototype does not ask for a name, email address, phone number,
          account, or payment details.
        - **No cookies or analytics configured by this project:** Streamlit itself may use
          essential session mechanisms to operate. Review your deployment platform if hosted.
        - **Security use:** predictions are for evaluation and investigation, not automated
          blocking or a guarantee that traffic is safe.
        - **Retention:** the application does not delete or rotate the local alert log. Manage
          generated logs according to your own environment's retention requirements.
        """
    )

st.markdown(
    f"""
    <footer class="app-footer">
      <strong>AI-NIDS</strong> · Research and educational prototype · © 2026
      <br>
      <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">Source repository</a>
      · No commercial transactions or customer support are provided by this prototype.
      <br>
      Keyboard accessible controls · Reduced-motion preference respected · No third-party images or fonts.
    </footer>
    """,
    unsafe_allow_html=True,
)
