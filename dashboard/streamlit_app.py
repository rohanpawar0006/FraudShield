"""FraudShield Analyst Investigation Dashboard & Feedback Loop.

Streamlit application providing:
1. Real-time alert triage & filtering
2. Case investigation with XGBoost Tree SHAP explanations, account history & graph context
3. Analyst feedback submission with persistent audit trail
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.common.config import settings  # noqa: E402
from src.feedback.service import (  # noqa: E402
    VALID_DISPOSITIONS,
    get_account_history,
    get_alerts_summary_kpis,
    get_case,
    get_feature_contributions,
    list_alerts,
    submit_feedback,
)

# Page configuration
st.set_page_config(
    page_title="FraudShield Analyst Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_graph_resources():
    """Load node mapping and rings summary into memory once."""
    mapping = {}
    if Path(settings.GRAPH_NODE_MAPPING_PATH).exists():
        try:
            mapping = joblib.load(settings.GRAPH_NODE_MAPPING_PATH)
        except Exception as e:
            st.warning(f"Failed to load graph node mapping: {e}")

    rings = {}
    if Path(settings.GRAPH_RINGS_PATH).exists():
        try:
            with open(settings.GRAPH_RINGS_PATH, "r") as f:
                rings = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load graph rings summary: {e}")

    return mapping, rings


def get_account_graph_context(
    account_id: str, mapping: Dict[str, int], rings: Dict[str, Any]
) -> Dict[str, Any]:
    """Retrieve training-window graph context for an account."""
    if not mapping or account_id not in mapping:
        return {
            "in_training_graph": False,
            "node_index": None,
            "is_fan_in_hub": False,
            "is_mule": False,
        }

    node_idx = mapping[account_id]
    hubs = rings.get("sample_fan_in_hubs", [])
    mules = rings.get("sample_forwarding_mules", [])

    return {
        "in_training_graph": True,
        "node_index": node_idx,
        "is_fan_in_hub": any(
            isinstance(h, dict) and h.get("account") == account_id for h in hubs
        ),
        "is_mule": account_id in mules,
    }


def seed_sample_alerts() -> int:
    """Seed demo alerts into database for demonstration and cloud environments."""
    import time

    from src.common.db import db_manager
    from src.serving.schemas import TransactionPayload
    from src.serving.scoring import ScoringEngine

    sample_txns = [
        {"step": 379, "type": "TRANSFER", "amount": 1250000.0, "nameOrig": "C1729268548", "oldbalanceOrg": 1250000.0, "nameDest": "C1519453344", "oldbalanceDest": 0.0},
        {"step": 380, "type": "CASH_OUT", "amount": 1250000.0, "nameOrig": "C1519453344", "oldbalanceOrg": 1250000.0, "nameDest": "C961243143", "oldbalanceDest": 45000.0},
        {"step": 381, "type": "TRANSFER", "amount": 7539165.76, "nameOrig": "C64015154", "oldbalanceOrg": 7539165.76, "nameDest": "C1744865437", "oldbalanceDest": 0.0},
        {"step": 382, "type": "CASH_OUT", "amount": 4417191.86, "nameOrig": "C790582807", "oldbalanceOrg": 4417191.86, "nameDest": "C690378183", "oldbalanceDest": 12000.0},
        {"step": 383, "type": "TRANSFER", "amount": 157277.0, "nameOrig": "C1601578028", "oldbalanceOrg": 157277.0, "nameDest": "C49314331", "oldbalanceDest": 0.0},
        {"step": 384, "type": "CASH_OUT", "amount": 176626.77, "nameOrig": "C2077002700", "oldbalanceOrg": 176626.77, "nameDest": "C1412132938", "oldbalanceDest": 0.0},
        {"step": 385, "type": "PAYMENT", "amount": 1420.50, "nameOrig": "C1234567890", "oldbalanceOrg": 5400.0, "nameDest": "M987654321", "oldbalanceDest": 0.0},
        {"step": 386, "type": "CASH_IN", "amount": 50000.0, "nameOrig": "C9988776655", "oldbalanceOrg": 12000.0, "nameDest": "C1122334455", "oldbalanceDest": 65000.0},
        {"step": 387, "type": "PAYMENT", "amount": 89.99, "nameOrig": "C5544332211", "oldbalanceOrg": 2300.0, "nameDest": "M1122334455", "oldbalanceDest": 0.0},
        {"step": 388, "type": "TRANSFER", "amount": 438859.10, "nameOrig": "C1632082016", "oldbalanceOrg": 438859.10, "nameDest": "C1877779310", "oldbalanceDest": 0.0},
        {"step": 389, "type": "CASH_OUT", "amount": 438859.10, "nameOrig": "C1877779310", "oldbalanceOrg": 438859.10, "nameDest": "C2233445566", "oldbalanceDest": 15000.0},
        {"step": 390, "type": "DEBIT", "amount": 320.0, "nameOrig": "C4433221100", "oldbalanceOrg": 4500.0, "nameDest": "C6655443322", "oldbalanceDest": 18000.0},
        {"step": 391, "type": "PAYMENT", "amount": 1250.0, "nameOrig": "C7766554433", "oldbalanceOrg": 8900.0, "nameDest": "M3344556677", "oldbalanceDest": 0.0},
        {"step": 392, "type": "TRANSFER", "amount": 190662.70, "nameOrig": "C1569153511", "oldbalanceOrg": 190662.70, "nameDest": "C1637089162", "oldbalanceDest": 0.0},
        {"step": 393, "type": "CASH_OUT", "amount": 190662.70, "nameOrig": "C1637089162", "oldbalanceOrg": 190662.70, "nameDest": "C9988112233", "oldbalanceDest": 8200.0},
    ]

    try:
        engine = ScoringEngine()
        seeded = 0
        for i, item in enumerate(sample_txns):
            txn_id = f"demo_txn_{i:04d}"
            payload = TransactionPayload(
                transaction_id=txn_id,
                step=item["step"],
                type=item["type"],
                amount=item["amount"],
                nameOrig=item["nameOrig"],
                oldbalanceOrg=item["oldbalanceOrg"],
                nameDest=item["nameDest"],
                oldbalanceDest=item["oldbalanceDest"],
            )
            resp = engine.score(payload)
            db_manager.save_transaction_and_score(
                txn_id=payload.txn_id,
                account_id=payload.nameOrig,
                dest_id=payload.nameDest,
                amount=payload.amount,
                txn_type=payload.type,
                ts=None,
                raw_payload=payload.model_dump(),
                xgb_score=resp.xgb_score,
                gnn_score=resp.gnn_score,
                blended_score=resp.fraud_score,
                model_version=resp.model_version,
                latency_ms=resp.latency_ms,
            )
            seeded += 1
        time.sleep(0.3)
        return seeded
    except Exception as e:
        st.sidebar.caption(f"Demo seeding note: {e}")
        return 0


# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Alerts"
if "selected_txn" not in st.session_state:
    st.session_state.selected_txn = None
if "alerts_page" not in st.session_state:
    st.session_state.alerts_page = 0

# Ensure database has alerts in cloud/demo environments
from src.common.db import db_manager  # noqa: E402

try:
    conn = db_manager.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM scores")
    current_score_count = cur.fetchone()[0]
    conn.close()
    if current_score_count == 0:
        seed_sample_alerts()
except Exception:
    pass


# Sidebar Navigation
st.sidebar.title("🛡️ FraudShield Console")
st.sidebar.markdown(f"**Environment:** `{settings.ENV}`")
st.sidebar.markdown(f"**Decision Threshold:** `{settings.DECISION_THRESHOLD}`")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["Alerts", "Case Investigation", "Feedback Action"],
    index=["Alerts", "Case Investigation", "Feedback Action"].index(
        st.session_state.current_page
        if st.session_state.current_page in ["Alerts", "Case Investigation", "Feedback Action"]
        else "Alerts"
    ),
)
st.session_state.current_page = nav_choice

st.sidebar.markdown("---")
if st.sidebar.button("⚡ Reload Sample Alerts"):
    with st.spinner("Populating sample alerts..."):
        seed_sample_alerts()
    st.sidebar.success("Sample alerts refreshed!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Quick Links**")
st.sidebar.markdown("- [GitHub Repository](https://github.com/rohanpawar0006/FraudShield)")
st.sidebar.markdown("- [API Documentation](https://github.com/rohanpawar0006/FraudShield/blob/main/docs/api/API.md)")
st.sidebar.markdown("- [Model Performance](https://github.com/rohanpawar0006/FraudShield/blob/main/docs/results/RESULTS.md)")


# =============================================================================
# View 1: Alerts View
# =============================================================================
if st.session_state.current_page == "Alerts":
    st.title("🚨 Fraud Risk Alerts Triage")
    st.caption("Real-time high-risk transaction alerts generated by FraudShield scoring engine.")

    # Header KPIs
    kpis = get_alerts_summary_kpis()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts", kpis["total_alerts"])
    col2.metric("Reviewed %", f"{kpis['pct_reviewed']}%")
    col3.metric("Confirmed Fraud", kpis["confirmed_count"])
    col4.metric("False Positive Rate", f"{kpis['false_positive_rate']}%")

    st.markdown("---")

    # Filters
    st.subheader("Filter Alerts")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])

    with f_col1:
        min_score = st.slider(
            "Minimum Score",
            min_value=0.50,
            max_value=1.00,
            value=float(settings.DECISION_THRESHOLD),
            step=0.01,
        )

    with f_col2:
        status_filter = st.selectbox(
            "Status",
            ["All", "New", "Under Review", "Confirmed Fraud", "False Positive"],
            index=0,
        )

    with f_col3:
        type_filter = st.multiselect(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            default=[],
        )

    # Query alerts
    status_arg = None if status_filter == "All" else status_filter
    type_arg = type_filter[0] if len(type_filter) == 1 else None

    page_limit = 50
    offset = st.session_state.alerts_page * page_limit
    alerts = list_alerts(
        status=status_arg,
        min_score=min_score,
        txn_type=type_arg,
        limit=page_limit,
        offset=offset,
    )

    # If multiple types selected, filter locally
    if len(type_filter) > 1:
        alerts = [a for a in alerts if a.get("type") in type_filter]

    st.subheader(f"Alerts Queue (Showing up to {page_limit} items)")

    if not alerts:
        st.info("No alerts found matching filter criteria. Try lowering the score slider or run the seed script.")
    else:
        # Table display
        table_rows = []
        for a in alerts:
            table_rows.append(
                {
                    "Transaction ID": a.get("txn_id"),
                    "Timestamp": a.get("ts"),
                    "Amount ($)": f"{a.get('amount', 0.0):,.2f}",
                    "Type": a.get("type"),
                    "Blended Score": f"{a.get('blended_score', 0.0):.4f}",
                    "Status": a.get("status", "New"),
                }
            )

        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Transaction selection to investigate
        txn_ids = [a.get("txn_id") for a in alerts]
        selected = st.selectbox("Select Transaction to Investigate:", txn_ids)
        if st.button("🔍 Open Case Investigation", key="btn_open_case"):
            st.session_state.selected_txn = selected
            st.session_state.current_page = "Case Investigation"
            st.rerun()

    # Pagination controls
    st.markdown("---")
    p_col1, p_col2, p_col3 = st.columns([1, 1, 4])
    with p_col1:
        if st.button("⬅️ Previous 50", disabled=(st.session_state.alerts_page == 0)):
            st.session_state.alerts_page = max(0, st.session_state.alerts_page - 1)
            st.rerun()
    with p_col2:
        if st.button("Next 50 ➡️", disabled=(len(alerts) < page_limit)):
            st.session_state.alerts_page += 1
            st.rerun()
    with p_col3:
        st.caption(f"Current Page: {st.session_state.alerts_page + 1}")


# =============================================================================
# View 2: Case Investigation View
# =============================================================================
elif st.session_state.current_page == "Case Investigation":
    st.title("🔬 Case Investigation")
    st.caption("Deep-dive transaction forensic analysis, model explainability, and counterparty context.")

    # Transaction Selector
    c_col1, c_col2 = st.columns([3, 1])
    with c_col1:
        txn_input = st.text_input(
            "Transaction ID",
            value=st.session_state.selected_txn or "",
            placeholder="e.g. seed_txn_10838",
        )
    with c_col2:
        if st.button("Search Case"):
            st.session_state.selected_txn = txn_input
            st.rerun()

    if not txn_input:
        st.info("Select an alert from the Alerts view or enter a transaction ID above to begin investigation.")
    else:
        case = get_case(txn_input)
        if not case:
            st.error(f"Transaction `{txn_input}` not found in database.")
        else:
            st.session_state.selected_txn = txn_input

            # Parse raw payload
            raw_data = case.get("raw")
            if isinstance(raw_data, str):
                try:
                    raw_data = json.loads(raw_data)
                except Exception:
                    raw_data = {}
            elif not isinstance(raw_data, dict):
                raw_data = {}

            # Status Banner
            status = case.get("status", "New")
            status_colors = {
                "New": "gray",
                "Under Review": "orange",
                "Confirmed Fraud": "red",
                "False Positive": "green",
            }
            st.markdown(
                f"### Case Status: :{status_colors.get(status, 'gray')}[{status}] "
                f"(Assigned: `{case.get('assigned_to') or 'Unassigned'}` | Updated: `{case.get('updated_at')}`)"
            )

            # 1. Transaction Details Section
            st.markdown("#### 1. Transaction Details")
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            d_col1.metric("Amount", f"${case.get('amount', 0.0):,.2f}")
            d_col2.metric("Type", case.get("type"))
            d_col3.metric("Timestamp", case.get("ts"))
            d_col4.metric("Step / Hour", raw_data.get("step", "N/A"))

            d_col5, d_col6 = st.columns(2)
            with d_col5:
                st.markdown(f"**Origin Account:** `{case.get('account_id')}`")
                st.markdown(f"Old Balance: `${raw_data.get('oldbalanceOrg', 0.0):,.2f}`")
                st.markdown(f"New Balance: `${raw_data.get('newbalanceOrig', 0.0):,.2f}`")
            with d_col6:
                st.markdown(f"**Destination Account:** `{case.get('dest_id')}`")
                st.markdown(f"Old Balance: `${raw_data.get('oldbalanceDest', 0.0):,.2f}`")
                st.markdown(f"New Balance: `${raw_data.get('newbalanceDest', 0.0):,.2f}`")

            st.markdown("---")

            # 2. Model Scores & Explanations Section
            st.markdown("#### 2. Model Scores & Explainability")
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Blended Score", f"{case.get('blended_score', 0.0):.4f}")
            s_col2.metric("XGBoost Score", f"{case.get('xgb_score', 0.0):.4f}")
            s_col3.metric(
                "GNN Score",
                f"{case.get('gnn_score'):.4f}" if case.get("gnn_score") is not None else "N/A",
            )
            s_col4.metric("Latency", f"{case.get('latency_ms', 0.0):.2f} ms")

            # Waterfall / Horizontal Bar Chart of Feature Contributions
            st.markdown("##### Top 8 Contributing Features (Tree SHAP / pred_contribs)")
            contribs = get_feature_contributions(raw_data, top_n=8)
            if contribs:
                contrib_df = pd.DataFrame(contribs)
                # Sort ascending for horizontal bar chart
                contrib_df = contrib_df.sort_values(by="contribution", ascending=True)

                fig, ax = plt.subplots(figsize=(9, 4.2))
                colors = ["#dc3545" if c > 0 else "#28a745" for c in contrib_df["contribution"]]
                ax.barh(contrib_df["feature"], contrib_df["contribution"], color=colors)
                ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Contribution to Fraud Score (Log-Odds Shift)")
                ax.set_title("XGBoost Tree SHAP Feature Attribution (Red = Increased Risk, Green = Reduced Risk)")
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Feature contributions could not be calculated for this payload.")

            st.markdown("---")

            # 3. Account History Section
            st.markdown("#### 3. Account History (Serving Database)")
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.markdown(f"**Origin Account History (`{case.get('account_id')}`):**")
                orig_history = get_account_history(case.get("account_id"), limit=5)
                if orig_history:
                    st.dataframe(pd.DataFrame(orig_history), use_container_width=True, hide_index=True)
                else:
                    st.caption("No other recorded transactions for origin account (ephemeral sender).")

            with h_col2:
                st.markdown(f"**Destination Account History (`{case.get('dest_id')}`):**")
                dest_history = get_account_history(case.get("dest_id"), limit=5)
                if dest_history:
                    st.dataframe(pd.DataFrame(dest_history), use_container_width=True, hide_index=True)
                else:
                    st.caption("No other recorded transactions for destination account.")

            st.markdown("---")

            # 4. Graph Context Section
            st.markdown("#### 4. Training Graph Counterparty Context")
            mapping, rings = load_graph_resources()
            orig_graph = get_account_graph_context(case.get("account_id"), mapping, rings)
            dest_graph = get_account_graph_context(case.get("dest_id"), mapping, rings)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown(f"**Origin Account:** `{case.get('account_id')}`")
                st.markdown(f"- In Training Graph: `{orig_graph['in_training_graph']}`")
                if orig_graph["in_training_graph"]:
                    st.markdown(f"- Node Index: `{orig_graph['node_index']}`")
                st.markdown(f"- Fan-In Hub Flag: `{orig_graph['is_fan_in_hub']}`")
                st.markdown(f"- Forwarding Mule Flag: `{orig_graph['is_mule']}`")

            with g_col2:
                st.markdown(f"**Destination Account:** `{case.get('dest_id')}`")
                st.markdown(f"- In Training Graph: `{dest_graph['in_training_graph']}`")
                if dest_graph["in_training_graph"]:
                    st.markdown(f"- Node Index: `{dest_graph['node_index']}`")
                st.markdown(f"- Fan-In Hub Flag: `{dest_graph['is_fan_in_hub']}`")
                st.markdown(f"- Forwarding Mule Flag: `{dest_graph['is_mule']}`")

            st.markdown("---")

            # 5. Direct Feedback Submission & Audit Trail
            st.markdown("#### 5. Analyst Feedback & Audit Trail")
            with st.form(key=f"feedback_form_{txn_input}"):
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    disposition = st.radio(
                        "Disposition",
                        ["Confirmed Fraud", "False Positive", "Under Review"],
                        index=0 if status not in VALID_DISPOSITIONS else list(VALID_DISPOSITIONS).index(status),
                    )
                with fb_col2:
                    analyst_name = st.text_input("Analyst Name", value="analyst_alice")

                notes = st.text_area("Investigation Notes / Evidence", placeholder="Reasoning for disposition...")
                submit_btn = st.form_submit_button("💾 Submit Disposition")

                if submit_btn:
                    if not analyst_name.strip():
                        st.error("Analyst Name is required.")
                    else:
                        submit_feedback(
                            transaction_id=txn_input,
                            disposition=disposition,
                            analyst=analyst_name.strip(),
                            note=notes.strip() if notes else None,
                        )
                        st.success(f"Feedback successfully logged: `{disposition}` by `{analyst_name}`.")
                        st.rerun()

            # Audit Trail Table
            audit_trail = case.get("audit_trail", [])
            if audit_trail:
                st.markdown("##### Historical Audit Trail")
                st.dataframe(pd.DataFrame(audit_trail), use_container_width=True, hide_index=True)


# =============================================================================
# View 3: Feedback Action View
# =============================================================================
elif st.session_state.current_page == "Feedback Action":
    st.title("✍️ Dedicated Analyst Feedback Action")
    st.caption("Submit or update case dispositions with full audit trail compliance.")

    selected_txn = st.text_input(
        "Transaction ID to Review",
        value=st.session_state.selected_txn or "",
        placeholder="e.g. seed_txn_10838",
    )

    if not selected_txn:
        st.info("Enter a transaction ID or choose one from the Alerts view.")
    else:
        case = get_case(selected_txn)
        if not case:
            st.error(f"Transaction `{selected_txn}` not found.")
        else:
            st.write(
                f"**Transaction:** `{selected_txn}` | **Amount:** `${case.get('amount', 0.0):,.2f}` | "
                f"**Score:** `{case.get('blended_score', 0.0):.4f}` | **Current Status:** `{case.get('status')}`"
            )

            with st.form(key="standalone_feedback_form"):
                disposition = st.radio("Review Disposition", ["Confirmed Fraud", "False Positive", "Under Review"])
                analyst = st.text_input("Analyst Name", value="analyst_bob")
                notes = st.text_area("Notes", placeholder="Add disposition details...")
                submitted = st.form_submit_button("Submit Review")

                if submitted:
                    if not analyst.strip():
                        st.error("Analyst Name is required.")
                    else:
                        submit_feedback(
                            transaction_id=selected_txn,
                            disposition=disposition,
                            analyst=analyst.strip(),
                            note=notes.strip() if notes else None,
                        )
                        st.success(f"Submitted {disposition} for {selected_txn}!")
                        st.rerun()

            audit_trail = case.get("audit_trail", [])
            if audit_trail:
                st.subheader("Historical Audit Trail")
                st.dataframe(pd.DataFrame(audit_trail), use_container_width=True, hide_index=True)
