"""Smoke test for Streamlit analyst dashboard using streamlit.testing.v1.AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_dashboard_smoke():
    """Verify that dashboard loads without uncaught exceptions and renders expected widgets."""
    app_path = (Path(__file__).resolve().parent.parent.parent / "dashboard" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # 1. No uncaught exceptions thrown during script execution
    assert not at.exception, f"Dashboard threw unexpected exception: {at.exception}"

    # 2. Key navigation radio exists in sidebar
    assert len(at.sidebar.radio) >= 1
    assert "Alerts" in at.sidebar.radio[0].options

    # 3. Main header and KPI metrics render
    assert len(at.title) >= 1
    assert "Alerts" in at.title[0].value or "Fraud Risk" in at.title[0].value
    assert len(at.metric) >= 4  # Total Alerts, Reviewed %, Confirmed Fraud, False Positive Rate
