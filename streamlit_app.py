"""Streamlit Community Cloud root entrypoint for FraudShield Analyst Console.

Delegates execution to dashboard/streamlit_app.py so the application launches
seamlessly whether Streamlit Cloud is configured to run `streamlit_app.py`
or `dashboard/streamlit_app.py`.
"""

import runpy
import sys
from pathlib import Path

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

dashboard_path = root_dir / "dashboard" / "streamlit_app.py"
runpy.run_path(str(dashboard_path), run_name="__main__")

