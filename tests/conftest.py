# tests/conftest.py

import sys
import os

# Add src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            print(f"✅ {report.nodeid} passed.")
        elif report.failed:
            print(f"❌ {report.nodeid} failed.")
        elif report.skipped:
            print(f"⚠️ {report.nodeid} skipped.")
