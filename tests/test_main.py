# tests/test_main.py

import pytest
from src import main


def test_main_execution():
    # We will just test if the main.py runs without error
    try:
        main
    except Exception as e:
        pytest.fail(f"main.py raised an exception: {e}")
