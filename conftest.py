"""pytest bootstrap.

Puts the repo root on sys.path so tests under tests/ can import the top-level
modules (db, ingestion, llm_service, ...) without an editable install. pytest
imports the nearest conftest.py before collecting tests, so this runs first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
