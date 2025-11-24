#!/usr/bin/env python3
"""Test logging output when server starts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.logging import setup_logging
from mcp_server.server import build_registry

setup_logging("INFO")

print("=" * 60)
print("Testing Server Logging")
print("=" * 60)
print()

registry = build_registry()

print()
print("=" * 60)
print("Registry built - check logs above for:")
print("- KB status")
print("- Eval status")
print("- Interp status")
print("- Governance status")
print("- Tool count")
print("=" * 60)

