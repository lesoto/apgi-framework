"""Thin shim — delegates to the apgi.scripts.fetch_data CLI entry point.

Prefer the installed command:  apgi-fetch [--list] [--dataset NAME]

Direct invocation (without install):
    python scripts/fetch_data.py [--list] [--dataset NAME]
"""

from apgi.scripts.fetch_data import main

if __name__ == "__main__":
    main()
