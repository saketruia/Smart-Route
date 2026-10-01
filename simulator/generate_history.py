"""``python -m simulator.generate_history`` entry point."""
import sys

from .cli import history_main

if __name__ == "__main__":
    sys.exit(history_main())
