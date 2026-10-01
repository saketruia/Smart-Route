"""``python -m simulator.validate`` entry point."""
import sys

from .cli import validate_main

if __name__ == "__main__":
    sys.exit(validate_main())
