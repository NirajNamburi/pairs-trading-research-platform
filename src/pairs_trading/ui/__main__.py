"""``python -m pairs_trading.ui [--reports DIR]`` starts the Streamlit viewer."""

import subprocess
import sys
from pathlib import Path


def build_command(argv: list[str]) -> list[str]:
    """The ``streamlit run`` command that forwards ``argv`` to the app.

    The ``--`` separator is inserted here, so a leading ``--`` typed by the user
    (``python -m pairs_trading.ui -- --reports DIR``) is dropped: forwarded verbatim it would
    reach the app's argument parser as an end-of-options marker and ``--reports`` would be
    silently ignored.
    """
    args = list(argv)
    if args[:1] == ["--"]:
        args = args[1:]
    app = Path(__file__).with_name("app.py")
    return [sys.executable, "-m", "streamlit", "run", str(app), "--", *args]


def main(argv: list[str] | None = None) -> int:
    return subprocess.call(build_command(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
