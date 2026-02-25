"""
Environment Variable Management with python-dotenv
===================================================
Reference implementation: loading, validating, and updating .env files
for financial API configurations.

Library: python-dotenv >= 1.0.0
Docs: https://saurabh-kumar.com/python-dotenv/
"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ["POLYGON_API_KEY", "ALPHAVANTAGE_API_KEY"]
OPTIONAL_KEYS = ["TICKERS"]

OPTIONAL_DEFAULTS = {
    "TICKERS": "AAPL,MSFT,GOOGL",
}


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def load_and_validate_env(env_path: str | None = None) -> dict[str, str]:
    """Load a .env file and validate that all required keys are present.

    Parameters
    ----------
    env_path : str or None
        Explicit path to a .env file.  When *None*, python-dotenv searches
        upward from the current working directory.

    Returns
    -------
    dict[str, str]
        Mapping of every required and optional key to its resolved value.

    Raises
    ------
    ValueError
        If one or more required environment variables are missing.
    """
    # load_dotenv() reads the .env file and injects values into os.environ.
    # override=False keeps existing shell exports intact.
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)

    # --- Validate required keys -------------------------------------------
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    # --- Collect all values ------------------------------------------------
    env_values: dict[str, str] = {}

    for key in REQUIRED_KEYS:
        env_values[key] = os.getenv(key)  # guaranteed non-None after check

    # os.getenv() accepts a default for optional keys
    for key in OPTIONAL_KEYS:
        env_values[key] = os.getenv(key, OPTIONAL_DEFAULTS.get(key, ""))

    return env_values


def update_env_key(key: str, value: str, env_path: str = ".env") -> None:
    """Programmatically write or update a single key in the .env file.

    Parameters
    ----------
    key : str
        Environment variable name.
    value : str
        New value to persist.
    env_path : str
        Path to the .env file (created if it does not exist).
    """
    # set_key() handles quoting and escaping automatically.
    success, key_out, value_out = set_key(env_path, key, value)
    if success:
        print(f"Updated {key_out}={value_out} in {env_path}")
    else:
        print(f"Failed to update {key} in {env_path}")


def generate_env_example(output_path: str = ".env.example") -> None:
    """Generate a .env.example template listing every known key.

    The file contains blank placeholders for required keys and
    commented defaults for optional keys so new developers can
    quickly bootstrap their local environment.
    """
    lines: list[str] = [
        "# ==========================================================",
        "# Environment Variables — StockLTSMTransformerQuantum",
        "# ==========================================================",
        "# Copy this file to .env and fill in your API keys.",
        "",
        "# --- Required ---------------------------------------------",
    ]

    for key in REQUIRED_KEYS:
        lines.append(f"{key}=")

    lines.append("")
    lines.append("# --- Optional (defaults shown) -------------------------")

    for key in OPTIONAL_KEYS:
        default = OPTIONAL_DEFAULTS.get(key, "")
        lines.append(f"{key}={default}")

    lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output_path}")


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Generate a template for new contributors
    generate_env_example()

    # 2. Load and validate the real .env
    try:
        config = load_and_validate_env()
        print("Environment loaded successfully.")
        print(f"  POLYGON_API_KEY    = {config['POLYGON_API_KEY'][:6]}...")
        print(f"  ALPHAVANTAGE_API_KEY = {config['ALPHAVANTAGE_API_KEY'][:6]}...")
        print(f"  TICKERS            = {config['TICKERS']}")
    except ValueError as exc:
        print(f"Configuration error: {exc}")

    # 3. Programmatic update (e.g. key rotation script)
    # update_env_key("POLYGON_API_KEY", "new_key_value_here")
