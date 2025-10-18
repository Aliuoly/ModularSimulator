"""Module entry-point for launching the dashboard via ``python -m``."""

from __future__ import annotations

from flask import Flask

from .app import create_app


def main() -> Flask:
    """Create and run the dashboard application."""

    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
    return app


if __name__ == "__main__":  # pragma: no cover - helper for scripted execution
    main()
