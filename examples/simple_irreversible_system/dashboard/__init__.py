"""Dashboard application for the simple irreversible system example."""

from .simulation import SimulationManager

try:  # pragma: no cover - optional Flask dependency
    from .app import create_app
except ModuleNotFoundError as exc:  # pragma: no cover - fallback when Flask unavailable
    def create_app(*args, **kwargs):  # type: ignore[override]
        raise ModuleNotFoundError(
            "Flask is required to use the dashboard application. Install it via `pip install flask`."
        ) from exc

__all__ = ["create_app", "SimulationManager"]
