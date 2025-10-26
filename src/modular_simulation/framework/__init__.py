"""Backward compatibility shim for the legacy framework package."""
from modular_simulation.core import System, create_system

__all__ = ["System", "create_system"]
