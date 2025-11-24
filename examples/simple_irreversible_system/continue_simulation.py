"""Demonstrate saving, loading, and resuming the irreversible system simulation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from modular_simulation.framework import System
from modular_simulation.utils.wrappers import day, second
from component_definition import control_elements, calculations, sensors
from process_definition import IrreversibleProcessModel


CHECKPOINT_PATH = Path(__file__).with_name("checkpoint_irreversible.json")


def build_system() -> System:
    dt = second(30)
    return System(
        dt=dt,
        process_model=IrreversibleProcessModel(),
        record_history=True,
        sensors=sensors,
        calculations=calculations,
        control_elements=control_elements,
    )


def save_checkpoint(system: System, path: Path) -> None:
    payload = system.save()
    path.write_text(json.dumps(payload, indent=2))


def load_checkpoint(path: Path) -> System:
    payload = json.loads(path.read_text())
    return System.load(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Starting initial simulation run...")
    system = build_system()
    system.step(day(1))

    logging.info("Saving checkpoint to %s", CHECKPOINT_PATH)
    save_checkpoint(system, CHECKPOINT_PATH)

    logging.info("Reloading system from checkpoint...")
    restored_system = load_checkpoint(CHECKPOINT_PATH)

    logging.info("Continuing the simulation after reload...")
    restored_system.extend_controller_sp_trajectory(cv_tag="B", value=0.2)
    restored_system.step(day(1))
    restored_system.extend_controller_sp_trajectory(cv_tag="B").ramp(0.3, rate=0.1 / day(1))
    restored_system.step(day(1))

    logging.info("Simulation resumed to t = %.2f days", restored_system.time / day(1))
    logging.info("Current volume V = %.3f L", restored_system.process_model.V)
    logging.info("Current concentration B = %.4f mol/L", restored_system.process_model.B)


if __name__ == "__main__":
    main()
