from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np

# Imports of the example module are performed inside tests to ensure
# the repository root is on sys.path in all environments.

# Bootstrap test path for module imports without importing before imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_connection_layer_two_tank_cascade_tracks_ad_hoc_coupled_rhs() -> None:
    # Local import to ensure import path setup is handled at runtime
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from examples.connection_layer.two_tank_reaction_comparison import TankCascadeConfig, run_demo

    result = run_demo(TankCascadeConfig(horizon_s=5.0, dt_s=0.05))

    assert result.max_abs_error_a <= 1.0e-7
    assert result.max_abs_error_b <= 2.5e-3
    assert result.rmse_b <= 2.0e-3

    tank_b_connection = np.array(result.tank_b_connection)
    tank_b_reference = np.array(result.tank_b_reference)
    assert abs(float(tank_b_connection[-1] - tank_b_reference[-1])) <= 2.0e-3


def test_two_tank_cascade_shows_expected_decay_and_lag_trends() -> None:
    # Local import to ensure import path setup is handled at runtime
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from examples.connection_layer.two_tank_reaction_comparison import TankCascadeConfig, run_demo

    result = run_demo(TankCascadeConfig(horizon_s=5.0, dt_s=0.05))
    tank_a = np.array(result.tank_a_connection)
    tank_b = np.array(result.tank_b_connection)

    assert result.monotonic_decay_a
    assert result.monotonic_decay_b
    assert result.b_lag_steps >= 1

    assert abs(float(tank_a[0] - 1.0)) <= 1.0e-12
    assert abs(float(tank_b[0] - 1.0)) <= 1.0e-12
    assert tank_a[-1] < tank_a[0]
    assert tank_b[-1] < tank_b[0]

    assert tank_b[4] > tank_a[4]


def test_example_uses_network_binding_path_without_adapter_or_private_stream_mutation() -> None:
    # Local import to ensure import path setup is handled at runtime
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = import_module("examples.connection_layer.two_tank_reaction_comparison")
    module_file = module.__file__
    assert module_file is not None
    source = Path(module_file).read_text(encoding="utf-8")

    assert "ConnectionNetwork" in source
    assert "ProcessModelAdapter" not in source
    assert "_input_streams" not in source
    assert "_output_streams" not in source
