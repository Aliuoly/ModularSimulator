import pytest
from pydantic import ValidationError

from modular_simulation.connection.state import MaterialState, PortCondition


def test_material_state_valid_construction() -> None:
    state = MaterialState(
        pressure=101325.0,
        temperature=300.0,
        mole_fractions=(0.7, 0.2, 0.1),
    )

    assert state.pressure == 101325.0
    assert state.temperature == 300.0
    assert state.mole_fractions == (0.7, 0.2, 0.1)


def test_material_state_invalid_mole_fraction_range_fails() -> None:
    with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
        _ = MaterialState(
            pressure=101325.0,
            temperature=300.0,
            mole_fractions=(1.1, -0.1),
        )


def test_material_state_invalid_mole_fraction_sum_fails() -> None:
    with pytest.raises(ValidationError, match="Mole fractions must sum to 1.0"):
        _ = MaterialState(
            pressure=101325.0,
            temperature=300.0,
            mole_fractions=(0.6, 0.3),
        )


def test_port_condition_structure_and_bidirectional_flow() -> None:
    state = MaterialState(
        pressure=250000.0,
        temperature=350.0,
        mole_fractions=(0.5, 0.5),
    )
    port = PortCondition(
        state=state,
        through_molar_flow_rate=-4.2,
        macro_step_time_s=12.5,
    )

    assert port.state == state
    assert port.through_molar_flow_rate == -4.2
    assert port.macro_step_time_s == 12.5
