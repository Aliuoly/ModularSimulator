import importlib

import pytest

material_models = importlib.import_module("modular_simulation.connection.material_models")
DensityModelSelection = material_models.DensityModelSelection
IdealGasDensityModel = material_models.IdealGasDensityModel
IncompressibleDensityModel = material_models.IncompressibleDensityModel
select_density_model = material_models.select_density_model


def test_incompressible_density_model_returns_constant_density() -> None:
    model = IncompressibleDensityModel(density_kg_per_m3=997.0)

    assert model.density(pressure=1.0e5, temperature=273.15) == 997.0
    assert model.density(pressure=5.0e5, temperature=350.0) == 997.0


def test_ideal_gas_density_scales_with_pressure_and_temperature() -> None:
    model = IdealGasDensityModel(specific_gas_constant_j_per_kg_k=287.0)

    rho_base = model.density(pressure=1.0e5, temperature=300.0)
    rho_double_pressure = model.density(pressure=2.0e5, temperature=300.0)
    rho_double_temperature = model.density(pressure=1.0e5, temperature=600.0)

    assert rho_double_pressure == pytest.approx(2.0 * rho_base)
    assert rho_double_temperature == pytest.approx(0.5 * rho_base)


@pytest.mark.parametrize(
    ("pressure", "temperature", "expected_message"),
    [
        (0.0, 300.0, "Pressure must be positive"),
        (-1.0, 300.0, "Pressure must be positive"),
        (101325.0, 0.0, "Temperature must be positive"),
        (101325.0, -10.0, "Temperature must be positive"),
    ],
)
def test_ideal_gas_density_rejects_non_positive_pressure_and_temperature(
    pressure: float,
    temperature: float,
    expected_message: str,
) -> None:
    model = IdealGasDensityModel(specific_gas_constant_j_per_kg_k=287.0)

    with pytest.raises(ValueError, match=expected_message):
        _ = model.density(pressure=pressure, temperature=temperature)


def test_density_model_selection_returns_expected_implementation() -> None:
    incompressible = select_density_model(
        DensityModelSelection(model="incompressible", parameters={"density_kg_per_m3": 995.0})
    )
    ideal_gas = select_density_model(
        DensityModelSelection(
            model="ideal_gas",
            parameters={"specific_gas_constant_j_per_kg_k": 287.0},
        )
    )

    assert isinstance(incompressible, IncompressibleDensityModel)
    assert isinstance(ideal_gas, IdealGasDensityModel)
