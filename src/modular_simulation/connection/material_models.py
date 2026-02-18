from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class DensityModel(Protocol):
    def density(self, *, pressure: float, temperature: float) -> float: ...


class IncompressibleDensityModel(BaseModel):
    density_kg_per_m3: float = Field(gt=0.0, description="Constant fluid density")

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        extra="forbid", arbitrary_types_allowed=True
    )

    def density(self, *, pressure: float, temperature: float) -> float:
        del pressure, temperature
        return self.density_kg_per_m3


class IdealGasDensityModel(BaseModel):
    specific_gas_constant_j_per_kg_k: float = Field(
        gt=0.0,
        description="Specific gas constant used in rho = P / (R_specific * T)",
    )

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        extra="forbid", arbitrary_types_allowed=True
    )

    def density(self, *, pressure: float, temperature: float) -> float:
        if pressure <= 0.0:
            raise ValueError("Pressure must be positive for ideal-gas density evaluation.")
        if temperature <= 0.0:
            raise ValueError("Temperature must be positive for ideal-gas density evaluation.")
        return pressure / (self.specific_gas_constant_j_per_kg_k * temperature)


class DensityModelSelection(BaseModel):
    model: Literal["incompressible", "ideal_gas"]
    parameters: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        extra="forbid", arbitrary_types_allowed=True
    )


def select_density_model(config: DensityModelSelection) -> DensityModel:
    if config.model == "incompressible":
        return IncompressibleDensityModel(**config.parameters)
    if config.model == "ideal_gas":
        return IdealGasDensityModel(**config.parameters)
    raise ValueError(f"Unsupported density model '{config.model}'.")
