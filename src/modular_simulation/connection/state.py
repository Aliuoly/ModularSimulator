from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


COMPOSITION_ABS_TOL = 1e-9


class MaterialState(BaseModel):
    pressure: float = Field(description="Absolute pressure at the boundary")
    temperature: float = Field(description="Absolute temperature at the boundary")
    mole_fractions: tuple[float, ...] = Field(
        min_length=1,
        description="Material composition as mole fractions ordered by component index",
    )

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        extra="forbid", arbitrary_types_allowed=True
    )

    @field_validator("mole_fractions")
    @classmethod
    def _validate_mole_fraction_range(
        cls, value: tuple[float, ...], info: ValidationInfo
    ) -> tuple[float, ...]:
        del info
        for fraction in value:
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("All mole fractions must be between 0.0 and 1.0.")
        return value

    @model_validator(mode="after")
    def _validate_mole_fraction_sum(self) -> "MaterialState":
        composition_sum = sum(self.mole_fractions)
        if abs(composition_sum - 1.0) > COMPOSITION_ABS_TOL:
            raise ValueError("Mole fractions must sum to 1.0.")
        return self


class PortCondition(BaseModel):
    state: MaterialState
    through_molar_flow_rate: float = Field(
        description="Signed through-flow at the port; positive/negative values indicate direction",
    )
    macro_step_time_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional macro-step timestamp in seconds",
    )

    model_config = ConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        extra="forbid", arbitrary_types_allowed=True
    )
