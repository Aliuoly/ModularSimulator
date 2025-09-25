# Van de Vusse CSTR

This benchmark implements the Van de Vusse reaction in a jacketed CSTR.  It
adds a derived calculation and a cascade temperature-control strategy on top of
the standard modelling workflow.

## System definition

* **Differential states (`States`)** – reactor concentrations `Ca` and `Cb`,
  reactor temperature `T`, and jacket temperature `Tk`.  The states are mapped
to array slices through the enum defined in `system_definitions.py` for easy
access inside the RHS.【F:examples/van_de_vusse_cstr/system_definitions.py†L6-L89】
* **Control elements (`ControlElements`)** – jacket inlet temperature `Tj_in`.
* **Algebraic states** – none are required for this model, so the algebraic
  vector is empty.【F:examples/van_de_vusse_cstr/system_definitions.py†L24-L36】
* **Constants** – feed conditions, kinetic parameters, physical properties and
  equipment geometry.  They are provided through the `VanDeVusseConstants`
  Pydantic model.【F:examples/van_de_vusse_cstr/system_definitions.py†L38-L59】

### Derived calculation

A `HeatDutyCalculation` consumes the measured reactor and jacket temperatures
and returns the instantaneous heat duty `Qk` using the overall heat-transfer
coefficient and area stored in the constants.【F:examples/van_de_vusse_cstr/system_definitions.py†L61-L83】

### Governing equations

The Van de Vusse model uses an Arrhenius rate for the primary reaction,

$$
k_1(T) = k_{10} \exp\left(-\frac{E_1}{T + 273.15}\right), \qquad r_1 = k_1(T)\,V_R\,C_A,
$$

leading to the mass and energy balances

\begin{align*}
\frac{\mathrm{d}C_A}{\mathrm{d}t} &= \frac{-r_1 + F (C_{A0} - C_A)}{V_R}, \\
\frac{\mathrm{d}C_B}{\mathrm{d}t} &= \frac{r_1 - F C_B}{V_R}, \\
\frac{\mathrm{d}T}{\mathrm{d}t} &= \frac{F \rho c_p (T_0 - T) - r_1 \Delta H_{r1} + k_w A_R (T_k - T)}{\rho c_p V_R}, \\
\frac{\mathrm{d}T_k}{\mathrm{d}t} &= \frac{F_j c_{p,K} (T_{j,\text{in}} - T_k) + k_w A_R (T - T_k)}{m_K c_{p,K}}.
\end{align*}

These expressions replicate the implementation in `system_definitions.py` and
form the coupled reactor and jacket dynamics advanced by the solver.

## Instrumentation

Sensors sample each state and the actuator at 0.1 h intervals using the
`SampledDelayedSensor` model.  This produces `TimeValueQualityTriplet`
measurements that are historized by the framework.【F:examples/van_de_vusse_cstr/run_simulation.py†L33-L44】

## Control strategy

A cascade of two `PIDController` instances regulates the process:

1. The outer loop manipulates `Tj_in` to maintain reactor temperature `T` on a
   scheduled setpoint profile.
2. The inner loop trims the temperature setpoint to keep the concentration `Cb`
   near its own trajectory.【F:examples/van_de_vusse_cstr/run_simulation.py†L56-L73】

## Simulation settings

* **Update period (`dt`)** – 0.01 h (≈36 s).
* **Integrator** – uses the default LSODA solver options provided by the base
  `System` class.
* **Variants** – both normal and Numba-accelerated systems are instantiated for
  comparison.【F:examples/van_de_vusse_cstr/run_simulation.py†L75-L102】

## Running the example

Run the script from the repository root:

```bash
python examples/van_de_vusse_cstr/run_simulation.py
```

The script simulates 12,000 updates, plots the measured temperatures and
concentrations, and overlays the calculated heat duty.
