# Irreversible Reaction with Energy Balance

This example augments the simple CSTR with a coupled energy balance and a
multiloop cascade controller.  It shows how additional physics and nested
control structures can be layered onto the same modelling pattern.

## System definition

* **Differential states (`States`)** – reactor volume `V`, concentrations of A
  `C_A` and B `C_B`, reactor temperature `T`, and jacket temperature `T_J`.  These
  balances are advanced by the `EnergyBalanceSystem.rhs` implementation.【F:examples/simple_irreversible_energy_balance_system/system_definitions.py†L6-L92】
* **Control elements (`ControlElements`)** – inlet volumetric flow `F_in` and
  jacket inlet temperature `T_J_in`.  Both can be manipulated by controllers.【F:examples/simple_irreversible_energy_balance_system/system_definitions.py†L16-L22】
* **Algebraic states (`AlgebraicStates`)** – discharge flow `F_out`, linked to
  the reactor volume through the same valve equation as the basic system.【F:examples/simple_irreversible_energy_balance_system/system_definitions.py†L24-L48】
* **Constants** – thermodynamic and hydraulic parameters including Arrhenius
  coefficients, heat capacities, heat-transfer area and jacket properties.【F:examples/simple_irreversible_energy_balance_system/system_definitions.py†L28-L46】

The Arrhenius temperature dependence and heat transfer between reactor and
jacket are implemented directly inside the RHS, demonstrating that complex
physics remain readable when mapped to the structured state vectors.【F:examples/simple_irreversible_energy_balance_system/system_definitions.py†L49-L121】

### Governing equations

The outlet flow is enforced algebraically via the valve characteristic,

$$
F_{\text{out}} = C_v \sqrt{V}.
$$

With the Arrhenius rate constant

$$
k(T) = k_0 \exp\left(-\frac{E_a}{R T}\right), \qquad r = k(T)\,V\,C_A,
$$

the coupled mass and energy balances read

\begin{align*}
\frac{\mathrm{d}V}{\mathrm{d}t} &= F_{\text{in}} - F_{\text{out}}, \\
\frac{\mathrm{d}C_A}{\mathrm{d}t} &= \frac{1}{V} \left(-r + F_{\text{in}} C_{A,\text{in}} - F_{\text{out}} C_A - C_A \frac{\mathrm{d}V}{\mathrm{d}t}\right), \\
\frac{\mathrm{d}C_B}{\mathrm{d}t} &= \frac{1}{V} \left(2 r - F_{\text{out}} C_B - C_B \frac{\mathrm{d}V}{\mathrm{d}t}\right), \\
\frac{\mathrm{d}T}{\mathrm{d}t} &= \frac{F_{\text{in}}}{V}\left(T_{\text{in}} - T\right) + \frac{\Delta H_r r}{\rho c_p V} - \frac{U A (T - T_J)}{\rho c_p V}, \\
\frac{\mathrm{d}T_J}{\mathrm{d}t} &= \frac{F_{J,\text{in}}}{V_J}\left(T_{J,\text{in}} - T_J\right) + \frac{U A (T - T_J)}{\rho_J c_{p,J} V_J}.
\end{align*}

Here $U A$ denotes the overall heat-transfer coefficient times area, and
$\rho c_p$ and $\rho_J c_{p,J}$ are the volumetric heat capacities of the
reactor and jacket fluids.  These expressions correspond directly to the code in
`system_definitions.py`.

## Instrumentation

Sensors use the sample/hold model to showcase different measurement dynamics:

* `F_out`, `V`, `T`, `T_J` – continuous measurements of key states.
* `F_in` – noisy actuator feedback with a 5% coefficient of variance.
* `B` – slow laboratory sample with 900 s sampling and transport delay.
* `T_J_in`, `jacket_flow` – additional instrumentation of the jacket circuit.

All sensors are `SampledDelayedSensor` instances configured in
`run_simulation.py`.【F:examples/simple_irreversible_energy_balance_system/run_simulation.py†L31-L56】

No calculated signals are defined in this case.

## Control strategy

A three-level cascade of `PIDController` objects is used:

1. **Outer loop** – adjusts `F_in` to hold reactor volume `V` at 1000 L.
2. **Middle loop** – manipulates `T_J_in` to maintain jacket temperature `T_J`.
3. **Inner loops** – cascaded controllers regulate reactor temperature `T` and
   ultimately track a scheduled concentration profile for `B`.

The nested controllers are constructed directly in the controller list, showing
how cascade relationships can be declared declaratively.【F:examples/simple_irreversible_energy_balance_system/run_simulation.py†L58-L99】

## Simulation settings

* **Update period (`dt`)** – 30 s between controller executions.
* **Integrator** – LSODA via the default `System` solver options.
* **Variants** – both a pure Python system and a Numba-accelerated system are
  created for comparison.【F:examples/simple_irreversible_energy_balance_system/run_simulation.py†L101-L138】

## Running the example

Execute the script from the repository root:

```bash
python examples/simple_irreversible_energy_balance_system/run_simulation.py
```

The script advances both systems over a multi-step setpoint sequence and plots
the resulting concentration, temperature and actuator trajectories.
