# Structure

Things that affect states: states, control elements

States and ControlElements are collectively held in the MeasurableQuantities class, along with the timestamp. These quantities, as the name suggests, are measurable. If the designer desires, they can fully observed or partially observed.

Things that affect control elements: measurements, calculations (which are directly done using measurements or other calculations)

Measurements and Calculations are collectively held in the UsableQuantities class. UsableQuantities class itself is instantiated using 2 dictionaries. 
 - `MeasurementDefinitions: Dict[str, Sensor]`. This is a dictionary that explicitly states what measurements are available and how they are obtained from the lower-level `MeasurableQuantities`.
The way measurements are obtained is through the `measurement_function` field of the `Sensor`, which is a callable with signature `measurement_function(MeasurableQuantities) -> float | NDArray[np.float64]`. 
 The `Sensor` class itself may perform stateful calculations such as measurement delays, noise additions, error simulations.
 Measurement is a `callable` with signature `__call__(self, measurable_quantities: MeasurableQuantities) -> float | NDArray[np.float64]`. This will be exclusively simple functions since statefulness is inherently not needed for simple measurements 
 - `CalculationDefinitions: Dict[str, Calculation]`. This is a dictionary that explicitly states what calculations are carried out. 
 Calculation is a `callable` with signature `__call__(self, usable_quantities: UsableQuantities) -> float | NDArray[np.float64]`. This is typically a `callable class` since many calculations require statefulness.
measurements are defined via a single dictionary in the structure of {"measurement name":MeasurementFunction}

It is valid to have a "measurement name" show up in the CalculationDefinition as well. For example, you receive the measurement, and then use a calculation to judge its validity. Since calculation always runs after measurement, the result of the calculation will always be overwriting the measurement in this case.

So, during a tick of the simulation, measurements get retrieved, calculations get performed (in the order as defined to avoid dependencies and loops), controller updates the control elements, and the states are evolved. 

### `UsableQuantities`
`UsableQuantities` is defined by 2 dictionarys:
-`MeasurementDefinitions: Dict[str, Sensor]`
-`CalculationDefinitions: Dict[str, Calculation]`
When `UsableQuantities` updates, it returns the result as `Dict[str, UsableResults]`. 

#### `Calculation`
`Calculation` is the generic class all custom calculations must inherit. Each `Calculation` uses an instance of the `UsableQuantities`'s output `Dict[str, UsableResults]` and returns `Any`. 
- For example, a custom class declared as `PropertyModel(Calculation)` can return a `callable` object that itself inherits from `Calculation`, but now returns a scalar value corresponding to the property. This model then might have usage pattern like `property_value = property_model(usable_quantities)`. In such a way, the PropertyModel output (which is, once again, a Calculation itself) could be used in a controller somewhere if necessary.

#### `ControlQuantities`
`ControlQuantities` is defined using a dictionary with key-value pairs following `Dict[str, controller]`. When the `ControlQuantities` updates, it returns a similar dictionary `Dict[str, ControllerOuput]`
`controller` takes a `UsableQuantities` object and a `Trajectories` object as inputs and returns a `ControlledQuantities`