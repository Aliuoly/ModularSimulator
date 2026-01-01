## Design of the `Sensors`

The `AbstractSensor` class is a base class for all sensors. A sensor is a component that measures a state of a `System` instance. 

By design, a sensor 
 - is tightly coupled to a `System` instance. The sensor must be initialized with a `System` instance before use. 





It extends the `AbstractComponent` class and implements the following methods required by `AbstractComponent`:
1. `initialize`: Initialize the sensor by binding it to a `System` instance.
2. `update`: Updates the sensor's signal. 
3. `to_dict`: Convert the sensor to a dictionary that specifies both its configuration and runtime state. 
4. `from_dict`: Load a sensor from a dictionary that specifies both its configuration and runtime state. 


