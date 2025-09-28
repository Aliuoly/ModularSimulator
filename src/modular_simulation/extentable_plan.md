In making the systems extendible, the naive plant approach, where multiple systems are appended onto each other, is not satisfactory. Afterall, these systems, if they were to interact with each other, necessarily already knew about each other in their systems' rhs and calculate_algebraic_values methods - in which case, might as well make them the same system to begin with. 

In order to gradually add parts to the system, it is necessary to make public facing methods such as
 - ```.add_inlet(additional_measurable: MeasurableQuantities, additional_sensors: List[Sensor], additional_calculations: List[Calculation], additional_controllers: List[Controller])```
 - ```.add_outlet(...)```

With that being said, it is now also necessary to distinguish between inlet and outlet flows, as they are treated differently in the system rhs. 

With that being said as well, it is now necessary to modularize the rhs into the accumulation = addition - removal + generation form rather than write it yourself. 

With that being said then again, it is now necessary to generalize the generation part of the rhs - meaning you now have to define reactions as part of the system, along with their rate constants and rate laws. 

For tanks with simple inlet/outlet flows and no reactions, this becomes very easy to do. 

Soooo, now we are full on aspen simulator. You need to now define unit processes - tanks, separators, CSTRs, PFRs, etc, each with their governing barebones rhs skeleton. 