# Dummy System
This dummy system follows the following governing equations, resembling to 2 CSTR with a first order irreversible reaction occuring at constant volume and temperature, where the flow is assumed to be plugflow between the CSTRs, and as such exhibits some timedelay in the concentration but not in the flow.

$$\frac{dy_1}{dt} = c_1u_1(t) - k_1y_1(t) - y_1(t)u_2(t)$$
$$\frac{dy_2}{dt} = y_1(t-\theta_1)u_2(t) - k_2y_2(t) - y_2(t)u_3(t)$$

where $u_1$ is a control element, and $u_2(t)=u_1(t)$ is an algebraic state, indicating that, in the equivalent CSTR in series model, the plug flow between tanks is constant density. 

In $\frac{dy_2}{dt}$, the first term has a time delay in $y_1(t-\theta_1)$. 

Note that, in this framework, it is impossible to specify and solve delayed-differential systems. However, it is possible to model it through a series of algebraic delays. 

$$y_{d,1}(t) = y_1(t-\theta_1)$$
$$\frac{dy_2}{dt} = y_{d,1}(t)u_2(t) - k_2y_2(t) - y_2(t)u_3(t)$$

The algebraic delays can be formulated as systems with a single algebraic state that updates at time steps which are divisible by the time delay. suppose that $dt = \frac{\theta_1}{2}$, then we will model the delay with 2 intemediate systems as such:

$$S_1: \frac{dy_1}{dt} = c_1u_1(t) - k_1y_1(t) - y_1(t)u_2(t)$$
$$S_{delay, A}: y_{delay,A}(t) = y_1(t)$$
$$S_{delay, B}: y_{delay,B}(t) = y_{delay,A}(t)$$
$$S_2: \frac{dy_2}{dt} = y_{delay,B}(t)u_2(t) - k_2y_2(t) - y_2(t)u_3(t)$$

The 'system of systems' then flows as such
1. Define the systems individually
2. Define the system of systems
    2A. Framework will check that the final set of states and control elements and sensors and controllers are resolvable. For example, in our system, the tag 'y_1' shows up in $S_1$'s states and $S_{delay, A}$'s algebraic states, so the framework combines the two and treat them as one and the same. The framework will do so for all systems in the aggregate system sequentially, such that recycle streams are also resolved. 
3. The systems in the aggregate system are then stepped one after the other. It is important to note that the system .steps are called in PARALLEL, not in series; if it was called in series, no timedelays would be possible to model - the result of the first system is applied immediately to the next, and the chain of effects continues.

The aggregate system shall be called a *plant*


