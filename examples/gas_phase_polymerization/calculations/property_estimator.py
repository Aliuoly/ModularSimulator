import numpy as np
from typing import Literal, Dict, Tuple, Deque
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr
from modular_simulation.usables import (
    Calculation, 
    Constant, 
    CalculatedTag, 
    MeasuredTag, 
    OutputTag
)
import collections


class PropertyEstimator(Calculation):
    """
    Inputs to calculation are:
        ---- MEASURED INPUTS ----
        1. mass production rate in ton/h ("mass_prod_rate")
        2. lab measured MI ("lab_MI")
        3. lab measured density ("lab_density")
        ---- CALCULATED INPUTS ----
        4. residence time in hours ("residence_time")
        5. mole ratio of comonomer:monomer ("rM2")
        6. mole ratio of hydrogen:monomer ("rH2")
    constants are:
        1. MI model parameters - array of 4 values. See model for details
        2. density model parameters - array of 4 values. See model for details
        4. EKF MI measurement coefficient of variation (relative standard deviation squared)
        5. EKF density measurement coefficient of variation (relative standard deviation squared)
        6. EKF MI model coefficient of variation (...)
        7. EKF density model coefficient of variation (...)
        8. MI model correction term variance (absolute)
        9. density model correction term variance (absolute)
    initial conditions required are:
        1. EKF initial covariance matrix P - 4x4 positive definite matrix
        2. initial cummulative MI
        3. initial cummulative density
    returns 
        1. instantaneous MI ("inst_MI")
        2. cummulative MI ("cumm_MI")
        3. instantaneous density ("inst_density")
        4. cummulative density ("cumm_density")

    
    """
    inst_MI_tag: OutputTag
    inst_density_tag: OutputTag
    cumm_MI_tag: OutputTag
    cumm_density_tag: OutputTag

    mass_prod_rate_tag: MeasuredTag
    lab_MI_tag: MeasuredTag
    lab_density_tag: MeasuredTag
    
    rM2_tag: CalculatedTag
    rH2_tag: CalculatedTag
    residence_time_tag: CalculatedTag

    MI_model_parameters: Constant = Field(
        default_factory = lambda: np.array([0.26884, 0.10955, 4.29082, 0.46542])
    )
    density_model_parameters: Constant = Field(
        default_factory = lambda: np.array([0.96287, 0.00170, 0.00097, 0.38550])
    )
    MI_meas_rel_noise: Constant = 0.3/2/3 # +- 0.2 when at value of 2 / 3 to get std
    density_meas_rel_noise: Constant = 3/918/3 # +- 2 when at value of 918, /3 to get std
    MI_model_rel_noise: Constant = 1e-2 #these are assumed uncertainty (stdev) in the inst. models
    density_model_rel_noise: Constant = 1e-2 # same as above
    a1_variance: Constant = (1e-3)**2 # absolute variance
    a2_variance: Constant = (1e-4)**2

    # some initial internal states
    P: Constant = Field(
        default_factory = lambda :  np.array(np.diag([(2.0*0.00001)**2, (918*0.00001)**2, (1e-2)**2, (1e-2)**2]))
        ) # initial variance estimate, 10% for MI, 10% for density, 0.1 for a1 and a2
    cumm_MI: Constant = 2.0
    cumm_density: Constant = 918.0

    

    _lab_MI_last: float = PrivateAttr()
    _lab_density_last: float = PrivateAttr()
    _t_lab_MI_last: float = PrivateAttr(default = 0.0)
    _t_lab_density_last: float = PrivateAttr(default = 0.0)
    _t_last_call: float = PrivateAttr(default = 0.0)
    _EKF_q: Deque = PrivateAttr(default_factory=collections.deque) # holds (t, EKF states) --> (t, x, P)
    _u_q: Deque = PrivateAttr(default_factory=collections.deque) # holds (t, model inputs) --> (t, rM2, rH2, tau, f)
    _first_iteration: bool = PrivateAttr(default = True)
    _x: NDArray[np.float64] = PrivateAttr()

    def model_post_init(self, context):
        super().model_post_init(context)
        self._lab_MI_last = self.cumm_MI
        self._lab_density_last = self.cumm_density

        self._x = np.array([
            self.cumm_MI, 
            self.cumm_density, 
            self.MI_model_parameters[0], 
            self.density_model_parameters[0]
            ]).reshape(4,1)

        self._EKF_q.append((0, self._x.copy(), self.P.copy()))

        #print("[INIT] x =", self._x.flatten())
        #print("[INIT] MI_params =", self.MI_model_parameters)
        #print("[INIT] density_params =", self.density_model_parameters)
        #print("[INIT] P =", np.diag(self.P))

    def _calculation_algorithm(
        self, 
        t: float, 
        inputs_dict: Dict[str, float | NDArray]
        ) -> Dict[str, float]:

        prod_rate = inputs_dict[self.mass_prod_rate_tag]
        tau = inputs_dict[self.residence_time_tag] # in hours
        rM2 = inputs_dict[self.rM2_tag]
        rH2 = inputs_dict[self.rH2_tag]
        lab_MI_val = float(inputs_dict[self.lab_MI_tag])
        lab_density_val = float(inputs_dict[self.lab_density_tag])
        # alittle special here, to avoid adding trivial calculations,
        # I am going to reach into the triplet dictionary and grab the 
        # time stamp of the lab sampling times
        lab_MI_sample_time = float(self._last_input_triplet_dict[self.lab_MI_tag].time)
        lab_density_sample_time = float(self._last_input_triplet_dict[self.lab_density_tag].time)
        if prod_rate < 5:
            cumm_MI_placeholder, cumm_density_placeholder = self.compute_yhat(self._x).flatten()
            return_dict = {
                self.inst_MI_tag: cumm_MI_placeholder,
                self.inst_density_tag: cumm_density_placeholder,
                self.cumm_MI_tag: cumm_MI_placeholder,
                self.cumm_density_tag: cumm_density_placeholder,
            }
            return return_dict
        # 1. obtain model inputs and append to input queue
        
        dt = (t - self._t_last_call) / 3600. # convert to hours
        self._t_last_call = t
        f = dt / (dt + tau) # estimated filter factor to save compute
        inputs = [t, rM2, rH2, tau, f]
        self._u_q.append(tuple(inputs))
        if self._first_iteration: # if first iteration, pop the default. Otherwise, use last value
            inst_MI_est, inst_density_est, cumm_MI_est, cumm_density_est = self._propagate_EKF(self._EKF_q.pop(), u_i = 1)
            self._first_iteration = False
        else:
            inst_MI_est, inst_density_est, cumm_MI_est, cumm_density_est = self._propagate_EKF(self._EKF_q[-1], u_i = 1)

        # 2. check which samples are available
        
        
        
        samples = []
        ts = []
        has_sample = False
        if lab_MI_val != self._lab_MI_last:
            self._t_lab_MI_last = lab_MI_sample_time
            self._lab_MI_last = lab_MI_val
            samples.append('MI')
            ts.append(lab_MI_sample_time)
            has_sample = True
        if lab_density_val != self._lab_density_last:
            self._t_lab_density_last = lab_density_sample_time
            self._lab_density_last = lab_density_val
            samples.append('density')
            ts.append(lab_density_sample_time)
            has_sample = True

        if has_sample:
            # 3. order available samples by time (oldest to newest)
            sorted_ts = []
            sorted_samples = []
            if len(samples) > 1:
                if ts[0] > ts[1]:
                    sorted_ts = list(reversed(ts))
                    sorted_samples = list(reversed(samples))
                elif ts[0] == ts[1]:
                    sorted_ts = [ts[0]]
                    sorted_samples = ['both']
            else:
                sorted_ts = [ts[0]]
                sorted_samples = [samples[0]]
            y = np.zeros((2,1))
            y[0,0] = lab_MI_val
            y[1,0] = lab_density_val

            # 4. process available samples from the oldest to the newest
            skip = False
            for i in range(len(sorted_samples)):
                # handle case at start of iteration where sample time may be negative
                if sorted_ts[i] <= 0:
                    continue
                # remove all recent history, as they will be recomputed below
                EKF_hist = self._EKF_q.pop()
                EKF_hist_backup = [EKF_hist]
                while EKF_hist[0] > sorted_ts[i]:
                    if len(self._EKF_q) == 0:
                        print("no prediction history at lab time, skipped processing")
                        for hist in EKF_hist_backup:
                            self._EKF_q.append(hist)
                        skip = True
                        break
                    EKF_hist = self._EKF_q.pop()
                    EKF_hist_backup += [EKF_hist]
                u_i = 1
                if skip:
                    break
                while self._u_q[-u_i][0] > sorted_ts[i]:
                    u_i += 1
                # update EKF state estimates (calculate posterior) at time of sample
                EKF_hist = self._improve_EKF(EKF_hist, y, sample = sorted_samples[i])
                # repropagate up until current time, with the current state estimates as output
                inst_MI_est, inst_density_est, cumm_MI_est, cumm_density_est = self._propagate_EKF(EKF_hist, u_i)
            
        #5. return estimation results
        assert len(self._EKF_q) == len(self._u_q), "Buffers are misaligned!"
        return_dict = {
            self.inst_MI_tag: inst_MI_est,
            self.inst_density_tag: inst_density_est,
            self.cumm_MI_tag: cumm_MI_est,
            self.cumm_density_tag: cumm_density_est,
        }
        return return_dict

    def _improve_EKF(self, EKF_hist: tuple, y: NDArray[np.float64], sample = Literal['both','MI','density']):
        Hmask = np.zeros((2,4))
        if sample == "both":
            Hmask[0,0] = 1
            Hmask[1,1] = 1
        elif sample == 'MI':
            Hmask[0,0] = 1
        elif sample == 'density':
            Hmask[1,1] = 1
        
        
        #   1. retrieve the prior estimates at sample time 
        #       (pop them from queue since the posterior estimates 
        #          for said time will be appended below right before the return)
        if len(self._EKF_q) == 0:
            t_lag, x_lag, P_lag = EKF_hist
        else:
            t_lag, x_lag, P_lag = self._EKF_q.pop() 
        #   2. update EKF at time of sampling
        H = self.compute_H(x_lag)*Hmask
        Rv = np.diag(([(x_lag[0,0]*self.MI_meas_rel_noise)**2, 
                                (x_lag[1,0]*self.density_meas_rel_noise)**2]))
        S = H @ P_lag @ H.T + Rv
        K = P_lag @ H.T @ np.linalg.inv(S)
        yhat = self.compute_yhat(x_lag)
        x_lag_plus = x_lag + K @ (y - yhat)
        P_lag_plus = P_lag - K @ H @ P_lag

        self.MI_model_parameters[0] = x_lag_plus[2]
        self.density_model_parameters[0] = x_lag_plus[3]

        #   3. update the history queue
        EKF_hist = (t_lag, x_lag_plus, P_lag_plus)
        self._EKF_q.append(EKF_hist)

        #print(" Updated MI_params =", self.MI_model_parameters)
        #print(" Updated density_params =", self.density_model_parameters)
        return EKF_hist

    def _propagate_EKF(self, EKF_hist: tuple, u_i: int):
        

        t_EKF, x, P  = EKF_hist
        t, rM2, rH2, tau, f = self._u_q[-u_i]

        # 0. retrieve model inputs from the input queue
        i = len(self._u_q) - u_i
        while i < len(self._u_q):
            t, rM2, rH2, tau, f = self._u_q[i]
            
            # 1. Compute state updates
            #   1A. calculate instantaneous model predictions
            inst_MI_est, inst_density_est = self.inst_property_estimator(rM2, rH2, tau)
            #   1B. calculate internal states x0 and x1
            x[0,0] = (f * inst_MI_est**(-0.286) + (1-f) * x[0,0]**(-0.286))**(-1/0.286)
            x[1,0] = (f * inst_density_est**(-1) + (1-f) * x[1,0]**(-1))**(-1)

            # 2. Compuate state update function jacobian (A)
            A = self._compute_jacobian(rM2, rH2, tau, f)
            # 3. Compute covariance updates (P). This is the prior version (P-)
            cumm_MI_next, cumm_density_next = x[:2,0]
            # the uncertainty in the cummulative model is estimated using taylor expansion
            # cummMI_next = (f*instMI**(-0.286) + (1-f)*cummMI**(-0.286))**(-1/0.286)
            # linear error propagation rule is that variance added due to instMI error is
            # Rw+ = (partial cummMI_next / partial instMI)**2 * err_instMI**2
            # partial cummMI_next / partial instMI = -0.286*(-1/0.286)*f*instMI**(-0.286-1)*cumm_MI_next**((-1/0.286 - 1)/(-1/0.286))
            # so the Rw+ for cummMI due to the uncertainty in instMI would be
            # (f*instMI**(-0.286-1)*cumm_MI_next**(1- -0.286))**2 * err_instMI**2 = (err_instMI*f*(cumm_MI_next / instMI)**(-0.286-1))*2
            # for density it is analogous, just different power
            Rw = np.diag([(self.MI_model_rel_noise*f*(cumm_MI_next/inst_MI_est)**(-0.286-1))**2,
                          (self.density_model_rel_noise*f*(cumm_density_next/inst_density_est)**(-1-1))**2,
                          self.a1_variance,
                          self.a2_variance])
            P = A @ P @ A.T + Rw

            # 4. Append to EKF history
            self._EKF_q.append((t, x.copy(), P.copy()))

            # 5. increment i
            i += 1

            # 6. save self.P 
            self.P = P.copy()
        self._x = x.copy()
        cumm_MI_est, cumm_density_est = self.compute_yhat(x).flatten()
        #print(f"MI inst: {inst_MI_est:5.3f}, cumm: {cumm_MI_est:5.3f}; density inst: {inst_density_est:5.1f}, cumm: {cumm_density_est:5.1f}")
        return inst_MI_est, inst_density_est, cumm_MI_est, cumm_density_est
    
    def get_uncertainty(self):
        # uncertainty in the states of interest - 
        # x0**(-1/0.286), x1**(-1)
        # H = jacobian of measurement of above, so
        # H[0,0] = -1/0.286*x0**(-1/0.286-1) at value of x0, etc.
        
        t, x, P = self._EKF_q[-1]
        H = self.compute_H(x)
        S = H @ P @ H.T
        return np.sqrt(np.diag(S))

    def inst_property_estimator(
            self, 
            rM2:float, 
            rH2:float, 
            tau:float, 
            *, 
            MI_parameters: NDArray[np.float64] | None = None, 
            density_parameters: NDArray[np.float64] | None = None
            ) -> Tuple[float, float]:
        """
        Empirical MI (Melt Index) soft sensor based on plant data AND 
            Empirical density soft sensor based on plant-derived regression.

        These models estimate the instantaneous MI (dg/min) and 
            instantaneous density (g/L) using the equations:

            MI = exp(3.5 * ln(a1 + b1 * rM2 + c1 * rH2) + d1 / τ)
            density = (a2 + b2 * ln(MI) - (c2 * rM2)**d2) * 1000

        where:
            - MI  = melt index, as estimated or measured [dg/min]
            - rM2 = molar ratio of monomer 2 to monomer 1 [mol/mol]
            - rH2 = molar ratio of hydrogen to monomer 1 [mol/mol]
            - τ = residence time of polymer in [hours] => bed_weight [t] / prod_rate [t/hr]

        Args:
            rM2: float
                see above
            rH2: float
                see above
            tau: float
                see above
            MI_parameters: NDArray[np.float64]
                array of 4 values used by the MI model above, corresponding to a1, b1, c1, d1
            density_parameters: NDArray[np.float64]
                array of 4 values used by the density model above, corresponding to a2, b2, c2, d2
        """
        # 1. define parameters
        if MI_parameters is None:
            a1, b1, c1, d1 = self.MI_model_parameters
        else:
            a1, b1, c1, d1 = MI_parameters

        if density_parameters is None:
            a2, b2, c2, d2 = self.density_model_parameters
        else:
            a2, b2, c2, d2 = density_parameters
        
        # 2. calculate instantaneous model predictions
        inst_MI = self._inst_MI_model_internal(rH2, rM2, tau, a1, b1, c1, d1)
        inst_density = self._inst_density_model_internal(rM2, inst_MI, a2, b2, c2, d2)
        return inst_MI, inst_density

    def _compute_jacobian(self, rM2: float, rH2: float, tau: float, f: float, *, x: NDArray[np.float64]|None = None, MI_parameters: NDArray[np.float64] | None = None, density_parameters: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """
        """
        # 1. define parameters
        if MI_parameters is None:
            a1, b1, c1, d1 = self.MI_model_parameters
        else:
            a1, b1, c1, d1 = MI_parameters

        if density_parameters is None:
            a2, b2, c2, d2 = self.density_model_parameters
        else:
            a2, b2, c2, d2 = density_parameters
        if x is None:
            x = self._x.copy()
        cumm_MI = x[0,0]
        cumm_density = x[1,0]
        inst_MI, inst_density = self.inst_property_estimator(rM2, rH2, tau)
        temp0 = 1-f
        temp1 = f/inst_MI**(1/3.5)
        temp2 = (temp0/cumm_MI**(1/3.5) + temp1)**(-4.5)
        temp3 = (f*inst_density**(-1) + temp0*cumm_density**(-1))**(-2)
        temp4 = temp3*inst_density**(-2)

        A = np.zeros((4,4))
        A[0,0] = temp0*temp2/cumm_MI**(1+1/3.5)
        A[1,1] = temp0*temp3/cumm_density**2
        A[2,2] = 1
        A[3,3] = 1

        A[0,2] = 3.5*f*temp2/(inst_MI**(2/3.5))
        A[1,2] = 3500*b2*temp1*temp4
        A[1,3] = 1000*f*temp4
        return A
    
    def compute_H(self, x):
        # go from x = [cumm_MI, cumm_density, a1, a2] to -> y = [cumm_MI, cumm_density]
        H = np.zeros((2,4))
        H[0,0] = 1
        H[1,1] = 1
        return H
    
    def compute_yhat(self, x):
        yhat = np.zeros((2,1))
        yhat[0,0] = x[0,0]
        yhat[1,0] = x[1,0]
        return yhat

    def inst_MI_model(self, rH2: float):
        a1, b1, c1, d1 = self.MI_model_parameters
        rM2 = self._last_input_value_dict["rM2"]
        tau = self._last_input_value_dict["residence_time"]
        return self._inst_MI_model_internal(rH2, rM2, tau, a1, b1, c1, d1)

    def inst_density_model(self, rM2: float):
        a1, b1, c1, d1 = self.MI_model_parameters
        a2, b2, c2, d2 = self.density_model_parameters
        tau = self._last_input_value_dict["residence_time"]
        rH2 = self._last_input_value_dict["rH2"]
        inst_MI = self._inst_MI_model_internal(rH2, rM2, tau, a1, b1, c1, d1)
        return self._inst_density_model_internal(rM2, inst_MI, a2, b2, c2, d2)

    def _inst_MI_model_internal(self, rH2, rM2, tau, a1, b1, c1, d1):
        return np.exp(3.5 * np.log(a1 + b1 * rM2 + c1 * rH2) + d1 / tau)
    
    def _inst_density_model_internal(self, rM2, inst_MI, a2, b2, c2, d2):
        return (a2 + b2 * np.log(inst_MI) - (c2*rM2) ** d2) * 1000.0

if __name__ == '__main__':
    PropertyEstimator(
        output_tags=["cumm_MI","cumm_density","inst_MI","inst_density"],
        measured_input_tags=["mass_prod_rate", "lab_MI","lab_MI_sample_time","lab_density","lab_density_sample_time"],
        calculated_input_tags=["residence_time","rM2","rH2"]
    )
        

