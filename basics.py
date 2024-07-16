import numpy as np
import scipy as sc
from scipy import stats

# Implementation of confidence calculator class
# ---------------------------------------------

class ConfidenceCalculator:

    def __init__(self, distr_type):
        # set the a priori distribution type
        self.distr_type = distr_type
        self.distr = distr_type()

    
    def coverage_calc(self, n_elements, n_events, tf1, tf2):
        """
        Calculate the coverage (current p-value)

        Parameters
        ----------
        n_elements : int
            The size of random variables sample
        
        n_events : int
            The number of repeats that determines
            the accuracy of calculations

        tf1 : array_like
            The values of lower tolerance factors
                
        tf2 : array_like
            The values of upper tolerance factors

        Returns
        -------
        coverage : array_like, shape=(len(tf1), len(tf2))
            The values of coverage (current p-value)
        """
        # generate the samples the "n_events"-th number of times
        task_shape = (n_elements, n_events)
        sample = self.distr.rvs(size=task_shape)

        # estimate the loc-scale parameters of samples
        locs = np.zeros(shape=n_events)
        scales = np.zeros(shape=n_events)
        for i, samp in enumerate(sample.T):
            locs[i], scales[i] = self.distr_type.fit(samp)

        # calculate the p-value
        L, TF1, TF2 = np.meshgrid(locs, tf1, tf2)
        S = np.meshgrid(scales, tf1, tf2)[0]
        LB = L - TF1 * S
        UB = L + TF2 * S
        self.coverage = self.distr.cdf(UB) - self.distr.cdf(LB)
        self.n_events = n_events


    def confidence(self, p):
        """
        Calculate the confidence probability
        of events that the true p_value is equal to p

        Parameters
        ----------
        p : dobule in range [0, 1]
            p-value to being checked

        Returns
        -------
        out : double in range [0, 1]
            The confidence probability of event
            that the true p_value is equal to p

            out = P{true p_value == p}
        """
        # event
        A = self.coverage >= p
        event_axis = 1
        # confidence probability of event
        out = A.sum(axis=event_axis) / self.n_events
        return out
    

    def survival(self, conf_required):
        """
        Estimate the p-value that have required
        confidence probability
        """
        # object function to minimize
        obj = lambda p_value: (conf_required - self.confidence(p_value)[0][0])**2
        res = sc.optimize.minimize(fun=obj, x0=0.5, method='Nelder-Mead').x
        out = res[0]
        return out