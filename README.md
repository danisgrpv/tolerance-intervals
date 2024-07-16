Source code for calclation of confidence probability vs tolerance factors dependence and estimation statistical parameters with required confidence probability

## Import
```text
import numpy as np
import scipy as sc
from tolerance.confidence import ConfidenceCalculator
```
## Initial parameters
```text
# tolerance factor arrays
k1 = np.linspace(0.05, 10, 100, dtype=np.float64)
k2 = np.copy(k1)
# sample size
ss = 10
# number of repeats
rn = 10000
# RV distribution
distr = stats.norm

## Create calculator
```text
# Initialization
calculator = ConfidenceCalculator(distr)
```

## Calculations
```text
# Calculating the coverages
calculator.calc_coverages(sample_size=ss, num_of_events=rn, tolerance_factor_lower=k1, tolerance_factor_upper=k2)
# Calculating the confidence
check_coverages = np.linspace(0, 1, 101)
# Confidence probability of event A = {check_coverages >= true_coverage}
calculator.calc_confidence(check_coverages)
```
