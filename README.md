Source code for calclation of confidence probability vs tolerance factors dependence and estimation statistical parameters with required confidence probability

## Import
```python
import numpy as np
import scipy as sc
from basics import ConfidenceCalculator
```
## Initial parameters
```python
# scope of definition and
# the arrays of tolerance factor values
tf1_min, tf1_max = 0, 5
tf2_min, tf2_max = 0, 5
tf1_array = np.linspace(tf1_min, tf1_max)
tf2_array = np.linspace(tf2_min, tf2_max)

# the sample size and number of calculation repeats
n_elements = 10
n_events = 10000
```

## Create calculator
```python
# a priory distribution
distr_type = stats.norm
# create the calculator
model = ConfidenceCalculator(distr_type)
```

## Calculations
```python
# p-value to being checked
p_value = 0.95

# calculate the coverage
model.coverage_calc(n_elements, n_events, tf1_array, tf2_array)
# probability of event that the true p_value is equal to p
conf = model.confidence(p_value)
```

## Result of calculations
```text
Result of calculations is the confidence probability (level) vs tolerance factors dependence:
```
```math
\gamma(P, tf1, tf2, n)
```
```text
In gif below parameter P is iterable
for n = 2:
```
![Plot](https://github.com/danisgrpv/tolerance-intervals/blob/master/plots/plot-2.gif)

```text
for n = 10:
```
![Plot](https://github.com/danisgrpv/tolerance-intervals/blob/master/plots/plot-1.gif)


## Processing of observations
```text
The probability of event with required confidence level is:
```
```math
P^* = {arg\,min}_P \ |\gamma(P, k1, k2, n) - \gamma_{required}|
```

```text
In gif below required confidence probability is iterable.
Probability of event vs required confidence probability dependence is shown:
```
![Plot](https://github.com/danisgrpv/tolerance-intervals/blob/master/plots/survival.gif)
