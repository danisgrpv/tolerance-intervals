import locale
import matplotlib
import numpy as np
import scipy as sc
from scipy import stats
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from basics import ConfidenceCalculator

# Set the font settings
locale.setlocale(locale.LC_NUMERIC, "de_DE")
plt.rcParams['axes.formatter.use_locale'] = True

font = {'family' : 'Times New Roman',
        'weight' : 'regular',
        'size'   : 14}

matplotlib.rc('font', **font)



# Example 1: calculation the confidence probability of
# events that true p-value is p for other obsertations
# ----------------------------------------------------

# p-value to being checked
p_value = 0.95

# scope of definition and
# the arrays of tolerance factor values
tf1_min, tf1_max = 0, 5
tf2_min, tf2_max = 0, 5
tf1_array = np.linspace(tf1_min, tf1_max)
tf2_array = np.linspace(tf2_min, tf2_max)

# the sample size and number of calculation repeats
n_elements = 10
n_events = 10000

# create the calculator
model = ConfidenceCalculator(stats.norm)
# calculate the coverage
model.coverage_calc(n_elements, n_events, tf1_array, tf2_array)

# probability of event that the true p_value is equal to p
conf = model.confidence(p_value)

# plot the results
TF1, TF2 = np.meshgrid(tf1_array, tf2_array)
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': '3d'})
ax.plot_surface(TF1, TF2, conf, cmap=cm.Spectral_r)
ax.set_title(f'Confidence probability $\gamma$(p-value={p_value}, tf1, tf2)',
             fontsize=14)
plt.show()
# plt.savefig('example 1.jpeg', dpi=1000)



# Example 2: Estimate the p-value that have required confidence probability
# -------------------------------------------------------------------------

# required confidence of result
conf_required = 0.95

# values for comparation
p_values = np.linspace(0, 1, 101)

tf1 = np.array([1])
tf2 = np.array([1])
model.coverage_calc(n_elements, n_events, tf1, tf2)
conf_ = model.confidence(p_values)[0]

# estimated p-value
p_estim = model.survival(conf_required)

# plot the result
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(p_values, conf_, color='b', lw=0.7)
ax.plot(p_estim, conf_required, color='r', lw=0.0, marker='s')
ax.plot([p_estim, p_estim], [0, conf_required], color='r', lw=0.5, linestyle='--')
ax.plot([0, p_estim], [conf_required, conf_required], color='r', lw=0.5, linestyle='--')
ax.set(xlabel='p-value', ylabel='confidence probability')
ax.set_yticks([conf_required, 0, 0.25, 0.5, 0.75, 1])
plt.show()