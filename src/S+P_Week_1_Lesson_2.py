'''
# Lesson 2
In the screencast for this lesson I go through a few scenarios for time series.
This notebook contains the code for that with a few little extras! :)

'''

# # Setup
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import pdb
matplotlib.use('Agg')
sns.set()
prefix_name = 'SPW1L2'


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def openfig(Figsize=(10, 6)):
    plt.figure(figsize=Figsize)


def savefig(filename):
    plt.savefig('../figure/{0}_{1}.png'.format(prefix_name, filename))
    plt.clf()


# # Common patterns of time series
# ## Trend and Seasonality
def trend(time, slope=0):
    return slope * time

# Let's create a time series that just trends upward:
time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)

openfig()
plot_series(time, series)
savefig('1_trendUpwards')


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
    reference:
        1. [Fourier Series](https://en.wikipedia.org/wiki/Fourier_series)
        2. [Activation Function]
           (https://en.wikipedia.org/wiki/Activation_function)
    """
    period_pattern = np.cos(season_time * 2 * np.pi)
    nonperiod_pattern = np.log((1 + np.exp(2*season_time)) /
                               (1 + np.exp(2*(season_time-1)))
                               ) / 2
    result = np.where(season_time < 0.5,
                      period_pattern,
                      nonperiod_pattern
                      )
    return result


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period
    period: 週期
    amplitude: 振幅
    phase: 平移 phase 個時間單位
    *seasonality: 只與時間戳記有關
    """
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

# Now let's generate a time series with a seasonal pattern:
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

openfig()
plot_series(time, series)
savefig('2_seasonalPattern')

# Now let's create a time series with both trend and seasonality:
baseline = 10
slope = 0.05
series = baseline + trend(time, slope) + \
    seasonality(time, period=365, amplitude=amplitude)

openfig()
plot_series(time, series)
savefig('3_trendAddSeasonality')


# ## Noise
# In practice few real-life time series have such a smooth signal.
# They usually have some noise,
# and the signal-to-noise ratio can sometimes be very low.
# Let's generate some white noise:
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=1729)

openfig()
plot_series(time, noise)
savefig('4_noise')

# Now let's add this white noise to the time series:
series += noise

openfig()
plot_series(time, series)
savefig('5_syntheticTimeSeries')


# ## Autocorrelation
def autocorrelation_1(time, amplitude, seed=None):
    '''
    1. Formula: ar(step) = φ1 * ar(step-50) + φ2 * ar(step-33)
    2. Generate random values and keep first 50 as baseline(100)
    '''
    rnd = np.random.RandomState(seed)
    φ1 = 0.5
    φ2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += φ1 * ar[step - 50]
        ar[step] += φ2 * ar[step - 33]
    return ar[50:] * amplitude

series = autocorrelation_1(time, 10, seed=1729)

openfig()
plot_series(time[:200], series[:200])
savefig('6_autocorrelation_1')


def autocorrelation_2(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude

series = autocorrelation_2(time, 10, seed=1729)

openfig()
plot_series(time[:200], series[:200])
savefig('7_autocorrelation_2')

# Autocorrelation add trend
series = autocorrelation_2(time, 10, seed=1729) + trend(time, 2)

openfig()
plot_series(time[:200], series[:200])
savefig('8_autocorrelationAddTrend')

# Autocorrelation add trend and seasonality
series = autocorrelation_2(time, 10, seed=1729) + \
    seasonality(time, period=50, amplitude=150) + \
    trend(time, 2)

openfig()
plot_series(time[:200], series[:200])
savefig('9_autocorrelationAddTrendAddSeasonality')

# Synthetic time series
series += noise

openfig()
plot_series(time[:200], series[:200])
savefig('10_syntheticTimeSeries_2')


# ## Impulses
def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series

series = impulses(time, 10, seed=1729)
openfig()
plot_series(time, series)
savefig('11_impulses')


def autocorrelation_3(source, φs):
    '''
    1. Formula: ar(step) = np.dot(φs, ar(step-lag))
    2. φs, isinstance(φs, dict): {(lag), (coefficient)}
    3. Modify is following the *source* (original time series)
    '''
    ar = source.copy()
    max_lag = len(φs)
    for step, value in enumerate(source):
        for lag, φ in φs.items():
            if step - lag > 0:
                ar[step] += φ * ar[step - lag]
    return ar

signal = impulses(time, 10, seed=1729)
series = autocorrelation_3(signal, {1: 0.8})
openfig()
plot_series(time, series)
plt.plot(time, signal, "k-")
savefig('12_autocorrelationAddImpulses')

signal = impulses(time, 10, seed=1729)
series = autocorrelation_3(signal, {10: 0.70, 50: -0.2})
openfig()
plot_series(time, series)
plt.plot(time, signal, "k-")
savefig('13_autocorrelationAddImpulses_2')

# ## Summary of creating synthetic time series
# Now, combine above patterns: (1) trend (2) seasonality (3) noise
# (4) autocorrelation (5) impulses
# parameters
baseline = 10
# trend
slope = 0.05
# seasonality
amplitude_seasonality = 40
period = 90
# noise
noise_level = 5
# impulses
num_impulses = 4
amplitude_impulses = 500
# autocorrelation
Φ = {1: 0.70, 30: -0.2}
# Generate synthetic time series
series_trend = trend(time, slope=slope)
series_season = seasonality(time, period=period,
                            amplitude=amplitude_seasonality)
series_noise = white_noise(time, noise_level, seed=1729)
series_signal = impulses(time, num_impulses=num_impulses,
                         amplitude=amplitude_impulses, seed=1729)
series = baseline + series_trend + series_season + series_noise + series_signal
series = autocorrelation_3(series, Φ)
series += np.abs(np.min(series))

openfig()
plot_series(time, series)
savefig('14_syntheticTimeSeries_3')

# # Statistical methods for time series
# ## Difference
# The usefule method to remove trends and seasonality
# Ref: [How to Remove Trends and Seasonality with a Difference Transform in Python]
# (https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/)
series_diff1 = series[1:] - series[:-1]
openfig()
plot_series(time[1:], series_diff1)
savefig('15_difference_13_2')

# ## Autocorrelation Plot
openfig()
autocorrelation_plot(series)
savefig('16_autocorrelationPlot_13_2')

# ## ARIMA
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# # Miscellaneous
# All right, this looks realistic enough for now. Let's try to forecast it.
# We will split it into two periods:
# the training period and the validation period
# (in many cases, you would also want to have a test period).
# The split will be at time step 1000.
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
