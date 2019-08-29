import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import pdb


# Functions of creating synthetic time series
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# Functions of creating feature and label
def t_window(x, size, shift=None, stride=1):
    try:
        nd = len(size)
    except TypeError:
        size = tuple(size for i in x.shape)
        nd = len(size)
    if nd != x.ndimension():
        raise ValueError("size has length {0} instead of "
                         "x.ndim which is {1}".format(len(size),
                                                      x.ndimension()))
    out_shape = tuple(xi-wi+1 for xi, wi in zip(x.shape, size)) + size
    if not all(i > 0 for i in out_shape):
        raise ValueError("size is bigger than input array along at "
                         "least one dimension")
    out_strides = x.stride() * 2
    return t.as_strided(x, out_shape, out_strides)


def t_apply(func, M):
    res = [func(m) for m in t.unbind(M, dim=0)]
    return res


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    '''
    series [t.tensor]: dataset
    '''
    if(isinstance(series, np.ndarray)):
        dataset = t.from_numpy(series)
    dataset = t_window(dataset, window_size + 1, shift=1)
    dataset = t_apply(lambda window: (window[:-1], window[-1:]), dataset)
    dataset = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataset


# Design structure of deep neural network model
class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.loss = None
        self.optimizer = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define metric and learning algorithms.
# As well as train(fit) and predict.
class Sequential():

    def __init__(self, model):
        self.model = model
        self.metric = None
        self.optimizer = None

    def t_compile(self, metric, optimizer):
        self.metric = metric
        self.optimizer = optimizer

    def t_fit(self, dataset, epochs, callbacks=[], verbose=0):
        Training_loss_values = []
        if(callbacks != []):
            scheduler = LambdaLR(self.optimizer, lr_lambda=callbacks)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataset, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.metric(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            if(callbacks != []):
                scheduler.step()
            Training_loss_values.append(running_loss / i)
        print('Finished Training')
        return Training_loss_values

    def t_predict(self, dataset):
        return self.model(dataset)


# Plot results
def openfig(Figsize=(10, 6)):
    plt.figure(figsize=Figsize)


def savefig(filename):
    plt.savefig('../figure/{0}_{1}.png'.format(prefix_name, filename))
    plt.clf()

t.manual_seed(1729)
np.random.seed(1729)
matplotlib.use('Agg')
sns.set()
plt.switch_backend('agg')
prefix_name = 'SPW2L3'

# Config of creating time series
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the time series
series = baseline + trend(time, slope) +\
    seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

# Train and validate split
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Config of creating feature and label
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# Create feature and label of training set
dataset = windowed_dataset(x_train, window_size, batch_size,
                           shuffle_buffer_size)
# Model build
model = Sequential(Net(input_size=window_size))
model.t_compile(metric=nn.MSELoss(), optimizer=optim.SGD(
                model.model.parameters(), lr=1e-6, momentum=0.9))
# Model train
model.t_fit(dataset, 100)
# Model predict
forecast = []
for time in range(len(series) - window_size):
    forecast_feature = series[time:time + window_size][np.newaxis]
    forecast.append(model.t_predict(t.from_numpy(forecast_feature)))

forecast = forecast[split_time-window_size:]
results = t.cat(forecast).detach().numpy()[:, 0]

# Plot forecast result
openfig()
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
savefig('1_forecast')

# Model evaluate
print(mean_absolute_error(x_valid, results))

# Another model
model2 = Sequential(Net(input_size=window_size))
model2.t_compile(metric=nn.MSELoss(), optimizer=optim.SGD(
                model2.model.parameters(), lr=1e-8, momentum=0.9))
# Callbacks with learning rate schedule
lr_schedule = [lambda epoch: 1e-8 * 10**(epoch / 20)]
history2 = model2.t_fit(dataset, 100, callbacks=lr_schedule)
# Plot
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
openfig()
plt.semilogx(lrs, history2)
plt.axis([1e-8, 1e-3, 0, 2500])
savefig('2_resultsLoss_learningRates')
print(lrs[np.argmin(history2)])

# Final model
window_size3 = 30
dataset3 = windowed_dataset(x_train, window_size3, batch_size,
                            shuffle_buffer_size)
model3 = Sequential(Net(input_size=window_size3))
model3.t_compile(metric=nn.MSELoss(), optimizer=optim.SGD(
                 model3.model.parameters(), lr=8e-06, momentum=0.9))
history3 = model3.t_fit(dataset3, 500)
epochs = range(len(history3))
openfig()
plt.plot(epochs, history3, 'b', label='Training Loss')
savefig('3_loss_epoch')

epochs = range(10, len(history3))
plot_loss = history3[10:]
# print(plot_loss)
openfig()
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
savefig('4_loss_epoch_removeFirst10')

forecast3 = []
for time in range(len(series) - window_size3):
    forecast_feature = series[time:time + window_size3][np.newaxis]
    forecast3.append(model3.t_predict(t.from_numpy(forecast_feature)))

forecast3 = forecast3[split_time-window_size3:]
results3 = t.cat(forecast3).detach().numpy()[:, 0]

openfig()
plot_series(time_valid, x_valid)
plot_series(time_valid, results3)
savefig('5_forecast2')

print(mean_absolute_error(x_valid, results3))
