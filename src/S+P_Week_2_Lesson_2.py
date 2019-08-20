'''
# Week 2 Lesson 2
In the screencast for this lesson I go through steps for slicing the windows of time series.
At the end, you can generate features and labels of dataset.

'''

# # Setup
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import pdb
print(t.__version__)
t.manual_seed(1729)
np.random.seed(1729)
matplotlib.use('Agg')
sns.set()
plt.switch_backend('agg')
prefix_name = 'SPW2L2'


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

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=1729)

# # Split training set and validating set
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
# What is it? Is it necessary? 這個是甚麼？
shuffle_buffer_size = 1000

# # Slice time series to several windows
# 將連續的資料分段
def t_window(x, size, shift=None, stride=1):
    try:
        nd = len(size)
    except TypeError:
        size = tuple(size for i in x.shape)
        nd = len(size)
    if nd != x.ndimension():
        raise ValueError("size has length {0} instead of "
                         "x.ndim which is {1}".format(len(size), x.ndimension())) 
    out_shape = tuple(xi-wi+1 for xi, wi in zip(x.shape, size)) + size
    if not all(i>0 for i in out_shape):
        raise ValueError("size is bigger than input array along at "
                         "least one dimension")
    out_strides = x.stride() * 2
    return t.as_strided(x, out_shape, out_strides)

# 對 tensor 物件所有 Row 輸入給 func 函數
def t_apply(func, M):
    res = [func(m) for m in t.unbind(M, dim=0) ]
    return res 

# 整理特徵與標籤
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

# Create features and labels of training set 整理資料
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)

# # Design the models 模型建立
# Build linear regression model
model = nn.Linear(window_size, 1)
# Build loss function
criterion = nn.MSELoss()
# Choose optimization method and their parameters
optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)


# # Train the models 模型訓練
def t_fit(model, dataset, epochs, verbose=0):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if verbose:
                iter_print = dataset.shape[0] // 100
                if i // iter_print == 0:
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / iter_print))
                    running_loss = 0.0
    print('Finished Training')
# Training process
t_fit(model, dataset, 100, 0)
print("Layer weights {}".format([model.weight, model.bias]))

# # Forecast 模型預測
forecast = []
for time in range(len(series) - window_size):
    # 先對所有的資料都預測
    # 假設預測前都可以有足夠的特徵，實務上不一定可行
    forecast_feature = series[time:time + window_size][np.newaxis]
    forecast.append(model(t.from_numpy(forecast_feature)))
# 只取 validating set
forecast = forecast[split_time-window_size:]
# 預測結果轉為 numpy array
results = t.cat(forecast).detach().numpy()[:, 0]


# # Plot results 繪製結果
def openfig(Figsize=(10, 6)):
    plt.figure(figsize=Figsize)


def savefig(filename):
    plt.savefig('../figure/{0}_{1}.png'.format(prefix_name, filename))
    plt.clf()

# Plotting
openfig()
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
savefig('forecast')

# # Evaluate performance of models 模型評估
print(mean_absolute_error(x_valid, results))