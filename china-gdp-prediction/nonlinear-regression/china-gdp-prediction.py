import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


df = pd.read_csv('../china_gdp.csv')
x_data, y_data = (df['Year'].values, df['Value'].values)

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)


def sigmoid(x, beta_1, beta_2):
    y = 1 / (1 + np.exp(-beta_1 * (x - beta_2)))
    return y

popt, pcov = curve_fit(sigmoid, xdata, ydata)


x = np.linspace(1980, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8, 5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()