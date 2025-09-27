import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv('../FuelConsumption.csv')
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

regression = LinearRegression()
train_y_ = regression.fit(train_x_poly, train_y)


test_x_poly = poly.transform(test_x)
test_y_ = regression.predict(test_x_poly)
r2_score = r2_score(test_y, test_y_)
