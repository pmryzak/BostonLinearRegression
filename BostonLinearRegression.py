from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


boston_market_data = datasets.load_boston()
boston_market_data_p = pd.DataFrame(data=boston_market_data.data, columns=[boston_market_data.feature_names])
print(boston_market_data_p.head())

scaler = StandardScaler()
boston_market_data['data'] = scaler.fit_transform(boston_market_data['data'])
boston_market_data_p = pd.DataFrame(data=boston_market_data.data, columns=[boston_market_data.feature_names])
print(boston_market_data_p.describe())
print(boston_market_data_p.head())

boston_train_data, boston_test_data, \
    boston_train_target, boston_test_target = \
    train_test_split(boston_market_data['data'], boston_market_data['target'], test_size=0.15)

print("Training dataset:")
print("boston_train_data:", boston_train_data.shape)
print("boston_train_target:", boston_train_target.shape)

print("\nTesting dataset:")
print("boston_test_data:", boston_test_data.shape)
print("boston_test_target:", boston_test_target.shape)

linear_regression = LinearRegression()
linear_regression.fit(boston_train_data, boston_train_target)

id = 5
linear_regression_prediction = linear_regression.predict(boston_test_data[id, :].reshape(1, -1))
print(linear_regression_prediction)

print("\nHouse with id = 5")
print("Model predicted for house {0} value {1}".format(id, linear_regression_prediction))
print("Real value for patient \"{0}\" is {1}".format(id, boston_test_target[id]))

print("\nMean squared error")
print(mean_squared_error(boston_test_target, linear_regression.predict(boston_test_data)))

print("\nVariance score: %.2f")
print(r2_score(boston_test_target, linear_regression.predict(boston_test_data)))

print("\nCross validation")
scores = cross_val_score(LinearRegression(), boston_market_data['data'], boston_market_data['target'], cv=5)
print(scores)
