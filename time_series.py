from pandas import read_csv
from matplotlib.pyplot import plot, show
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


data = read_csv('jena_climate_2009_2016.csv/jena_climate_2009_2016.csv').to_numpy()[:,2]
print(data.shape)
# Because the data is recorded every 10 minutes, you get 144 data points per day.
sequence_data = []
for i in range(0, 144*356*5, 144):  # data for 5 years
    sequence_data.append(data[i: i + 144].mean())  # avg per day
plot(sequence_data)  # 5 * 365 days
show()
X, y = [], []  # predict based on last 7 days, the model will figure out trend and season
for i in range(len(sequence_data) - 7):
    X.append(sequence_data[i: i + 7])
    y.append(sequence_data[i + 7])
print(len(X))

split_point = len(y) - 100  # 100 points for test
x_train, y_train, x_test, y_test = X[:split_point], y[:split_point], X[split_point:], y[split_point:]
print(len(x_train))
print(len(x_test))
model = MLPRegressor((5), max_iter=200)
model.fit(x_train, y_train)
predicts = model.predict(x_test)
print(round(mean_absolute_error(predicts, y_test)))
plot(range(0, split_point), y_train)
plot(range(split_point, len(X)), predicts, c='red')
plot(range(split_point, len(X)), y_test, c='blue')
show()

