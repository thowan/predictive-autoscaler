import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime(x, '%Y-%m')

sales = pd.read_csv("sales-cars.csv", index_col=0, parse_dates=[0], date_parser=parser)
print(sales.head())
#sales.plot()
#plt.show()
sales_diff = sales.diff(periods=1)
sales_diff = sales_diff[1:]

print(sales_diff.head())
#plot_acf(sales)
#plot_acf(sales_diff)
#plt.show()

X = sales.values

train = X[0:27] #27 train
test = X[27:] #9 data

predictions = []

model_ar = AutoReg(train,5)
model_ar.fit = model_ar.fit()

predictions = model_ar.fit.predict(start=26, end=36)
plt.plot(test)
plt.plot(predictions)
plt.show()
