import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.formula.api as sm
# import statsmodels.api as sm
# from sklearn.datasets import load_boston


data=pd.read_csv('data.csv')
data.drop('B', axis=1, inplace=True)
data.drop('CRIM', axis=1, inplace=True)
data.drop('ZN', axis=1, inplace=True)
data.drop('LSTAT', axis=1, inplace=True)
data.drop('MEDV', axis=1, inplace=True)

data.to_csv('reduced_data.csv', index=False)

dataset=pd.read_csv('reduced_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,8].values

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=1/3,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
test_rmse=(np.sqrt(mean_squared_error(y_test, y_pred)))
test_r2=r2_score(y_test, y_pred)

# y_pred=regressor.predict(x_train)
# test_rmse=(np.sqrt(mean_squared_error(y_train, y_pred)))
# test_r2=r2_score(y_train, y_pred)


print(test_rmse)
print(test_r2)








#plot graph to check difference between predicted and normal outcome

# fig,ax = plt.subplots()
# ax.scatter(y_test,y_pred)
# #ax.set_x_label('Measured')
# #ax.set_y_label('Predicted')
# ax.plot([y_test.min(),y_test.max()],[y_pred.min(),y_pred.max()],'k--',lw=4)
# ax.set_title('Actual(x) vs Predicted(y)')
# fig.show()
# plt.savefig('Actual vs Predicted.pdf')