import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.api as sm

dataset = pd.read_csv('HousingData.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,13].values

imputer = Imputer(missing_values='NaN',strategy="mean")
imputer = imputer.fit(x)
x = imputer.transform(x)

#Using backward elimination, find parameters which actually have an impact

def backwardElimination(x,sl):
    numVars=len(x[0])
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(y,x,sl).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        if(maxVar>sl):
            for j in range(0,numVars-i):
                if(regressor_OLS.pvalues[j].astype(float)==maxVar):
                    x=np.delete(x,j,axis=1)
        #print(regressor_OLS.summary())
    # print(regressor_OLS.summary())
    return x

SL=0.05
x=np.append(arr=np.ones((506,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
x_opt=x_opt.astype(float)
x_modelled=backwardElimination(x_opt,SL)

x_train, x_test, y_train, y_test= train_test_split(x_modelled,y,test_size=1/3,random_state=0)

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