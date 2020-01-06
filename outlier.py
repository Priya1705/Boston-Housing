import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

data = pd.read_csv('data.csv')
# print(data.head())

# print(data.CRIM.describe())
# bad
value1=data.CRIM
# bad
value2=data.ZN
# good
value3=data.INDUS
# okay
value4=data.CHAS
# good
value5=data.NOX
# okay
value6=data.RM
# good
value7=data.AGE
# okay
value8=data.DIS
# good
value9=data.RAD
# good
value10=data.TAX
# okay
value11=data.PTRATIO
# disaster
value12=data.B
# okaish
value13=data.LSTAT
# okaish
value14=data.MEDV

box=[value1,value2,value3,value4,value5,value6,value7,value8,value9,value10,value11,value12,value13,value14]
plt.boxplot(box)

# value1=np.random.random_integers(1,100,5)

# plt.hist(value1 , bins=1)
# plt.ylabel('no. of times')
plt.show()