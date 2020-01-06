import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

data = pd.read_csv('HousingData.csv')
print(data.head())
# x = dataset.iloc[:,:-1].values
# y = dataset.iloc[:,13].values
dataset=data.iloc[:,:]

imputer = Imputer(missing_values='NaN',strategy="mean")
imputer = imputer.fit(dataset)
dataset = imputer.transform(dataset)

dataset=pd.DataFrame(dataset)
print(dataset.head())

dataset.to_csv('data.csv', index=False, header=False)