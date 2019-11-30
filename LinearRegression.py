#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data
df=pd.read_csv("winequality-white.csv",sep=";")
df.head()

#Splitting Independent and Dependent Variables
#Feature Matrix
x=df.iloc[:,:-1] 
#Target Vector
y=df.iloc[:,11]

#Splitting Training and Testing Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)
print(y_test)

#Fitting the LinearRegression Model
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_train,y_train)
y_predict=linear_reg.predict(x_test)

#Rounding the floating val to integer value to predict quality of the wine
for i in range(len(y_predict)):
  y_predict[i]=round(y_predict[i])
print(y_predict)

#Accuracy of the Model
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_predict))
print("Root Mean Squared Error:",rmse)
