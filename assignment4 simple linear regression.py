# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:55:46 2023

@author: 91913
"""
#1) Delivery_time -> Predict delivery time using sorting time 
#Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

#step1 : import the data files
import pandas as pd
import numpy as np
df = pd.read_csv("delivery_time.csv")
df.shape
df.head()

# step2 : EDA
import matplotlib.pyplot as plt
plt.scatter(df['Delivery Time'],df['Sorting Time'])
plt.show()
df.corr()

# split as x and y variables
Y = df["Delivery Time"]
X = df[["Sorting Time"]]


# step3 : fitting the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)


#step4 : calculating the metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

#==============================================================================

#==============================================================================
#2) Salary_hike -> Build a prediction model for Salary_hike
#Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

#step1 : import the data files
import pandas as pd
import numpy as np
df = pd.read_csv("Salary_Data.csv")
df.shape
df.head()

# step2 : EDA
import matplotlib.pyplot as plt
plt.scatter(df['Salary'],df['YearsExperience'])
plt.show()
df.corr()

# split as x and y variables
Y = df["Salary"]
X = df[["YearsExperience"]]


# step3 : fitting the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)


#step4 : calculating the metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

#==============================================================================