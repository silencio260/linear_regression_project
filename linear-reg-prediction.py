import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
dataset = pd.read_csv('Salary_Data.csv')
dataset.head()

print(dataset)

#Data Preprocessing

X = dataset.iloc[:,:-1].values  #independent variable array
y = dataset.iloc[:,1].values  #dependent variable vector


#Split Data into Test and Train Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)


#Fitting linear regression model into the training set
#From sklearn’s linear model library, import linear regression class. 
#Create an object for a linear regression class called regressor.
#To fit the regressor into the training set, 
#we will call the fit method – function to fit the regressor into the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train) #actually produces the linear eqn for the data



#. Predicting the test set results
#We create a vector containing all the predictions of the test set salaries. 
#The predicted salaries are then put into the vector called y_pred.
#(contains prediction for all observations in the test set)

#predict method makes the predictions for the test set.
# Hence, the input is the test set. 
#The parameter for predict must be an array or sparse matrix, hence input is X_test


y_pred = regressor.predict(X_test) 
print("--Prediction Salary---")
print(y_pred)
print("---Real Salary----")
print(y_test)

#Visualize the result


#plot for the TRAIN
#Plot the points
 
plt.scatter(X_train, y_train, color='red') # plotting the observation line
 
plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line
 
plt.title("Salary vs Experience (Training set)") # stating the title of the graph
 
plt.xlabel("Years of experience") # adding the name of x-axis
plt.ylabel("Salaries") # adding the name of y-axis
plt.show() # specifies end of graph



#plot for the TEST
 
plt.scatter(X_test, y_test, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line
 
plt.title("Salary vs Experience (Testing set)")
 
plt.xlabel("Years of experience") 
plt.ylabel("Salaries") 
plt.show()

