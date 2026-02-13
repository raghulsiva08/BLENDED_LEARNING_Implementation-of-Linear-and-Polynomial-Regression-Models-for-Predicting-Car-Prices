# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset using pandas.
2.Select input features and target variable (price).
3.Split the data into training and testing sets.
4.Train the Linear Regression model with scaling.
5.Train the Polynomial Regression model (degree 2).
6.Predict prices using both models.
7.Evaluate performance using MSE, MAE, and R² score and compare results.

## Program:
~~~
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by:RAGHUL.S 
RegisterNumber: 212225040325 
*/

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_error
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')

data=data.drop(['car_ID','CarName'],axis=1)#removes unnecessary columns
data = pd.get_dummies(data, drop_first=True)# Handle categorical variables

X = data.drop('price', axis=1)
Y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

print("Name: RAGHUL.S")
print("Reg. No: 212225040325")
print("\n=== Cross-validation ===")
cv_scores=cross_val_score(model,X,Y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")
# 5. Test set evaluation
Y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.2f}")
print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.2f}")
print(f"R²: {r2_score(Y_test, Y_pred):.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()],[Y_test.min(), Y_test.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
~~~

## Output:
![ex2 name](https://github.com/user-attachments/assets/00c1a953-d96b-4c30-b97b-366686c98b74)

![ex2 poly](https://github.com/user-attachments/assets/03bd6eb1-cb54-483d-9c9c-ac1c55e79e5f)

![ex2 graph](https://github.com/user-attachments/assets/e17f4b2a-62d5-48af-bdab-0bf70b90ac6a)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
