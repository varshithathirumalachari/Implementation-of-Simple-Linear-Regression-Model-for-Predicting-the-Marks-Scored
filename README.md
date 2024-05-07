# EX-2Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the needed packages

2.Assigning hours To X and Scores to Y

3.Plot the scatter plot

4.Use mse,rmse,mae formmula to find 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VARSHITHA A T
RegisterNumber:  212221040176
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print("df.head")

df.head()

print("df.tail")

df.tail()

Y=df.iloc[:,1].values
print("Array of Y")
Y

X=df.iloc[:,:-1].values
print("Array of X")
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Array values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
## df.head():
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/5f6b2573-e2cc-4f99-ae06-d8f5a52a0e1b)
## df.tail():
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/70c7b853-a85c-472e-a957-52e6ed569723)
## Array of X:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/717a6515-e9a0-423a-9c7b-59743183716a)
## Array of Y:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/b28b0294-727b-4e1e-ab1f-fdda9e25ed60)
## Y_Pred:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/ae1b73b6-30b9-4ace-8e7f-0230d5978a08)
## y_test:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/f4b2651d-3dd3-4f01-b17c-63cb243b2a76)
## Training set:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/daedd2e2-bc75-4db8-938a-1793e7a5bce4)
## Test set:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/0ec6f281-4bd1-49f5-8e4e-2129a885994c)
## Values of MSE,MAE,RMSE:
![image](https://github.com/varshithathirumalachari/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131793193/36d4003a-d73c-4df1-b16c-704ace2e469b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
