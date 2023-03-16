#Name: EMBERGA, Felix Miguel D.
#Date Submitted: 12/8/2022
#import assets needed for the program
import numpy as np
from sklearn.linear_model import LinearRegression
#Assign Values into Array
x = np.array([0,500,1000, 1500, 2000, 2500, 3000, 3500, 4000,
4500, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
13000, 14000, 15000]).reshape(-1,1)
y = np.array([1.225, 1.167, 1.112, 1.058, 1.006, 0.957,
0.909, 0.863, 0.819, 0.777, 0.736, 0.66, 0.59, 0.526,
0.467, 0.413, 0.365, 0.312, 0.226, 0.228, 0.195])
#main
model = LinearRegression().fit(x,y)
r_sq = model.score(x, y)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept: {new_model.intercept_}")
print(f"slope: {new_model.coef_}")
#predict y value with following variable
new_x = np.array(
    [1875,
    2250,
    5423,
    10004,
    12137]).reshape(-1,1)
y_pred = model.predict(new_x)
print(f"predicted response:\n{y_pred}")