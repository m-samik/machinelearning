import numpy
import pandas as pd
from sklearn.linear_model import LinearRegression


data= pd.read_csv('salary_data.csv')

y= data['Salary']
x= data['YearsExperience']
x=x.values.reshape(-1,1)

model = LinearRegression()

model.fit(x,y)

p=int(input("Enter The Experience to which Salary will get predicted :   "))
out=model.predict([[p]])

print(out)
