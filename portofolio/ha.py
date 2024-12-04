from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=load_diabetes()
df=pd.DataFrame(data.data,columns=data.feature_names)
x=df[['bmi']]
y=df['bp']
plt.plot(x,y,'b.')
from sklearn.preprocessing import PolynomialFeatures
pf1= PolynomialFeatures(degree=10,include_bias=False)
pf2= PolynomialFeatures(degree=15,include_bias=False)
pf3= PolynomialFeatures(degree=20,include_bias=False)
pf4= PolynomialFeatures(degree=30,include_bias=False)

x_p1=pf1.fit_transform(x)
x_p2=pf2.fit_transform(x)
x_p3=pf3.fit_transform(x)
x_p4=pf4.fit_transform(x)

from sklearn.linear_model import LinearRegression

r=LinearRegression()
r.fit(x_p1,y)
r.fit(x_p2,y)
r.fit(x_p3,y)
r.fit(x_p4,y)

x_lf = np.linspace(x.min(),x.max(),num=100)
y_lf = r.intercept_

for i in range(len(pf1.powers_)):
  exp = pf1.powers_[i][0]
  y_lf1 = y_lf + r.coef_[i] * (x_lf ** exp)
for i in range(len(pf2.powers_)):
  exp = pf2.powers_[i][0]
  y_lf2 = y_lf + r.coef_[i] * (x_lf ** exp)
for i in range(len(pf3.powers_)):
  exp = pf3.powers_[i][0]
  y_lf3 = y_lf + r.coef_[i] * (x_lf ** exp)
for i in range(len(pf4.powers_)):
  exp = pf4.powers_[i][0]
  y_lf4 = y_lf + r.coef_[i] * (x_lf ** exp)

import matplotlib.pyplot as plt

f=plt.figure()
s=f.add_subplot(111)
s.plot(x_lf,y_lf1,'b.',label='degree 10',color='g')
s.plot(x_lf,y_lf2,'b.',label='degree 15',color='y')
s.plot(x_lf,y_lf3,'b.',label='degree 20',color='r')
s.plot(x_lf,y_lf4,'b.',label='degree 30',color='b')
s.legend()
plt.show()
