'''
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

x,y=make_regression(n_samples=20, n_features=1,random_state=42,noise=3.0)
r =LinearRegression()
r.fit(x,y)
p=r.predict(x)
plt.plot(x,y,'b.')
plt.plot(x, p,'r-')
plt.show()

from sklearn.preprocessing import MinMaxScaler
t=MinMaxScaler()
t.fit(x)
scaled=t.transform(x)
print(scaled)

#pipelines, series of transformer with estimator

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression#membuat data
from sklearn.preprocessing import MinMaxScaler#tranform
from sklearn.linear_model import LinearRegression#predict
import matplotlib.pyplot as plt
x,y = make_regression(n_samples=20,n_features=1,random_state=1,noise=6.0)

Pipeline = Pipeline([
    ('scale',MinMaxScaler()),
    ('regression',LinearRegression())
])

Pipeline.fit(x,y)
p=Pipeline.predict(x)
plt.plot(x,y,'b.')
plt.plot(x, p,'r-')
plt.show()
'''

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x=np.array([8,4,12,4,32,2,9,7]).reshape(8,1)
y=np.array([42,88,21,80,28,18,68,78]).reshape(8,1)
r =LinearRegression()
r.fit(x,y)
p=r.predict(x)
plt.plot(x,y,'b.')
plt.plot(x, p,'r-')
plt.show()