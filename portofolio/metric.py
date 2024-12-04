'''
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

x=np.array([8,4,12,4,32,2,9,7]).reshape(8,1)
print(x)
t=MinMaxScaler()
t.fit(x)

xs=t.transform(x)
print(xs)
'''
#metric
#mse= mean_squared_error
#untuk mencari selisih dari nilai asli dengan nilai prediksi
from sklearn.metrics import mean_squared_error#membuat selisih antara nilai asli dan nilai prediksi
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

x,y=make_regression(n_samples=20, n_features=1,random_state=42,noise=3.0)
Pipeline = Pipeline([
    ('scale',MinMaxScaler()),
    ('regression',LinearRegression())
])

Pipeline.fit(x,y)
p=Pipeline.predict(x)
plt.plot(x,y,'b.')
plt.plot(x, p,'r-')
plt.show()
print(mean_squared_error(y,p))
print(Pipeline.score(x,y))

