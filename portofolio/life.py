import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a = pd.read_csv('life_expectancy_years.csv')
'''
print(a.head())

condition=a['country']=='Indonesia'
print(a[condition])

idr_life=a.iloc[77][1:].values
print(idr_life)

years=np.array(range(int(a.columns[1:].min()), int(a.columns[1:].max())+1))

print(years)

plt.plot(years,idr_life,'b.')
plt.show()

c1=a['country']=='Japan'
c2=a['country']=='Indonesia'
c3=a['country']=='Zimbabwe'

j=a.iloc[84][1:].values
i=a.iloc[77][1:].values
z=a.iloc[186][1:].values

years=np.array(range(int(a.columns[1:].min()), int(a.columns[1:].max())+1))

f=plt.figure()
s=f.add_subplot(111)
s.plot(years,j,'b.',label='Japan',color='g')
s.plot(years,i,'b.',label='Indonesia',color='y')
s.plot(years,z,'b.',label='Zimbabwe',color='r')
s.set_title('Perbandingan Negara')
s.annotate('WW2',xy=(1939,50),xytext=(1910,60),arrowprops=dict(facecolor='black',shrink=1))
s.legend()
plt.show()
#'''
idr_life=a.iloc[77][1:].values
years=np.array(range(int(a.columns[1:].min()), int(a.columns[1:].max())+1))
df = pd.DataFrame()
df['years']=years
df['Life Expentancy']=idr_life

from sklearn.model_selection import train_test_split
x=df[['years']]
y=df['Life Expentancy']
x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)#test size 20 nya jd ulangan
print('data training')
print(x_train.shape)
print(y_train.shape)
print('data testing')
print(x_test.shape)
print(y_test.shape)
print('data asli')
print(x.shape)
print(y.shape)

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

m=LinearRegression()
m.fit(x_train, y_train)
pred = m.predict(x_test)

mse=mean_squared_error(y_test,pred)
print(mse)