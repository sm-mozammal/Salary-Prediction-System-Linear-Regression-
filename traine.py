import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


#load data
data = pd.read_csv('data.csv')

x = data[['experience']]
y = data['salary']

print(x)
print(y)

model = LinearRegression()

model.fit(x,y)

joblib.dump(model, 'model.pkl')

print('Model trained and saved as model.pkl')


