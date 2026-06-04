from fastapi import FastAPI
import joblib


app = FastAPI()

model = joblib.load('model.pkl')

@app.get('/')
def home():
    return {'message': 'Welcome to the Salary Prediction API'}

@app.post('/predict')
def predict(experience: float):
    prediction = model.predict([[experience]])
    return {
        'experience': experience,
        'predicted_salary': prediction[0]
        }