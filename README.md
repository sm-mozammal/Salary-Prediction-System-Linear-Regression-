# Salary Prediction using Linear Regression

## Project Overview
This project implements a simple linear regression approach to predict employee salary from input features. It exposes a FastAPI service for running predictions and includes scripts to train the model.

## Linear Regression (brief)
- Type: Supervised learning (regression).
- Model form: $y = w_1 x_1 + w_2 x_2 + \\\dots + b$ (a weighted sum plus bias).
- Training objective: minimize mean squared error (MSE) between predicted and actual salaries.
- Common solvers: closed-form normal equation or iterative gradient descent.
- Evaluation: MSE, MAE, and $R^2$ score measure how well the model fits the data.

## How this project is organized
- `data.csv` — dataset with input features and target salary values.
- `traine.py` — training script (loads data, trains the linear regression model, and saves the model or parameters).
- `main.py` — FastAPI app for running predictions.
- `requirements.txt` — Python dependencies needed to run the code.

## Quick start
1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python traine.py
```

4. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Then send requests to the API endpoints defined in `main.py`.

If `traine.py` saves model parameters or a serialized model, the FastAPI app should load those to perform predictions. If not, run training first and then adapt `main.py` to load the trained weights.

## Notes & tips
- Inspect `traine.py` to see which features are used and whether any preprocessing (scaling, encoding) is applied — match that when sending inputs to `main.py`.
- If you want more robust results, consider using scikit-learn's `LinearRegression` or `Pipeline` utilities and evaluate with cross-validation.
- Add a small example in `main.py` or a new `example_inputs.txt` showing how to format feature inputs for predictions.

If you'd like, I can open `traine.py` and `main.py`, add example CLI arguments, or include a saved example so running predictions is immediate.
