from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
mlflow.set_tracking_uri("http://localhost:5000")
import pandas as pd

app = FastAPI(title="Iris Classifier API")

# Load the MLflow model from registry
model = mlflow.pyfunc.load_model("models:/LogisticRegressionModel/Production")

# Define the input schema using Pydantic
class IrisFeatures(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/predict")
def predict(features: IrisFeatures):
    input_df = pd.DataFrame([{
        "sepal_length_(cm)": features.sepal_length_cm,
        "sepal_width_(cm)": features.sepal_width_cm,
        "petal_length_(cm)": features.petal_length_cm,
        "petal_width_(cm)": features.petal_width_cm
    }])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
