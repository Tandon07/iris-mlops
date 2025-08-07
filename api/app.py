from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from logger import logging
import sqlite3
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

mlflow.set_tracking_uri("http://localhost:5000")

app = FastAPI(title="Iris Classifier API")

# NEW: Attach Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

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


iris_predictions = Counter("iris_predictions_total", "Total number of predictions made")


@app.post("/predict")
def predict(features: IrisFeatures):
    iris_predictions.inc()
    input_df = pd.DataFrame([{
        "sepal_length_(cm)": features.sepal_length_cm,
        "sepal_width_(cm)": features.sepal_width_cm,
        "petal_length_(cm)": features.petal_length_cm,
        "petal_width_(cm)": features.petal_width_cm
    }])
    prediction = model.predict(input_df)
    predicted_class = int(prediction[0])

    # Logging to file
    logging.info(f"Input: {input_df.to_dict(orient='records')[0]} --> Prediction: {predicted_class}")

    # Logging to SQLite
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction INTEGER
        )
    ''')
    cursor.execute('''
        INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        str(datetime.now()),
        features.sepal_length_cm,
        features.sepal_width_cm,
        features.petal_length_cm,
        features.petal_width_cm,
        predicted_class
    ))
    conn.commit()
    conn.close()

    return {"prediction": predicted_class}
