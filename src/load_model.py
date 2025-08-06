import mlflow.pyfunc
mlflow.set_tracking_uri("http://localhost:5000")

# Load the best registered model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/LogisticRegressionModel/Production")

# Example input (based on Iris features)
sample_input = {
    "sepal_length_(cm)": 5.1,
    "sepal_width_(cm)": 3.5,
    "petal_length_(cm)": 1.4,
    "petal_width_(cm)": 0.2
}

import pandas as pd
input_df = pd.DataFrame([sample_input])

# Make prediction
prediction = model.predict(input_df)
print("Prediction:", prediction)
