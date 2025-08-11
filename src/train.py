import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri("http://localhost:5000")


def load_data():
    df = pd.read_csv("data/iris.csv")

    # Ensure correct data types
    df = df.apply(pd.to_numeric, errors='ignore')

    # Handle missing values for features
    imputer = SimpleImputer(strategy="mean")
    feature_cols = df.drop("target", axis=1).columns
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # Encode categorical target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-Test Split
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_log_model(model, model_name, params, scale=False):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        if scale:
            # Create pipeline for scaling + model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            mlflow.sklearn.log_model(pipeline, "model",
                                     registered_model_name=model_name)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mlflow.sklearn.log_model(model, "model",
                                     registered_model_name=model_name)

        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        print(f"{model_name} Accuracy:", acc)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Logistic Regression (scaling important)
    lr_params = {"C": 1.0, "max_iter": 80}
    lr_model = LogisticRegression(**lr_params)
    train_and_log_model(lr_model, "LogisticRegressionModel",
                        lr_params, scale=True)

    # Random Forest (scaling not necessary)
    rf_params = {"n_estimators": 100, "max_depth": 2}
    rf_model = RandomForestClassifier(**rf_params)
    train_and_log_model(rf_model, "RandomForestModel", rf_params, scale=False)
