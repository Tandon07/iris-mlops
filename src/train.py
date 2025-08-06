import pandas as pd
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv("data/iris.csv")
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name, params):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        print(f"{model_name} Accuracy:", acc)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Logistic Regression
    lr_params = {"C": 1.0, "max_iter": 80}
    lr_model = LogisticRegression(**lr_params)
    train_and_log_model(lr_model, "LogisticRegressionModel", lr_params)

    # Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 2}
    rf_model = RandomForestClassifier(**rf_params)
    train_and_log_model(rf_model, "RandomForestModel", rf_params)

