from sklearn.datasets import load_iris


def load_iris_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df


if __name__ == "__main__":
    df = load_iris_data()
    df.to_csv("data/Iris.csv", index=False)
