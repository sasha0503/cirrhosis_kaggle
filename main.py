import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


def transform_data(data):
    data["Age"] = data["Age"] / 365
    data["Stage"] = data["Stage"].map({1: "1", 2: "2", 3: "3", 4: "4"})

    for col in data.select_dtypes(include=['object']):
        data[col] = LabelEncoder().fit_transform(data[col])

    numerical_features = data.select_dtypes(include=["int64", "float64"]).columns
    numerical_features = numerical_features.drop("id")

    data[numerical_features] = MinMaxScaler().fit_transform(data[numerical_features])
    data[numerical_features] = StandardScaler().fit_transform(data[numerical_features])


if __name__ == '__main__':
    dataframe = pd.read_csv("data/train.csv")
    transform_data(dataframe)