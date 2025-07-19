import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from functions import generate_classification_dataset, generate_regression_dataset

def KNN_fit_and_predict(df: pd.DataFrame):
    y = df["churn"].values
    X = df[["account_length", "customer_service_calls"]].values

    # print(X.shape, y.shape)
    knn_classifier = KNeighborsClassifier(n_neighbors=6)
    knn_classifier.fit(X, y)

    x_new_points = np.array([[30.0, 17.5],
                            [107.0, 24.1],
                            [213.0, 10.9]])

    y_pred = knn_classifier.predict(x_new_points)
    print("Predictions: {}".format(y_pred)) 

def train_test_split_and_compute_accuracy(df: pd.DataFrame):
    X = df.drop("churn", axis=1).values
    y = df["churn"].values

    # test_size equal to 20%
    # random_state to 42
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    print(knn.score(x_test, y_test))
    return x_train, x_test, y_train, y_test

def overfit_and_underfit(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    neighbors = np.arange(1, 13)
    train_accuracies = {}
    test_accuracies = {}

    for neighbor in neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        knn.fit(x_train, y_train)

        # Compute accuracy
        train_accuracies[neighbor] = knn.score(x_train, y_train)
        test_accuracies[neighbor] = knn.score(x_test, y_test)
    # print(neighbors, '\n', train_accuracies, '\n', test_accuracies)
    return neighbors, train_accuracies, test_accuracies

def regression_fit_and_predict(df:pd.DataFrame):
    X = df.drop("sales", axis=1).values
    y = df["sales"].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    return lr, x_train, x_test, y_train, y_test, y_pred

def evaluate_regression_performance(reg: LinearRegression, x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray):
    r_squared = reg.score(x_test, y_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return r_squared, rmse

def perform_cross_validation(df: pd.DataFrame):
    X = df.drop("sales", axis=1).values
    y = df["sales"].values

    kf = KFold(n_splits=6, shuffle=True, random_state=5)
    lr = LinearRegression()
    cv_scores = cross_val_score(lr, X, y, cv=kf)

    print('Cross-Validation Scores: ', cv_scores)
    print('Mean = ', np.mean(cv_scores))
    print('Standard Deviation = ', np.std(cv_scores))
    print('Confidence Interval = ', np.quantile(cv_scores, [0.025, 0.975]))

if __name__ == "__main__":
    data = generate_classification_dataset()

    KNN_fit_and_predict(data)
    features_train, features_test, labels_train, labels_test = train_test_split_and_compute_accuracy(data)
    neighbors, train_accuracies, test_accuracies = overfit_and_underfit(features_train, features_test, labels_train, labels_test)

    plt.title("KNN: Model Complex Curve")
    # visualize how performance changes as the model becomes less complex
    plt.plot(neighbors, list(train_accuracies.values()), label="Training Accuracy")
    plt.plot(neighbors, list(test_accuracies.values()), label="Testing Accuracy")

    plt.legend()
    plt.xlabel("Nb of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()

    data = generate_regression_dataset()

    reg, x_train, x_test, y_train, y_test, y_pred = regression_fit_and_predict(data)
    print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

    r2_score, rmse_score = evaluate_regression_performance(reg, x_test, y_test, y_pred)
    print("R^2: {}".format(r2_score))
    print("RMSE: {}".format(rmse_score))

    perform_cross_validation(data)
    