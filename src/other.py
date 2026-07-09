import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("YT-MLOPS-Greedsearch-Exp")

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42
)

# Hyperparameter Grid
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [5, 10, 15]
}

best_accuracy = 0
best_params = None

# Iterate over all parameter combinations
for params in ParameterGrid(param_grid):

    with mlflow.start_run():

        # Train model
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        rf.fit(X_train, y_train)

        # Predictions
        y_pred = rf.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log Parameters
        mlflow.log_params(params)

        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Tags
        mlflow.set_tags({
            "Author": "Person-2",
            "Project": "Wine Classification"
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=wine.target_names,
            yticklabels=wine.target_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(
            f"CM_MD{params['max_depth']}_NE{params['n_estimators']}"
        )

        filename = (
            f"cm_md{params['max_depth']}"
            f"_ne{params['n_estimators']}.png"
        )

        plt.savefig(filename)
        plt.close()

        # Log Artifacts
        mlflow.log_artifact(filename)

        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

        # If running from a .py file
        # mlflow.log_artifact(__file__)

        # Log Model
        mlflow.sklearn.log_model(rf, "Random-Forest-Model")

        print(
            f"Params: {params} "
            f"Accuracy: {accuracy:.4f}"
        )

        # Track Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

print("\nBest Accuracy:", best_accuracy)
print("Best Parameters:", best_params)