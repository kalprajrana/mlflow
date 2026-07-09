from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# import dagshub
# dagshub.init(repo_owner='kalpraj.rana.in', repo_name='mlflow', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/kalpraj.rana.in/mlflow.mlflow")

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the diagnosis target column
df['diagnosis'] = data.target

# Show the first 5 rows
print(df.head())

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.autolog()
mlflow.set_experiment('Breast_Cancer_Classification')

with mlflow.start_run():
    model = MLPClassifier(
        hidden_layer_sizes=(8,8,4),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro"))

    mlflow.log_artifact(__file__)

    mlflow.set_tag("author", "Person-1")


