"""
Train a teen mental health model and save it as model.pkl.

This version predicts `depression_label` using the rest of the dataset
columns as input features.
"""

import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Step 1: Load the dataset.
data = pd.read_csv("data/mental_health.csv")

# Step 2: Choose the target column we want to predict.
target = "depression_label"

# Step 3: Split the dataset into features (X) and target (y).
# We keep every other column as an input feature.
X = data.drop(columns=[target])
y = data[target]

# Step 4: Separate numeric columns and categorical columns.
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# Numeric columns: fill missing values with the median.
numeric_transformer = SimpleImputer(strategy="median")

# Categorical columns: fill missing values, then one-hot encode them.
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Step 5: Build the preprocessing object.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

# Step 6: Build the full machine learning pipeline.
# The pipeline keeps preprocessing and model training together.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Step 7: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train the model.
model.fit(X_train, y_train)

# Step 9: Measure how well the model performed.
# Accuracy shows how many predictions were correct.
# Log loss shows how confident and correct the probability predictions were.
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_probabilities = model.predict_proba(X_train)
test_probabilities = model.predict_proba(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
train_loss = log_loss(y_train, train_probabilities)
test_loss = log_loss(y_test, test_probabilities)

print("Training features used by the model:")
for column in X.columns:
    print(f"- {column}")

print("\nTraining results:")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Training loss: {train_loss:.4f}")
print(f"Validation loss: {test_loss:.4f}")
print(f"Validation accuracy: {test_accuracy:.4f}")

# Step 10: Show a few example predictions.
predictions = model.predict(X_test.head(5))

print("\nSample predictions:")
for i, prediction in enumerate(predictions, start=1):
    print(f"Example {i}: Predicted depression label = {prediction}")

# Step 11: Save the trained pipeline.
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nUpdated model saved as model.pkl")
print(f"Final accuracy: {test_accuracy:.4f}")
