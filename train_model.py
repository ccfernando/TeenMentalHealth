"""
Simple teen mental health project.

This script:
1. Loads the CSV file
2. Cleans the data
3. Trains a model
4. Makes predictions
5. Saves the model as model.pkl
"""

import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Step 1: Load the dataset.
data = pd.read_csv("data/mental_health.csv")

# Step 2: Choose what we want to predict.
# In this dataset, we will predict the stress level.
target = "stress_level"

# X = input columns, y = output column.
X = data.drop(columns=[target])
y = data[target]

# Step 3: Separate number columns and text columns.
numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# Fill missing values in number columns with the median.
numeric_transformer = SimpleImputer(strategy="median")

# Fill missing values in text columns, then convert text into numbers.
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine both preprocessing steps into one object.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

# Step 4: Create the full machine learning pipeline.
# This pipeline first preprocesses the data, then trains the model.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Step 5: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train the model.
model.fit(X_train, y_train)

# Step 7: Test the model with a few examples.
predictions = model.predict(X_test.head(5))

print("Sample predictions:")
for i, prediction in enumerate(predictions, start=1):
    print(f"Example {i}: Predicted stress level = {prediction}")

# Step 8: Save the trained model.
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")
