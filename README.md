# Teen Mental Health Predictor

This project is a beginner-friendly Flask machine learning web app that predicts whether a student is at risk of depression.

The app uses a scikit-learn `Pipeline` with:

- `SimpleImputer` for missing values
- `OneHotEncoder` for categorical features
- `ColumnTransformer` for preprocessing
- `LogisticRegression` for classification

The trained pipeline is saved as `model.pkl` and loaded by the Flask app for predictions.

## What The App Predicts

The model predicts:

- `depression_label = 1` -> `At Risk of Depression`
- `depression_label = 0` -> `Not At Risk`

This is a school or learning project and should not be treated as a medical diagnosis tool.

## Features

- Trains a machine learning model from `data/mental_health.csv`
- Uses a full scikit-learn pipeline so preprocessing and prediction stay together
- Displays training metrics such as accuracy and log loss
- Saves the trained model as `model.pkl`
- Runs a Flask web app with a simple HTML form
- Shows the prediction result in a modal
- Includes basic error handling for failed predictions
- Works with deployment platforms like Render

## Project Structure

```text
TeenMentalHealth/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- model.pkl
|-- README.md
|-- data/
|   `-- mental_health.csv
`-- templates/
    `-- index.html
```

## Dataset Columns

The dataset includes these columns:

- `age`
- `gender`
- `daily_social_media_hours`
- `platform_usage`
- `sleep_hours`
- `screen_time_before_sleep`
- `academic_performance`
- `physical_activity`
- `social_interaction_level`
- `stress_level`
- `anxiety_level`
- `addiction_level`
- `depression_label`

The training script uses `depression_label` as the target column and uses the remaining columns as input features.

## Requirements

Before running the project, make sure you have:

- Python 3.10 or newer
- `pip`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ccfernando/TeenMentalHealth.git
cd TeenMentalHealth
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train The Model

Run:

```bash
python train_model.py
```

The training script:

- loads the dataset from `data/mental_health.csv`
- sets `depression_label` as the prediction target
- uses `X = data.drop(columns=[target])` and `y = data[target]`
- preprocesses numeric and categorical columns
- trains a Logistic Regression model
- prints training accuracy, training loss, validation loss, validation accuracy, and final accuracy
- saves the updated pipeline as `model.pkl`

Example output includes:

- training accuracy
- training loss
- validation loss
- validation accuracy
- final accuracy

## Run The Flask App

Start the web app:

```bash
python app.py
```

Then open the local server in your browser. During local development, it will usually be:

```text
http://127.0.0.1:10000/
```

The app is configured with:

- `host="0.0.0.0"`
- `port=os.environ.get("PORT", 10000)`

This makes it easier to deploy on Render.

## Form Inputs

The Flask app sends form values into a pandas `DataFrame` before prediction.

The current input fields are:

- `age`
- `gender`
- `daily_social_media_hours`
- `platform_usage`
- `sleep_hours`
- `screen_time_before_sleep`
- `academic_performance`
- `physical_activity`
- `social_interaction_level`
- `stress_level`
- `anxiety_level`
- `addiction_level`

Important:

- `depression_label` is not entered by the user because it is the prediction target
- the input columns must match the training features exactly
- categorical values are case-sensitive and must match the dataset exactly

Current categorical values:

- `gender`: `male`, `female`
- `platform_usage`: `Instagram`, `TikTok`, `Both`
- `social_interaction_level`: `low`, `medium`, `high`

## Prediction Output

When the form is submitted:

- `1` is displayed as `At Risk of Depression`
- `0` is displayed as `Not At Risk`

The result is shown in a modal in the HTML interface.

## Files Explained

### `train_model.py`

This script trains the machine learning model and saves it as `model.pkl`.

### `app.py`

This is the Flask application. It:

- loads the saved model once at startup
- collects user input from the form
- converts the input into a pandas `DataFrame`
- makes a prediction with `model.predict()`
- returns a user-friendly result message

### `templates/index.html`

This file contains the form, page styling, and the modal used to display prediction results.

### `data/mental_health.csv`

This is the dataset used for training the model.

### `model.pkl`

This is the saved trained scikit-learn pipeline used by the Flask app.

## Error Handling

The Flask app wraps prediction in a `try/except` block. If something fails, the page shows a simple user-friendly error message instead of crashing.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

## Troubleshooting

### `ModuleNotFoundError`

Install the dependencies again:

```bash
pip install -r requirements.txt
```

### `FileNotFoundError: model.pkl`

Train the model first:

```bash
python train_model.py
```

### Form submission fails

- make sure all fields are filled in
- make sure categorical values match the dataset exactly
- make sure `model.pkl` was created from the current training script

## GitHub

Repository:

`https://github.com/ccfernando/TeenMentalHealth`
