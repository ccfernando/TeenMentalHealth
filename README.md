# Teen Mental Health Predictor

This project is a simple machine learning web application built with Python, Flask, pandas, and scikit-learn.

It uses a dataset of teen lifestyle and mental health indicators to predict a student's `stress_level`.

The project has two main parts:

- `train_model.py` trains the machine learning model and saves it as `model.pkl`
- `app.py` runs a Flask web app where a user can enter values in a form and get a predicted stress level

## Features

- Loads data from `data/mental_health.csv`
- Preprocesses numeric and categorical data automatically
- Trains a Logistic Regression model using a scikit-learn pipeline
- Saves the full trained pipeline into `model.pkl`
- Runs a browser-based form for making predictions

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

## Requirements

Before running the project, make sure you have:

- Python 3.10 or newer installed
- `pip` installed and working
- Internet access for installing dependencies the first time

## How To Run The Project

### 1. Download or clone the project

If you already have the files locally, you can skip this step.

To clone from GitHub:

```bash
git clone https://github.com/ccfernando/TeenMentalHealth.git
cd TeenMentalHealth
```

### 2. Create a virtual environment

Using a virtual environment is recommended so the project dependencies do not affect other Python projects on your computer.

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

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

This installs:

- `Flask`
- `pandas`
- `scikit-learn`

### 4. Train the model

Run the training script to create or refresh the saved model file:

```bash
python train_model.py
```

What this script does:

- reads the dataset from `data/mental_health.csv`
- separates the target column `stress_level`
- fills missing values
- converts categorical text values into numeric features
- trains a Logistic Regression model
- saves the trained pipeline to `model.pkl`

After it finishes, you should see output in the terminal showing sample predictions and a message confirming that `model.pkl` was saved.

### 5. Start the Flask web app

```bash
python app.py
```

If the app starts correctly, Flask will show a local development server address similar to:

```text
http://127.0.0.1:5000/
```

Open that address in your browser.

### 6. Use the prediction form

On the web page, enter the student details such as:

- age
- gender
- daily social media hours
- platform usage
- sleep hours
- screen time before sleep
- academic performance
- physical activity
- social interaction level
- anxiety level
- addiction level
- depression label

After clicking `Predict`, the app will display the predicted `stress_level`.

## Quick Start

If you want the shortest setup flow:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Files Explained

### `app.py`

This is the Flask application.

- loads `model.pkl`
- shows the input form
- collects submitted form values
- converts the values into a pandas DataFrame
- sends the data to the trained model for prediction
- displays the predicted stress level on the page

### `train_model.py`

This is the machine learning training script.

- loads the dataset
- splits input features and target
- preprocesses the data
- trains the model
- saves the model to `model.pkl`

### `templates/index.html`

This file contains the HTML form and inline CSS for the web interface.

### `data/mental_health.csv`

This is the dataset used for training.

### `model.pkl`

This is the saved trained model pipeline used by the Flask app.

## Notes

- The app predicts `stress_level`, not a medical diagnosis.
- This project is best treated as a school or learning project.
- `debug=True` is enabled in `app.py`, which is useful during development but should be disabled for production deployment.
- If you retrain the model, `model.pkl` will be overwritten with the newest trained version.

## Troubleshooting

### `ModuleNotFoundError`

Install dependencies again:

```bash
pip install -r requirements.txt
```

### `FileNotFoundError: model.pkl`

Run the training script first:

```bash
python train_model.py
```

### Flask page does not open

- make sure `python app.py` is still running
- check that you opened `http://127.0.0.1:5000/`
- confirm no other app is already using port `5000`

## GitHub

Repository:

`https://github.com/ccfernando/TeenMentalHealth`
