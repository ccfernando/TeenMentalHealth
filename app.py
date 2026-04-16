import pickle

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)


# Load the trained model once when the app starts.
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Get values from the form and convert number fields to floats.
        user_data = {
            "age": float(request.form["age"]),
            "gender": request.form["gender"],
            "daily_social_media_hours": float(request.form["daily_social_media_hours"]),
            "platform_usage": request.form["platform_usage"],
            "sleep_hours": float(request.form["sleep_hours"]),
            "screen_time_before_sleep": float(request.form["screen_time_before_sleep"]),
            "academic_performance": float(request.form["academic_performance"]),
            "physical_activity": float(request.form["physical_activity"]),
            "social_interaction_level": request.form["social_interaction_level"],
            "anxiety_level": float(request.form["anxiety_level"]),
            "addiction_level": float(request.form["addiction_level"]),
            "depression_label": float(request.form["depression_label"]),
        }

        # Put the user input into a DataFrame because the model expects table-like data.
        input_data = pd.DataFrame([user_data])

        # Make a prediction using the saved model.
        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
