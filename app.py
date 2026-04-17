import os
import pickle

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)


# Load the trained model only once when the app starts.
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# These are the expected input columns.
# We prefer the columns stored inside the trained model so the app stays in sync
# with training. The fallback list keeps the app beginner-friendly and explicit.
FEATURE_COLUMNS = list(
    getattr(
        model,
        "feature_names_in_",
        [
            "age",
            "gender",
            "daily_social_media_hours",
            "platform_usage",
            "sleep_hours",
            "screen_time_before_sleep",
            "academic_performance",
            "physical_activity",
            "social_interaction_level",
            "stress_level",
            "anxiety_level",
            "addiction_level",
        ],
    )
)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    confidence_text = None

    if request.method == "POST":
        try:
            # Collect and convert the form values.
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
                "stress_level": float(request.form["stress_level"]),
                "anxiety_level": float(request.form["anxiety_level"]),
                "addiction_level": float(request.form["addiction_level"]),
            }

            # Create a DataFrame because scikit-learn pipelines expect tabular input.
            input_data = pd.DataFrame([user_data], columns=FEATURE_COLUMNS)

            # Make the prediction.
            prediction = int(model.predict(input_data)[0])

            if prediction == 1:
                prediction_text = "At Risk of Depression"
            else:
                prediction_text = "Not At Risk"

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                class_labels = list(model.classes_)
                predicted_index = class_labels.index(prediction)
                confidence = probabilities[predicted_index] * 100
                confidence_text = f"{confidence:.1f}%"

        except Exception:
            prediction_text = (
                "Something went wrong while making the prediction. "
                "Please check your inputs and try again."
            )
            confidence_text = None

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        confidence_text=confidence_text,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
