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

FIELD_RULES = {
    "age": {"type": "int", "min": 13, "max": 19},
    "gender": {"type": "choice", "choices": ["male", "female"]},
    "daily_social_media_hours": {"type": "float", "min": 1.0, "max": 8.0},
    "platform_usage": {"type": "choice", "choices": ["Instagram", "TikTok", "Both"]},
    "sleep_hours": {"type": "float", "min": 4.0, "max": 9.0},
    "screen_time_before_sleep": {"type": "float", "min": 0.5, "max": 3.0},
    "academic_performance": {"type": "float", "min": 2.0, "max": 4.0},
    "physical_activity": {"type": "float", "min": 0.0, "max": 2.0},
    "social_interaction_level": {
        "type": "choice",
        "choices": ["low", "medium", "high"],
    },
    "stress_level": {"type": "int", "min": 1, "max": 10},
    "anxiety_level": {"type": "int", "min": 1, "max": 10},
    "addiction_level": {"type": "int", "min": 1, "max": 10},
}


def validate_form_input(form_data):
    user_data = {}

    for field_name, rules in FIELD_RULES.items():
        raw_value = form_data.get(field_name, "").strip()

        if raw_value == "":
            raise ValueError(f"{field_name.replace('_', ' ').title()} is required.")

        if rules["type"] == "choice":
            if raw_value not in rules["choices"]:
                allowed_values = ", ".join(rules["choices"])
                raise ValueError(
                    f"{field_name.replace('_', ' ').title()} must be one of: {allowed_values}."
                )
            user_data[field_name] = raw_value
            continue

        try:
            numeric_value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"{field_name.replace('_', ' ').title()} must be a number."
            ) from error

        if rules["type"] == "int" and not numeric_value.is_integer():
            raise ValueError(
                f"{field_name.replace('_', ' ').title()} must be a whole number."
            )

        if numeric_value < rules["min"] or numeric_value > rules["max"]:
            raise ValueError(
                f"{field_name.replace('_', ' ').title()} must be between "
                f"{rules['min']} and {rules['max']}."
            )

        if rules["type"] == "int":
            user_data[field_name] = int(numeric_value)
        else:
            user_data[field_name] = numeric_value

    return user_data


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    confidence_text = None
    error_text = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            user_data = validate_form_input(form_data)

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

        except ValueError as error:
            error_text = str(error)
            prediction_text = None
            confidence_text = None
        except Exception:
            error_text = (
                "Something went wrong while making the prediction. "
                "Please check your inputs and try again."
            )
            prediction_text = None
            confidence_text = None

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        confidence_text=confidence_text,
        error_text=error_text,
        form_data=form_data,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
