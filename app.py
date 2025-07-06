from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("health_checkup_model.pkl")
preprocessor = joblib.load("health_checkup_preprocessor.pkl")

@app.route('/')
def index():
    return render_template('index.html')  # Make sure templates/index.html exists

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Correct 22 feature names matching your model
        feature_names = [
            "age", "height(cm)", "weight(kg)", "waist(cm)", "eyesight(left)", "eyesight(right)",
            "hearing(left)", "hearing(right)", "systolic", "relaxation", "fasting blood sugar",
            "cholesterol", "hemoglobin", "serum creatinine", "ast", "alt", "gtp",
            "ldl", "hdl", "triglyceride", "urine protein", "gender"
        ]

        # Collect input from form
        features = [request.form.get(name) for name in feature_names]

        # Create DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)
        print("Input DataFrame:")
        print(input_df)

        # Preprocess the input
        processed_input = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(processed_input)
        score = prediction[0]

        # Risk classification
        if score <= 0.5:
            risk_level = "🟢 Good Dental Health (Low Risk) 😄"
        elif score <= 1.5:
            risk_level = "🟡 Moderate Dental Health (Some Risk) 🙂"
        else:
            risk_level = "🔴 Poor Dental Health (High Risk) 😟"

        # Final text to display (with line break)
        result_text = f"🦷 Predicted Dental Caries Score: {score:.2f}<br>{risk_level}"

        # Send to HTML
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        print("Prediction error:")
        traceback.print_exc()
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
