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
    return render_template('index.html')  # Make sure index.html is in the 'templates/' folder

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Request method:", request.method)
    try:
        # Define the correct feature names (21 fields from training)
        feature_names = [
            "age", "height(cm)", "weight(kg)", "waist(cm)", "eyesight(left)", "eyesight(right)",
            "hearing(left)", "hearing(right)", "systolic", "relaxation", "fasting blood sugar",
            "hemoglobin", "serum creatinine", "ast", "alt", "gtp", "ldl", "hdl",
            "triglyceride", "urine protein", "gender"
        ]
        
        # Get form data
        features = [request.form.get(name) for name in feature_names]

        # Create DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)
        print("Input DataFrame:")
        print(input_df)

        # Preprocess input
        processed_input = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(processed_input)

        return render_template('index.html',
                               prediction_text=f"Predicted Dental Caries Score: {prediction[0]:.2f}")

    except Exception as e:
        print("Prediction error:")
        traceback.print_exc()
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
