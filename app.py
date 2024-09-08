from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load("model.pkl")
lbl_encoders = joblib.load("lbl_encoders.pkl")  # Assuming you saved them similarly
na_vals = joblib.load("na_vals.pkl")  # Assuming you saved them similarly

# Check if loaded objects are not None
if lbl_encoders is None:
    raise ValueError("Label encoders failed to load properly.")
if na_vals is None:
    raise ValueError("NA values failed to load properly.")

@app.route('/')
def home():
    return "Loan Prediction API is up and running!"

def preprocess_data(df):
    # Handle missing values and encode categorical features
    for feature in df.columns:
        if df[feature].isna().any():
            df[feature].fillna(na_vals.get(feature, 0), inplace=True)
        # Apply label encoding only to categorical features
        if feature in lbl_encoders and not pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = lbl_encoders[feature].transform(df[feature])
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle form data or JSON
        data = request.form.to_dict() if request.form else request.json

        # Debug: Log the incoming data
        print("Received data:", data)

        # Check if the input data is valid
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Expected feature names (should match those used in the model)
        expected_features = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed", 
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]

        # Ensure incoming data contains all expected features
        missing_features = [feature for feature in expected_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert data to DataFrame with correct feature names
        df = pd.DataFrame([data])

        # Debug: Log the DataFrame after conversion
        print("DataFrame:", df)

        # Preprocess the data
        df = preprocess_data(df)

        # Debug: Log the preprocessed DataFrame
        print("Preprocessed DataFrame:", df)

        # Make predictions
        predictions = model.predict(df)
        print("Raw predictions:", predictions)  # Debug: Log raw predictions

        # Ensure predictions are valid
        if predictions is None or len(predictions) == 0:
            print("Error: No predictions returned by the model")
            return jsonify({"error": "Prediction failed, no predictions returned"}), 500
        
        # Convert predictions to 'Yes' or 'No'
        predictions = ["Yes" if pred == 1 else "No" for pred in predictions]

        # Debug: Log final predictions
        print("Final predictions:", predictions)

        # Return the predictions as JSON
        return jsonify({"predictions": predictions}), 200

    except KeyError as e:
        # Debug: Log the KeyError message
        print(f"KeyError: {str(e)}")
        return jsonify({"error": f"Missing or incorrect data field: {str(e)}"}), 400
    except Exception as e:
        # Debug: Log the exception message
        print("Error:", str(e))
        # Handle any other exceptions and return a 500 status code
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Heroku, or default to 5001 for local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)  # Bind to the correct port
