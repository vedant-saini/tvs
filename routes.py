from flask import Blueprint, render_template, request, jsonify
import requests

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None  # Initialize prediction as None

    if request.method == 'POST':
        data = {
            'Gender': request.form.get('gender'),
            'Married': request.form.get('married'),
            'Dependents': request.form.get('dependents'),
            'Education': request.form.get('education'),
            'Self_Employed': request.form.get('self_employed'),
            'ApplicantIncome': request.form.get('income'),
            'CoapplicantIncome': request.form.get('coapplicant_income'),
            'LoanAmount': request.form.get('loan_amount'),
            'Loan_Amount_Term': request.form.get('loan_amount_term'),
            'Credit_History': request.form.get('credit_history'),
            'Property_Area': request.form.get('property_area')
        }           

        #data = {
            
            #'LoanAmount': request.form.get('loan_amount'),
            #'Credit_History': request.form.get('credit_history'),
            #'ApplicantIncome': request.form.get('income'),
            # Add other necessary form fields
        #}

        response = requests.post('https://tvs-credit-risk-predictor-05f9b318b3cd.herokuapp.com/predict', json=data)
        print("Response content:", response.content)  # Debug: Log raw response content
        response = requests.post('https://tvs-credit-risk-predictor-05f9b318b3cd.herokuapp.com/predict', json=data)
        print("Response content:", response.content)  # Debug: Log raw response content
        if response.status_code == 200:
            response_json = response.json()
            if response_json and 'predictions' in response_json:
                prediction = response_json['predictions'][0]  # Get the first prediction
            else:
                print("Error: Received empty or invalid response from the API")
                prediction = "Error: Invalid response from the API"
        else:
            print(f"API call failed with status code: {response.status_code} and content: {response.content}")
            prediction = f"Error: API call failed with status code: {response.status_code}"

    return render_template('prediction.html', prediction=prediction)