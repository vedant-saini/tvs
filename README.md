**Loan Risk Prediction System**

**Overview**

The Loan Risk Prediction System is a web-based application designed to predict the risk associated with loan approvals. 
By leveraging a machine learning model, the system provides real-time predictions based on applicant data, helping financial institutions 
automate and streamline the loan approval process.

**Features**

Real-Time Predictions: Utilize a machine learning model to predict loan approval outcomes instantly.

User-Friendly Interface: Built with HTML, CSS, and Flask templates for a seamless user experience.

Scalability: Deployed on Heroku for easy access and scalability.

Technologies Used

Backend: Python, Flask

Frontend: HTML, CSS

Machine Learning: scikit-learn, joblib

Deployment: Heroku

Project Structure

Loan-Risk-Prediction-System/

│

├── app.py                   # Main Flask application

├── routes.py                # Application routes and logic

├── model.pkl                # Pre-trained machine learning model

├── lbl_encoders.pkl         # Label encoders for categorical data

├── templates/

│   ├── base.html            # Base HTML template

│   ├── home.html            # Home page template

│   └── prediction.html      # Loan prediction page template

├── static/

│   └── styles.css           # CSS for styling the application

├── Training Dataset.csv     # Training dataset for the model

├── Test Dataset.csv         # Test dataset for evaluation

├── README.md                # Project documentation (this file)

└── run.py                   # Script to run the Flask app

Installation and Setup

Clone the repository:


git clone https://github.com/yourusername/loan-risk-prediction.git

cd loan-risk-prediction

Install dependencies:
pip install -r requirements.txt

Run the application:
python run.py

Access the application:
Open your browser and navigate to http://localhost:5000.

How It Works

User Input: The user inputs applicant details via the web interface.

Prediction: The input data is processed and fed into the machine learning model to predict the loan approval outcome.

Output: The system displays whether the loan is approved or not, along with the associated risk level.
Deployment

The application is deployed on Heroku for easy access and scalability. You can view the live application here.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any questions or issues, please contact Vedant Saini at vedantsaini21@gmail.com.

