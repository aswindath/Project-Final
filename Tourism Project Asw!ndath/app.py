from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load preprocessor and model
preprocessor = load("preprocessor.joblib")
model = load("knn_model.joblib")

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        features = {
            "Age": int(request.form["age"]),
            "CityTier": request.form["city_tier"],
            "Occupation": request.form["occupation"],
            "Gender": request.form["gender"],
            "NumberOfPersonVisiting": int(request.form["num_person_visiting"]),
            "PreferredPropertyStar": request.form["preferred_property_star"],
            "MaritalStatus": request.form["marital_status"],
            "NumberOfChildrenVisiting": int(request.form["num_children_visiting"]),
            "Designation": request.form["designation"],
            "MonthlyIncome": int(request.form["monthly_income"])
        }

        # Preprocess input data
        input_data = pd.DataFrame(features, index=[0])
        input_data = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction
        return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
