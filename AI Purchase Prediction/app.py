from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model & transformer
model = joblib.load("purchase_prediction_model.pkl")
transformer = joblib.load("transformer.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Serve the web form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        gender = request.form.get("gender")
        age = request.form.get("age")
        salary = request.form.get("salary")

        # Validate inputs
        if not gender or not age or not salary:
            return jsonify({"error": "Please provide all input values"}), 400

        # Convert inputs to DataFrame
        input_data = pd.DataFrame([[gender, int(age), int(salary)]], columns=["Gender", "Age", "EstimatedSalary"])

        # Apply transformations
        transformed_data = transformer.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)[0]
        result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

        return jsonify({"prediction": int(prediction), "message": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
