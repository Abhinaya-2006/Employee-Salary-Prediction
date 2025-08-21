import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model, label encoder, and column names
model = joblib.load('salary_model.pkl')
le = joblib.load('label_encoder.pkl')
model_columns = joblib.load('model_columns.pkl')

# Single route for the entire application
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        data = request.form.to_dict()
        
        # Prepare data for prediction
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data)
        
        # Ensure all columns are in the correct order as in training data
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        
        # Make a prediction
        prediction_encoded = model.predict(input_data)
        prediction_text = le.inverse_transform(prediction_encoded)
        
        return jsonify({'prediction': str(prediction_text[0])})

if __name__ == '__main__':
    app.run(debug=True)