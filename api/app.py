from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Gradient Boosting Model API"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = gradient_boosting_model.predict([data])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)