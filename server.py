from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
from gevent.pywsgi import WSGIServer
import logging

app = Flask(__name__)

current_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_path, 'XGBReg_model.pkl')

try:
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    logging.error(f"Error: 'XGBReg_model.pkl' file not found in the directory: {model_path}")
    exit()

X_train = joblib.load('X_train.pkl')
scaler = joblib.load('scaler.pkl')

# Обучение StandardScaler на обучающих данных
scaler.fit(X_train)

@app.errorhandler(500)
def internal_server_error(e):
    logging.error(f'Internal Server Error: {e}')
    return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided in the request'}), 400

    try:
        data_x = np.array(data).reshape(1, -1)
        scaled_data = scaler.transform(data_x)
        prediction = model.predict(scaled_data).astype(float)
        predicted_original = np.exp(prediction)  # Применяем обратное преобразование логарифма
        return jsonify({'prediction': predicted_original[0]})
    except ValueError as ve:
        logging.error(f'ValueError during prediction: {ve}')
        return jsonify({'error': 'Invalid data format in the request'}), 400
    except Exception as e:
        logging.error(f'Prediction error: {e}')
        return jsonify({'error': 'Prediction error occurred'}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    http_server = WSGIServer(('', 5000), app)
    print("Flask server is running...")
    http_server.serve_forever()