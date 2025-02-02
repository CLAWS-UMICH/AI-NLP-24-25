from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

# Initialize Flask and SocketIO
app = Flask(__name__)  # Define app in global scope
socketio = SocketIO(app)  # Attach SocketIO to the app

# Define model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")  # Adjust as needed

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    print(f'Received message: {data}')
    emit('response', {'data': 'Message received'})

# API endpoint for image processing
@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Receive image as base64 string
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform YOLO detection
        results = model(image)

        # Extract detected objects
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0]
                })

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
