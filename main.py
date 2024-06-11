from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Function to process the image
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(resized_img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded. Please choose an image.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file uploaded. Please choose an image.")

    try:
        img_path = f'static/uploads/{file.filename}'
        file.save(img_path)
        img = process_image(img_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction[0])
        predicted_letter = chr(ord('a') + predicted_class)

        return render_template('index.html', prediction=f' Braille Character: {predicted_letter}')
    except Exception as e:
        return render_template('index.html', prediction=f"An error occurred: {str(e)}")

@app.route('/clear', methods=['GET'])
def clear():
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
