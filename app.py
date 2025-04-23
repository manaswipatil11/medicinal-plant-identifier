from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from labels import labels

app = Flask(__name__)
model = load_model('model/model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # adjust if your model input is different
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    return render_template('index.html', label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
