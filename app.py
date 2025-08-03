from flask import Flask, render_template, request
from PIL import Image
import pytesseract
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_preprocess(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (better than Otsu for uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 15
    )

    return Image.fromarray(thresh)

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_text = ''
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filepath = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(filepath)

            # Preprocess and OCR
            processed_img = clean_preprocess(filepath)
            extracted_text = pytesseract.image_to_string(
                processed_img,
                config="--oem 3 --psm 6"
            )

            os.remove(filepath)

    return render_template('index.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)
