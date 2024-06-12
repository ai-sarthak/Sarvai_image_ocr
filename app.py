from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np
import base64

app = Flask(__name__)

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use dilation and erosion to remove small white noises and to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return processed_image

@app.route('/extract_text', methods=['POST'])
def extract_text():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    processed_image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    return jsonify({'extracted_text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
