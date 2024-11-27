from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import pytesseract
import numpy as np
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

class ImageResponse(BaseModel):
    extracted_text: str

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for better OCR accuracy."""
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

@app.post("/extract_text", response_model=ImageResponse)
async def extract_text(image: UploadFile = File(...)):
    """Endpoint to extract text from an uploaded image."""
    
    # Check file extension
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed types: png, jpg, jpeg, bmp, tiff.")
    
    try:
        # Read the image as bytes and decode it using OpenCV
        image_bytes = await image.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image file.")
        
        # Preprocess the image for OCR
        processed_image = preprocess_image(image)

        # Perform OCR using pytesseract
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text detected in the image.")

        return {"extracted_text": extracted_text.strip()}
    
    except Exception as e:
        # Handle any unforeseen errors during image processing
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
