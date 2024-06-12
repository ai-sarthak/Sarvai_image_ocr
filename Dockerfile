FROM python:3.9-slim

# Install Tesseract OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    apt-get clean

# Set up the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the Flask app
COPY . .

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
