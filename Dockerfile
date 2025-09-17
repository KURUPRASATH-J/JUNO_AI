FROM python:3.9-slim

WORKDIR /code

# Install system dependencies including tesseract for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /code/.cache/huggingface && \
    mkdir -p /code/.cache/torch && \
    chmod -R 777 /code/.cache

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/code/.cache/huggingface
ENV HF_HOME=/code/.cache/huggingface
ENV TORCH_HOME=/code/.cache/torch
ENV HF_HUB_CACHE=/code/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/code/.cache/sentence_transformers

# Copy requirements and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY . /code

# Create necessary directories
RUN mkdir -p /code/static /code/templates

# Expose port
EXPOSE 7860

# Set environment variable for Hugging Face Spaces
ENV PYTHONPATH=/code

# Command to run the application
CMD ["python", "app.py"]
