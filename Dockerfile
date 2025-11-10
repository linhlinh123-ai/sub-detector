# Dockerfile
FROM python:3.10-slim

# install system deps including tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
