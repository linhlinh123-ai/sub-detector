# Dockerfile
FROM python:3.10-slim

# system deps for opencv-headless and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libjpeg-dev libpng-dev \
    ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy and install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy application
COPY detect_sub_region_light.py /app/detect_sub_region_light.py
COPY server.py /app/server.py

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
