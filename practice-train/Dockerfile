# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

## Copy requirements to /app
# COPY pyproject.toml /app/
COPY requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Temp install requirements
RUN pip3 install -r requirements.txt

# Copy remaining directory contents to /app
COPY practice_train /app/practice_train/
# In future I will pull data from remote storage - data is a coupe KB so unneccessary for now
COPY data /app/data/

CMD ["python", "practice_train/train.py"]