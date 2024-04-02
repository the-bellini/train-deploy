# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set HuggingFace access token in environment from passing an argument
ARG access_token
ENV access_token=$access_token

# Set working directory in container
WORKDIR /app

# Copy directory contents to /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-
poetry/poetry/master/getr-poetry.py | python -

# Use poetry to install dependencies
RUN /root/.poetry/bin/poetry config virtualenvs.create false \
    && /root/.poetry/bin/poetry install --no-interaction --no-ansi

CMD ["python", "practice_train/train.py"]