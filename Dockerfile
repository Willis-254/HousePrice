FROM python:3.8-slim-bullseye

WORKDIR /app

# Copy requirements file first for better layer caching
COPY ./requirements.txt .

# Install AWS CLI and Python dependencies
RUN apt-get update && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt