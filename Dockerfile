# Use a base image with Python 3.10 (or 3.11)
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ \
    libatlas-base-dev libhdf5-dev libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install the required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

