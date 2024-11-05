# Use a base image with Python
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
#COPY . .

# Command to run your Python script
#CMD ["python", "census.py"]
