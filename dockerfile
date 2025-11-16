# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency files (requirements.txt and runtime.txt) into the container
COPY requirements.txt .
COPY runtime.txt .

# Install any required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port your application listens on 
EXPOSE 8000 

# Define the command to run your application (the Startup command)
# The 'app:app' part assumes your Flask/Django/FastAPI instance is named 'app' in 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]