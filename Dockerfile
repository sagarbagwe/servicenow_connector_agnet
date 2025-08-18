# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit runs on port 8080 in Cloud Run
EXPOSE 8080

# Streamlit entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
