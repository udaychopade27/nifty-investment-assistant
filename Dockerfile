FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Kolkata
# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with reload (IMPORTANT)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

