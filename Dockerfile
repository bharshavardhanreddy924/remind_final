FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0
ENV DEBUG=False

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libboost-all-dev \
    libopencv-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dlib and other Python dependencies
RUN pip install --no-cache-dir dlib==19.24.1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and static files
RUN mkdir -p static/uploads/faces
RUN mkdir -p static/uploads/known_faces
RUN mkdir -p static/uploads/memories
RUN chmod -R 777 static/uploads

# Download NLTK resources
RUN python -m nltk.downloader punkt stopwords wordnet omw-1.4

# Expose port for Cloud Run/GCP
EXPOSE $PORT

# Run the application using JSON array for CMD
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "wsgi:app"]
