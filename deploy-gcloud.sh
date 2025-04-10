#!/bin/bash

# Configuration
PROJECT_ID="your-gcp-project-id"  # Replace with your GCP project ID
REGION="us-central1"              # Replace with your preferred region
SERVICE_NAME="remind-app"         # The service name in Cloud Run
MONGO_CONNECTION_STRING="your-mongodb-connection-string"  # Replace with your MongoDB connection string

# Build the Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME .

# Push the image to Google Container Registry
echo "Pushing image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "MONGO_URI=$MONGO_CONNECTION_STRING,GOOGLE_APPLICATION_CREDENTIALS=/secrets/cloud-key.json"

echo "Deployment completed!"
echo "Your app should be available at: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')" 