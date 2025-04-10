# ReMind App Deployment Guide

This document provides instructions for deploying the ReMind application to Google Cloud Platform using Docker.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) installed on your machine
2. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured
3. A Google Cloud Platform account with billing enabled
4. A MongoDB instance (either MongoDB Atlas or self-hosted)

## Local Development

### Building and Running Locally

1. Clone the repository and navigate to the project directory:
   ```bash
   cd remind_final
   ```

2. Build the Docker image:
   ```bash
   docker build -t remind-app .
   ```

3. Run the container locally:
   ```bash
   docker run -p 8080:8080 -e MONGO_URI="your-mongodb-connection-string" remind-app
   ```

4. Access the application at `http://localhost:8080`

## Deploying to Google Cloud Run

### Manual Deployment

1. Update the configuration variables in `deploy-gcloud.sh`:
   - `PROJECT_ID`: Your Google Cloud project ID
   - `REGION`: Your preferred Google Cloud region
   - `SERVICE_NAME`: A name for your Cloud Run service
   - `MONGO_CONNECTION_STRING`: Your MongoDB connection string

2. Make the deployment script executable:
   ```bash
   chmod +x deploy-gcloud.sh
   ```

3. Run the deployment script:
   ```bash
   ./deploy-gcloud.sh
   ```

### Setting Up MongoDB

For the application to work properly, you'll need a MongoDB instance. We recommend using MongoDB Atlas:

1. Create a free account on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster
3. Set up database access (username and password)
4. Whitelist all IP addresses (0.0.0.0/0) or specific IP ranges
5. Get your connection string from the Atlas dashboard
6. Pass the connection string as an environment variable when deploying

## Environment Variables

The application uses the following environment variables:

- `MONGO_URI`: MongoDB connection string
- `PORT`: The port on which the application runs (default: 8080)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials file

## Additional Configurations

### Setting Up Custom Domain

1. In Google Cloud Console, navigate to Cloud Run
2. Select your service
3. Go to "Domain Mappings" tab
4. Click "Add Mapping" and follow the instructions

### Configuring SSL

Cloud Run automatically provides SSL certificates for custom domains through Google-managed certificates.

## Troubleshooting

- **Application crashes or doesn't start**: Check the Cloud Run logs in Google Cloud Console
- **Database connection issues**: Verify your MongoDB connection string and ensure that the IP whitelist includes Google Cloud IPs
- **Performance issues**: Consider upgrading your Cloud Run instance's CPU and memory settings 