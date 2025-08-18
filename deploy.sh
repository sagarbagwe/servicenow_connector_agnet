#!/bin/bash
set -e

# --- Configuration ---
PROJECT_ID="sadproject2025"
REGION="us-central1"
SERVICE_NAME="servicenow-ai-agent"
SERVICE_ACCOUNT_EMAIL="servicenow-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"
CONNECTION_NAME="sn-connector-prod"
GEMINI_MODEL="gemini-2.5-flash"

echo "🚀 Starting deployment of ServiceNow AI Agent..."

# --- 1. Setup ---
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com appintegration.googleapis.com connectors.googleapis.com

echo "🔑 IMPORTANT: This script assumes the service account (${SERVICE_ACCOUNT_EMAIL}) has the following IAM roles: Vertex AI User, Application Integration Invoker, Connector Invoker, Connectors Viewer."

# --- 2. Build & Deploy ---
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}"
echo "🏗️  Building container image..."
gcloud builds submit --tag $IMAGE_PATH

echo "🚀 Deploying to Cloud Run service: $SERVICE_NAME..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_PATH \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --service-account $SERVICE_ACCOUNT_EMAIL \
    --port 8080 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},CONNECTION_NAME=${CONNECTION_NAME},GEMINI_MODEL=${GEMINI_MODEL}"

# --- 3. Finalize ---
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "🎉 Deployment Complete! Your agent is at: $SERVICE_URL"