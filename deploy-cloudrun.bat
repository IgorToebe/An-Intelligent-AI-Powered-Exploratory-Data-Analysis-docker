@echo off
REM Google Cloud Run Deployment Script for Windows
REM Configure these variables before running

set PROJECT_ID=groovy-rope-471520-c9
set SERVICE_NAME=i2a2-eda-platform
set REGION=us-central1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo 🚀 Starting deployment to Google Cloud Run...

REM Check if gcloud is authenticated
gcloud auth list --filter=status:ACTIVE --format="value(account)" | findstr "@" >nul
if errorlevel 1 (
    echo ❌ Please authenticate with Google Cloud first:
    echo    gcloud auth login
    exit /b 1
)

REM Set the project
echo 📝 Setting project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo 🔧 Enabling required APIs...
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com

REM Create GCS bucket if it doesn't exist
echo 🪣 Checking/creating GCS bucket...
gsutil ls gs://i2a2-eda-uploads >nul 2>&1
if errorlevel 1 (
    echo 📦 Creating GCS bucket i2a2-eda-uploads...
    gsutil mb -p %PROJECT_ID% -c STANDARD -l %REGION% gs://i2a2-eda-uploads
    echo ✅ Bucket created successfully!
) else (
    echo ✅ Bucket i2a2-eda-uploads already exists
)

REM Build and push the image
echo 🔨 Building Docker image...
docker build -t %IMAGE_NAME% .

echo 📤 Pushing image to Google Container Registry...
docker push %IMAGE_NAME%

REM Deploy to Cloud Run
echo 🚀 Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE_NAME% ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 2Gi ^
    --cpu 1 ^
    --timeout 300s ^
    --concurrency 80 ^
    --max-instances 10 ^
    --min-instances 0 ^
    --port 8080 ^
    --set-env-vars PORT=8080,GOOGLE_CLOUD_PROJECT=%PROJECT_ID%,GCS_BUCKET_NAME=i2a2-eda-uploads

echo ✅ Deployment completed!
echo 🌐 Your app should be available at:
gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"

echo.
echo 📋 Next steps:
echo 1. Set up your Google API key as a secret:
echo    gcloud secrets create google-api-key --data-file=api-key.txt
echo 2. Update the service to use the secret:
echo    gcloud run services update %SERVICE_NAME% --update-secrets=GOOGLE_API_KEY=google-api-key:latest --region=%REGION%

pause