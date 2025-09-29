@echo off
REM Script para verificar configuração do Cloud Run

set PROJECT_ID=groovy-rope-471520-c9
set SERVICE_NAME=ai-powered-exploratory-data-analysis
set REGION=southamerica-east1

echo 🔍 Verificando configuração do serviço Cloud Run...

echo.
echo 📋 Configuração do serviço:
gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format="yaml(spec.template.spec.template.spec.containers[0].env)"

echo.
echo 🌐 URL do serviço:
gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format="value(status.url)"

echo.
echo 🪣 Status do bucket GCS:
gsutil ls gs://i2a2-eda-uploads >nul 2>&1 && echo "✅ Bucket existe" || echo "❌ Bucket não encontrado"

echo.
echo ✅ Verificação concluída!
pause