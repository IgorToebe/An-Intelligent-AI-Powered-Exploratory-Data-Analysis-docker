@echo off
REM Script para verificar configuração do Cloud Run

set PROJECT_ID=groovy-rope-471520-c9
set SERVICE_NAME=i2a2-eda-platform
set REGION=us-central1

echo 🔍 Verificando configuração do serviço Cloud Run...

echo.
echo 📋 Variáveis de ambiente configuradas:
gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format="value(spec.template.spec.template.spec.containers[0].env[].name,spec.template.spec.template.spec.containers[0].env[].value)"

echo.
echo 🌐 URL do serviço:
gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format="value(status.url)"

echo.
echo 🪣 Verificando bucket GCS:
gsutil ls gs://i2a2-eda-uploads

echo.
echo ✅ Verificação concluída!
pause