@echo off
echo 🧪 Testando configuração do Google Cloud Storage...
echo.

REM Configurar variáveis de ambiente se não estiverem definidas
if "%GOOGLE_CLOUD_PROJECT%"=="" (
    echo ⚙️ Configurando variáveis de ambiente...
    set GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
    set GCS_BUCKET_NAME=i2a2-eda-uploads
)

echo 📋 Projeto: %GOOGLE_CLOUD_PROJECT%
echo 🪣 Bucket: %GCS_BUCKET_NAME%
echo.

REM Executar teste
python test-gcs.py

echo.
echo 📋 Se o teste falhou, execute:
echo    gcloud auth application-default login
echo    gcloud config set project %GOOGLE_CLOUD_PROJECT%

pause