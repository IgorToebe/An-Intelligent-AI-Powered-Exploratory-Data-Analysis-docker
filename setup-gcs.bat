@echo off
REM Script para configurar Google Cloud Storage para upload de arquivos grandes
REM Execute este script após fazer deploy no Cloud Run

set PROJECT_ID=%1
set BUCKET_NAME=%2
set REGION=%3

if "%PROJECT_ID%"=="" set PROJECT_ID=your-project-id
if "%BUCKET_NAME%"=="" set BUCKET_NAME=i2a2-eda-uploads
if "%REGION%"=="" set REGION=us-central1

echo 🗄️ Configurando Google Cloud Storage...
echo 📋 Projeto: %PROJECT_ID%
echo 🪣 Bucket: %BUCKET_NAME%
echo 🌍 Região: %REGION%

REM Verificar se está logado
gcloud auth list --filter=status:ACTIVE --format="value(account)" | findstr "@" >nul
if errorlevel 1 (
    echo ❌ Faça login no Google Cloud primeiro:
    echo    gcloud auth login
    exit /b 1
)

REM Definir projeto
echo 📝 Configurando projeto...
gcloud config set project %PROJECT_ID%

REM Verificar se bucket já existe
gsutil ls gs://%BUCKET_NAME% >nul 2>&1
if errorlevel 1 (
    echo 🆕 Criando bucket %BUCKET_NAME%...
    gsutil mb -p %PROJECT_ID% -c STANDARD -l %REGION% gs://%BUCKET_NAME%
    
    if errorlevel 1 (
        echo ❌ Erro ao criar bucket
        exit /b 1
    ) else (
        echo ✅ Bucket criado com sucesso!
    )
) else (
    echo ✅ Bucket %BUCKET_NAME% já existe
)

REM Configurar CORS
echo 🔧 Configurando CORS...
(
echo [
echo     {
echo         "origin": ["https://*.streamlit.app", "https://*.run.app", "http://localhost:*"],
echo         "method": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
echo         "responseHeader": ["Content-Type", "Access-Control-Allow-Origin"],
echo         "maxAgeSeconds": 3600
echo     }
echo ]
) > cors.json

gsutil cors set cors.json gs://%BUCKET_NAME%

if errorlevel 1 (
    echo ❌ Erro ao configurar CORS
    del cors.json
    exit /b 1
) else (
    echo ✅ CORS configurado com sucesso!
    del cors.json
)

REM Configurar lifecycle
echo 🧹 Configurando limpeza automática...
(
echo {
echo     "rule": [
echo         {
echo             "action": {"type": "Delete"},
echo             "condition": {
echo                 "age": 1,
echo                 "matchesPrefix": ["uploads/"]
echo             }
echo         }
echo     ]
echo }
) > lifecycle.json

gsutil lifecycle set lifecycle.json gs://%BUCKET_NAME%

if errorlevel 1 (
    echo ⚠️ Aviso: Erro ao configurar lifecycle (não crítico)
) else (
    echo ✅ Lifecycle configurado (arquivos temporários serão removidos após 1 dia)
)
del lifecycle.json

echo.
echo 🎉 Configuração concluída!
echo.
echo 📋 Próximos passos:
echo 1. Atualize as variáveis de ambiente no Cloud Run:
echo    GOOGLE_CLOUD_PROJECT=%PROJECT_ID%
echo    GCS_BUCKET_NAME=%BUCKET_NAME%
echo.
echo 2. Ou configure no secrets.toml:
echo    GOOGLE_CLOUD_PROJECT = "%PROJECT_ID%"
echo    GCS_BUCKET_NAME = "%BUCKET_NAME%"
echo.
echo 3. Redeploy sua aplicação
echo.
echo 🔗 Links úteis:
echo    Bucket: https://console.cloud.google.com/storage/browser/%BUCKET_NAME%
echo    IAM: https://console.cloud.google.com/iam-admin/iam

pause