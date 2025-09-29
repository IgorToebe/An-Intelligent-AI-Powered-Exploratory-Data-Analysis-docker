#!/bin/bash

# Script para configurar Google Cloud Storage para upload de arquivos grandes
# Execute este script após fazer deploy no Cloud Run

PROJECT_ID="${1:-your-project-id}"
BUCKET_NAME="${2:-i2a2-eda-uploads}"
REGION="${3:-us-central1}"

echo "🗄️ Configurando Google Cloud Storage..."
echo "📋 Projeto: $PROJECT_ID"
echo "🪣 Bucket: $BUCKET_NAME"
echo "🌍 Região: $REGION"

# Verificar se está logado
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "❌ Faça login no Google Cloud primeiro:"
    echo "   gcloud auth login"
    exit 1
fi

# Definir projeto
echo "📝 Configurando projeto..."
gcloud config set project $PROJECT_ID

# Verificar se bucket já existe
if gsutil ls gs://$BUCKET_NAME 2>/dev/null; then
    echo "✅ Bucket $BUCKET_NAME já existe"
else
    echo "🆕 Criando bucket $BUCKET_NAME..."
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
    
    if [ $? -eq 0 ]; then
        echo "✅ Bucket criado com sucesso!"
    else
        echo "❌ Erro ao criar bucket"
        exit 1
    fi
fi

# Configurar CORS para permitir uploads diretos do browser
echo "🔧 Configurando CORS..."
cat > cors.json << EOF
[
    {
        "origin": ["https://*.streamlit.app", "https://*.run.app", "http://localhost:*"],
        "method": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "responseHeader": ["Content-Type", "Access-Control-Allow-Origin"],
        "maxAgeSeconds": 3600
    }
]
EOF

gsutil cors set cors.json gs://$BUCKET_NAME

if [ $? -eq 0 ]; then
    echo "✅ CORS configurado com sucesso!"
    rm cors.json
else
    echo "❌ Erro ao configurar CORS"
    rm cors.json
    exit 1
fi

# Configurar lifecycle para limpar uploads antigos (opcional)
echo "🧹 Configurando limpeza automática..."
cat > lifecycle.json << EOF
{
    "rule": [
        {
            "action": {"type": "Delete"},
            "condition": {
                "age": 1,
                "matchesPrefix": ["uploads/"]
            }
        }
    ]
}
EOF

gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME

if [ $? -eq 0 ]; then
    echo "✅ Lifecycle configurado (arquivos temporários serão removidos após 1 dia)"
    rm lifecycle.json
else
    echo "⚠️ Aviso: Erro ao configurar lifecycle (não crítico)"
    rm lifecycle.json
fi

# Configurar permissões para o Cloud Run
echo "🔐 Configurando permissões..."

# Obter email do serviço do Cloud Run (assumindo que já foi deployado)
SERVICE_ACCOUNT=$(gcloud run services describe i2a2-eda-platform --region=$REGION --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || echo "")

if [ -z "$SERVICE_ACCOUNT" ]; then
    echo "⚠️ Serviço Cloud Run não encontrado. Configure as permissões manualmente:"
    echo "   1. Vá para IAM & Admin no Console"
    echo "   2. Adicione a role 'Storage Object Admin' para a conta de serviço do Cloud Run"
else
    echo "📧 Conta de serviço: $SERVICE_ACCOUNT"
    
    # Dar permissões de Storage Object Admin
    gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin gs://$BUCKET_NAME
    
    if [ $? -eq 0 ]; then
        echo "✅ Permissões configuradas!"
    else
        echo "⚠️ Erro ao configurar permissões automaticamente"
        echo "Configure manualmente no Console do Google Cloud"
    fi
fi

echo ""
echo "🎉 Configuração concluída!"
echo ""
echo "📋 Próximos passos:"
echo "1. Atualize as variáveis de ambiente no Cloud Run:"
echo "   GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
echo "   GCS_BUCKET_NAME=$BUCKET_NAME"
echo ""
echo "2. Ou configure no secrets.toml:"
echo "   GOOGLE_CLOUD_PROJECT = \"$PROJECT_ID\""
echo "   GCS_BUCKET_NAME = \"$BUCKET_NAME\""
echo ""
echo "3. Redeploy sua aplicação"
echo ""
echo "🔗 Links úteis:"
echo "   Bucket: https://console.cloud.google.com/storage/browser/$BUCKET_NAME"
echo "   IAM: https://console.cloud.google.com/iam-admin/iam"