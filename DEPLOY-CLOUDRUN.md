# 🐳 Deploy no Google Cloud Run

Este guia te ajudará a fazer deploy da Plataforma de Análise Exploratória Inteligente no Google Cloud Run usando Docker.

## 📋 Pré-requisitos

1. **Google Cloud Account** com billing habilitado
2. **Docker** instalado localmente
3. **Google Cloud SDK (gcloud)** instalado
4. **Chave API do Google Gemini**

## 🚀 Passos para Deploy

### 1. Configurar Google Cloud

```bash
# Instalar Google Cloud SDK (se não tiver)
# https://cloud.google.com/sdk/docs/install

# Autenticar
gcloud auth login

# Criar projeto (opcional)
gcloud projects create your-project-id --name="I2A2 EDA Platform"

# Configurar projeto
gcloud config set project your-project-id

# Habilitar APIs necessárias
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Configurar Variáveis de Ambiente

Edite o arquivo `deploy-cloudrun.bat` (Windows) ou `deploy-cloudrun.sh` (Linux/Mac) e configure:

```bash
PROJECT_ID="your-project-id"        # Seu ID do projeto
SERVICE_NAME="i2a2-eda-platform"   # Nome do serviço
REGION="us-central1"                # Região (ou southamerica-east1 para Brasil)
```

### 3. Configurar API Key

Crie um arquivo `api-key.txt` com sua chave do Google Gemini:

```bash
echo "sua-google-api-key-aqui" > api-key.txt
```

### 4. Deploy Automático

**Windows:**

```cmd
deploy-cloudrun.bat
```

**Linux/Mac:**

```bash
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh
```

### 5. Configurar Google Cloud Storage (para arquivos grandes)

**Automático:**

```bash
# Windows
setup-gcs.bat your-project-id

# Linux/Mac
chmod +x setup-gcs.sh
./setup-gcs.sh your-project-id
```

**Manual:**

```bash
# Criar bucket
gsutil mb -p your-project-id -l us-central1 gs://i2a2-eda-uploads

# Configurar CORS
echo '[{"origin":["https://*.streamlit.app","https://*.run.app"],"method":["PUT","POST"],"maxAgeSeconds":3600}]' > cors.json
gsutil cors set cors.json gs://i2a2-eda-uploads
```

### 5. Deploy Manual (alternativo)

```bash
# 1. Habilitar APIs
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com

# 2. Build da imagem
docker build -t gcr.io/your-project-id/i2a2-eda-platform .

# 3. Push para registry
docker push gcr.io/your-project-id/i2a2-eda-platform

# 4. Deploy no Cloud Run
gcloud run deploy i2a2-eda-platform \
    --image gcr.io/your-project-id/i2a2-eda-platform \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300s \
    --port 8080

# 5. Configurar secret da API
gcloud secrets create google-api-key --data-file=api-key.txt
gcloud run services update i2a2-eda-platform \
    --update-secrets=GOOGLE_API_KEY=google-api-key:latest \
    --region=us-central1
```

## 🔧 Configurações do Cloud Run

### Recursos Alocados:

- **CPU**: 1 vCPU
- **Memória**: 2 GB
- **Timeout**: 5 minutos
- **Concorrência**: 80 requests simultâneos
- **Instâncias**: 0-10 (auto-scaling)

### Estimativa de Custos:

- **Requests**: $0.40 por 1M requests
- **CPU**: $0.00002400 por vCPU-segundo
- **Memória**: $0.00000250 por GB-segundo
- **Free tier**: 2M requests/mês + 360k GB-segundos/mês

## 🌐 Acesso à Aplicação

Após o deploy, você receberá uma URL similar a:

```
https://i2a2-eda-platform-[hash]-uc.a.run.app
```

## 🔒 Segurança

### Configurar Domínio Personalizado (Opcional):

```bash
gcloud run domain-mappings create \
    --service i2a2-eda-platform \
    --domain your-domain.com \
    --region us-central1
```

### Configurar Autenticação (Opcional):

```bash
gcloud run services remove-iam-policy-binding i2a2-eda-platform \
    --member="allUsers" \
    --role="roles/run.invoker" \
    --region=us-central1
```

## 📊 Monitoramento

### Visualizar Logs:

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=i2a2-eda-platform" --limit 50 --format json
```

### Métricas no Console:

- Acesse: https://console.cloud.google.com/run
- Selecione seu serviço
- Vá na aba "Métricas"

## 🛠️ Troubleshooting

### Problemas Comuns:

1. **Erro de autenticação**:

   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

2. **Quota excedida**:

   ```bash
   gcloud compute regions list
   # Escolha região com menos uso
   ```

3. **Timeout no deploy**:

   ```bash
   # Aumentar timeout
   gcloud run services update i2a2-eda-platform --timeout=600s
   ```

4. **Erro de memória**:
   ```bash
   # Aumentar memória
   gcloud run services update i2a2-eda-platform --memory=4Gi
   ```

## 🔄 Atualizações

Para atualizar a aplicação:

```bash
# Rebuild e push
docker build -t gcr.io/your-project-id/i2a2-eda-platform .
docker push gcr.io/your-project-id/i2a2-eda-platform

# Deploy nova versão
gcloud run deploy i2a2-eda-platform \
    --image gcr.io/your-project-id/i2a2-eda-platform \
    --region us-central1
```

## 📞 Suporte

Para mais informações:

- [Documentação Cloud Run](https://cloud.google.com/run/docs)
- [Preços Cloud Run](https://cloud.google.com/run/pricing)
- [Limites e Quotas](https://cloud.google.com/run/quotas)
