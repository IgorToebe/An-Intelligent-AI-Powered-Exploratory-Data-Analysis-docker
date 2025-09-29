# 📁 Upload de Arquivos Grandes com Google Cloud Storage

## 🎯 Objetivo

Esta solução permite o upload de arquivos CSV de até **200MB** contornando o limite de 32MB do Cloud Run através do Google Cloud Storage com signed URLs.

## 🏗️ Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Cloud Run     │    │ Google Cloud    │
│   Frontend      │    │   Backend       │    │    Storage      │
│                 │    │                 │    │                 │
│ 1. Solicita URL │───▶│ 2. Gera Signed  │    │                 │
│                 │    │    URL          │    │                 │
│                 │    │                 │    │                 │
│ 4. Upload       │────┼─────────────────┼───▶│ 3. Recebe       │
│    Direto       │    │                 │    │    Arquivo      │
│                 │    │                 │    │                 │
│ 6. Solicita     │───▶│ 5. Baixa e      │◀───│                 │
│    Processamento│    │    Processa     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Como Funciona

### 1. **Solicitação de Upload**

- Usuário seleciona arquivo grande (>30MB)
- Frontend solicita signed URL ao backend

### 2. **Geração de Signed URL**

- Backend cria URL temporária (30min)
- URL permite upload direto ao GCS

### 3. **Upload Direto**

- Arquivo é enviado diretamente ao GCS
- Não passa pelo Cloud Run (bypass do limite)

### 4. **Processamento**

- Backend baixa arquivo do GCS
- Converte para DataFrame
- Remove arquivo temporário

## 📋 Configuração

### Variáveis de Ambiente

```bash
# No Cloud Run
GOOGLE_CLOUD_PROJECT=your-project-id
GCS_BUCKET_NAME=i2a2-eda-uploads

# No secrets.toml (desenvolvimento)
GOOGLE_CLOUD_PROJECT = "your-project-id"
GCS_BUCKET_NAME = "i2a2-eda-uploads"
```

### Permissões Necessárias

O Cloud Run precisa das seguintes roles:

- `Storage Object Admin` - Para criar/ler/deletar objetos
- `Storage Admin` - Para gerenciar signed URLs

## 🛠️ Setup do Bucket

### Automático

```bash
# Windows
setup-gcs.bat your-project-id

# Linux/Mac
./setup-gcs.sh your-project-id
```

### Manual

```bash
# 1. Criar bucket
gsutil mb -p your-project-id -l us-central1 gs://i2a2-eda-uploads

# 2. Configurar CORS
cat > cors.json << EOF
[{
    "origin": ["https://*.streamlit.app", "https://*.run.app", "http://localhost:*"],
    "method": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "responseHeader": ["Content-Type", "Access-Control-Allow-Origin"],
    "maxAgeSeconds": 3600
}]
EOF

gsutil cors set cors.json gs://i2a2-eda-uploads

# 3. Configurar limpeza automática (opcional)
cat > lifecycle.json << EOF
{
    "rule": [{
        "action": {"type": "Delete"},
        "condition": {
            "age": 1,
            "matchesPrefix": ["uploads/"]
        }
    }]
}
EOF

gsutil lifecycle set lifecycle.json gs://i2a2-eda-uploads
```

## 🔒 Segurança

### Signed URLs

- ✅ Tempo limitado (30 minutos)
- ✅ Acesso específico (PUT apenas)
- ✅ Tamanho limitado (200MB max)
- ✅ Tipo MIME validado

### CORS

- ✅ Domínios permitidos específicos
- ✅ Métodos limitados
- ✅ Headers controlados

### Limpeza Automática

- ✅ Arquivos removidos após 1 dia
- ✅ Apenas pasta "uploads/"
- ✅ Não afeta outros arquivos

## 💰 Custos

### Google Cloud Storage

- **Armazenamento**: $0.020 por GB/mês (Standard)
- **Operações**: $0.005 por 1K operações
- **Rede**: Grátis para mesmo projeto

### Estimativa Mensal

```
100 uploads de 150MB/dia:
- Armazenamento: ~$0.10 (temporário)
- Operações: ~$0.15
- Total: ~$0.25/mês
```

## 📊 Limites

| Recurso             | Limite     |
| ------------------- | ---------- |
| Tamanho máximo      | 200MB      |
| Tempo de upload     | 5 minutos  |
| Signed URL validade | 30 minutos |
| Retenção temporária | 1 dia      |
| Uploads simultâneos | 10         |

## 🔍 Monitoramento

### Logs Úteis

```bash
# Logs do Cloud Run
gcloud logging read "resource.type=cloud_run_revision" --limit=50

# Logs do Cloud Storage
gcloud logging read "resource.type=gcs_bucket" --limit=50

# Métricas de upload
gcloud logging read "jsonPayload.message=~'Upload.*GCS'" --limit=20
```

### Métricas no Console

- **Cloud Run**: Requests, latência, erros
- **Cloud Storage**: Operações, bandwidth, armazenamento
- **Cloud Monitoring**: Dashboards customizados

## 🚨 Troubleshooting

### Problemas Comuns

#### 1. Erro 403 - Permissões

```bash
# Verificar IAM
gcloud projects get-iam-policy your-project-id

# Adicionar permissão
gcloud projects add-iam-policy-binding your-project-id \
    --member="serviceAccount:SERVICE_ACCOUNT" \
    --role="roles/storage.objectAdmin"
```

#### 2. CORS Error

```bash
# Verificar CORS
gsutil cors get gs://i2a2-eda-uploads

# Reconfigurar
gsutil cors set cors.json gs://i2a2-eda-uploads
```

#### 3. Upload Timeout

- Verificar tamanho do arquivo
- Verificar conexão de rede
- Aumentar timeout no código

#### 4. Signed URL Expirada

- URLs válidas por apenas 30 minutos
- Regenerar se necessário
- Verificar fuso horário do servidor

## 🔄 Fallback Strategy

Se GCS não estiver disponível:

1. **Fallback automático** para upload tradicional
2. **Limite de 30MB** aplicado
3. **Aviso ao usuário** sobre limitação
4. **Funcionalidade mantida** para arquivos pequenos

## 📈 Otimizações Futuras

### Possíveis Melhorias

- **Upload resumable** para arquivos muito grandes
- **Compressão automática** antes do upload
- **Streaming processing** para economizar memória
- **Cache inteligente** para arquivos frequentes
- **Multi-part upload** para paralelização

### Alternativas

- **Firebase Storage** - Mais simples para MVPs
- **AWS S3** - Se já usando AWS
- **Azure Blob** - Se já usando Azure
- **Chunked upload** - Dividir arquivo em partes
