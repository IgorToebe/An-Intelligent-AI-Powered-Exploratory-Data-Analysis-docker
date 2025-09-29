# 🔧 SOLUÇÃO: Variáveis de Ambiente no Cloud Run

## ❌ Problema Detectado

```
☁️ Executando no Google Cloud Run - autenticação automática
❌ GOOGLE_CLOUD_PROJECT não configurado no Cloud Run
🚨 Google Cloud Storage não está configurado!
```

## ✅ Soluções

### 1. Re-deploy com Variáveis Corretas

Execute o deploy novamente para garantir que as variáveis de ambiente sejam aplicadas:

```bash
.\deploy-cloudrun.bat groovy-rope-471520-c9
```

### 2. Verificar Configuração Atual

Execute para verificar se as variáveis estão configuradas:

```bash
.\check-cloudrun-config.bat
```

### 3. Configurar Manualmente (se necessário)

Se o script automático falhar, configure manualmente:

```bash
gcloud run services update i2a2-eda-platform \
    --region us-central1 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9,GCS_BUCKET_NAME=i2a2-eda-uploads
```

### 4. Verificar Bucket GCS

Confirme que o bucket existe:

```bash
gsutil ls gs://i2a2-eda-uploads
```

Se não existir, crie:

```bash
gsutil mb -p groovy-rope-471520-c9 -c STANDARD -l us-central1 gs://i2a2-eda-uploads
```

## 🔍 Diagnóstico

1. **No Cloud Run**: Variáveis de ambiente devem ser configuradas durante o deploy
2. **Bucket GCS**: Deve existir no projeto `groovy-rope-471520-c9`
3. **Permissões**: O serviço Cloud Run precisa de acesso ao GCS

## 📋 Checklist de Verificação

- [ ] Variáveis de ambiente configuradas no Cloud Run
- [ ] Bucket `i2a2-eda-uploads` existe
- [ ] Projeto `groovy-rope-471520-c9` ativo
- [ ] APIs do GCS e Cloud Run habilitadas

## 🚀 Próximos Passos

1. Execute `.\deploy-cloudrun.bat` para re-deploy com configuração correta
2. Teste o upload de arquivo para verificar se GCS está funcionando
3. Use `.\check-cloudrun-config.bat` para verificar configuração
