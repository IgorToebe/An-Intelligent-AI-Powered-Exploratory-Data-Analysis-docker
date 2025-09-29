# 🐳 Docker + Google Cloud Run - Setup Completo

## ✅ Arquivos Criados

### 📦 Docker:
- `Dockerfile` - Configuração da imagem Docker
- `.dockerignore` - Arquivos a ignorar no build
- `docker-compose.yml` - Para desenvolvimento local

### ☁️ Google Cloud Run:
- `cloudrun-service.yaml` - Configuração do serviço
- `deploy-cloudrun.sh` - Script de deploy (Linux/Mac)
- `deploy-cloudrun.bat` - Script de deploy (Windows)
- `DEPLOY-CLOUDRUN.md` - Documentação completa

### 🧪 Testes:
- `test-docker.sh` - Teste local (Linux/Mac)
- `test-docker.bat` - Teste local (Windows)

### 🔐 Configuração:
- `api-key.txt.example` - Exemplo de configuração da API
- `.gitignore` - Arquivos a ignorar no Git

## 🚀 Quick Start

### 1. Preparar API Key:
```bash
# Copie o arquivo exemplo
cp api-key.txt.example api-key.txt

# Edite e adicione sua chave real
notepad api-key.txt  # Windows
nano api-key.txt     # Linux/Mac
```

### 2. Teste Local:
```bash
# Windows
test-docker.bat

# Linux/Mac
chmod +x test-docker.sh
./test-docker.sh
```

### 3. Deploy no Cloud Run:
```bash
# Edite primeiro os scripts com seu PROJECT_ID
# Windows
deploy-cloudrun.bat

# Linux/Mac
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh
```

## 💰 Estimativa de Custos (Cloud Run)

- **Free Tier**: 2M requests/mês + 360k GB-segundos/mês
- **Além do free**: ~$0.40 por 1M requests
- **Memória/CPU**: ~$0.024 por hora de uso contínuo

## 🔧 Configurações Otimizadas

- **Imagem**: Python 3.11 slim (menor tamanho)
- **Recursos**: 2GB RAM, 1 vCPU
- **Auto-scaling**: 0-10 instâncias
- **Timeout**: 5 minutos
- **Lazy loading**: IA carregada apenas quando necessário

## 📊 Features Incluídas

- ✅ Health checks automáticos
- ✅ Logs estruturados
- ✅ Auto-scaling inteligente
- ✅ SSL/HTTPS automático
- ✅ Monitoramento integrado
- ✅ Zero downtime deploys

## 🆘 Troubleshooting

### Docker build falha:
```bash
# Limpar cache
docker system prune -f
docker build --no-cache -t i2a2-eda-platform .
```

### Cloud Run timeout:
```bash
# Aumentar timeout
gcloud run services update i2a2-eda-platform --timeout=600s
```

### Erro de memória:
```bash
# Aumentar memória
gcloud run services update i2a2-eda-platform --memory=4Gi
```

## 📞 Links Úteis

- [Console Google Cloud](https://console.cloud.google.com/)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [Status Page](https://status.cloud.google.com/)

---

**Pronto para deploy! 🚀**