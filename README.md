# 🤖 Plataforma de Análise Exploratória Inteligente

Uma aplicação web avançada para análise exploratória de dados usando inteligência artificial.

## ✨ Funcionalidades

- **🔍 Análise Exploratória Automática**: Análise inteligente de datasets com insights automáticos
- **📊 Visualizações Avançadas**: Gráficos interativos e personalizáveis
- **🤖 Análises Avançadas**: Clustering inteligente (K-Means, DBSCAN) com otimização automática
- **📈 Testes Estatísticos**: Testes de normalidade, correlação, Mann-Whitney, qui-quadrado
- **🎨 Interface Moderna**: Interface web responsiva e intuitiva

## 🛠️ Stack Tecnológico

- **Frontend**: Streamlit
- **Backend**: Python 3.11+
- **IA**: Google Gemini API + LangChain
- **Análises Avançadas**: Scikit-learn
- **Análise**: Pandas + NumPy
- **Visualização**: Matplotlib + Seaborn
- **Estatística**: SciPy

## 🚀 Deploy no Streamlit Cloud

### ☁️ Acesso Direto
**🌐 App Online**: [Link da aplicação no Streamlit Cloud]

### 🛠️ Deploy Próprio

1. **Fork este repositório no GitHub**

2. **Acesse [Streamlit Cloud](https://share.streamlit.io/)**

3. **Crie um novo app**:
   - Repository: seu-usuario/seu-repositorio  
   - Branch: main
   - Main file: app.py

4. **Configure os Secrets** (Settings → Secrets):
   ```toml
   GOOGLE_API_KEY = "sua_google_gemini_api_key_aqui"
   ```
   
   Obtenha sua API key gratuita em: https://makersuite.google.com/app/apikey

5. **Deploy automático** - Pronto! 

### 🖥️ Execução Local (Opcional)

```bash
git clone <repository-url>
pip install -r requirements.txt
# Configure GOOGLE_API_KEY no arquivo .streamlit/secrets.toml
streamlit run app.py
```

📖 **Instruções detalhadas**: [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)

## 📝 Como Usar

1. **Upload de Dados**: Faça upload de um arquivo CSV
2. **Análise IA**: Use linguagem natural para fazer perguntas sobre seus dados
3. **Visualizações**: Explore gráficos interativos e análises estatísticas
4. **Análises Avançadas**: Aplique clustering, redução de dimensionalidade e outros algoritmos de análise exploratória

## 📊 Exemplos de Comandos

```
"Analise a distribuição da coluna Amount"
"Execute clustering K-means com 4 grupos"
"Teste a normalidade dos dados de V1"
"Compare Amount entre grupos de Class"
"Detecte outliers na coluna Time"
```

## 📁 Estrutura do Projeto

```
Programa/
├── app.py                  # Aplicação principal Streamlit
├── agente/                 # Módulos do agente IA
│   ├── agente_core.py     # Core do agente
│   ├── ferramentas.py     # Ferramentas de análise
│   └── memoria.py         # Sistema de memória
├── processamento/         # Processamento de dados
│   ├── carregador_dados.py # Carregamento de arquivos
│   └── motor_analise.py   # Motor de análise
├── visualizacao/          # Geração de gráficos
│   └── gerador_graficos.py
├── temp_graficos/         # Gráficos temporários
├── requirements.txt       # Dependências Python
└── README.md             # Este arquivo
```

## ⚙️ Configuração

Crie um arquivo `.env` com:

```env
GOOGLE_API_KEY=sua_chave_api_aqui
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

© 2025 Igor Töebe Lopes Farias. Todos os direitos reservados.
