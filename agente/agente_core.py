"""
🤖 Agente de Análise Exploratória de Dados (EDA)

Este módulo implementa um agente inteligente que interpreta perguntas em linguagem
natural sobre dados e executa análises estatísticas, visualizações e análises avançadas.

Principais funcionalidades:
- Interpretação de linguagem natural para análise de dados
- Execução automatizada de análises estatísticas
- Geração de visualizações interativas
 - Aplicação de algoritmos para análises avançadas (clustering, PCA, regressão)
- Memória conversacional para contexto

Autor: Sistema de Análise Exploratória
Versão: 2.0
"""

import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import pandas as pd
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.tools import Tool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# Módulos locais
from agente.ferramentas import criar_ferramentas_agente
from agente.memoria import MemoriaConversa
from config_manager import get_config, get_api_key

# Flag de disponibilidade
LANGCHAIN_AVAILABLE = True


class AgenteEDA:
    """
    Agente principal para Análise Exploratória de Dados.
    Utiliza LangChain para interpretar perguntas e executar análises.
    """
    
    def __init__(self, dataframe: pd.DataFrame, api_key: Optional[str] = None, llm_model: Optional[str] = None):
        """
        Inicializa o agente EDA.
        
        Args:
            dataframe: DataFrame para análise
            api_key: Chave da API Google Gemini (opcional, pode vir do ambiente)
            llm_model: Nome do modelo LLM a ser utilizado (opcional)
        """
        self.df = dataframe
        self.memoria = MemoriaConversa()
        self.llm_model = llm_model
        
        # Configurar API key do Google Gemini
        # Sempre atualiza a variável de ambiente se uma nova API key é fornecida
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
        
        # Verificar se temos uma API key disponível (prioridade: parâmetro -> secrets -> .env)
        current_api_key = api_key or get_api_key()
        if not current_api_key:
            raise ValueError("ERRO DE AUTENTICAÇÃO: Chave API do Google Gemini não encontrada.\n\n"
                           "CONFIGURAÇÃO NECESSÁRIA:\n"
                           "1. Obtenha uma chave API gratuita: https://makersuite.google.com/app/apikey\n"
                           "2. Insira a chave no campo 'Chave da API do Motor IA' (barra lateral)\n\n"
                           "NOTA: A API do Google Gemini oferece nível gratuito para uso básico.\n\n"
                           "SUPORTE TÉCNICO: Contate o desenvolvedor Igor Töebe para assistência.")
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("ERRO DE DEPENDÊNCIA: Framework LangChain não disponível.\n\n"
                            "INSTALAÇÃO NECESSÁRIA:\n"
                            "pip install langchain langchain-google-genai\n\n"
                            "SUPORTE TÉCNICO: Contate o desenvolvedor Igor Töebe para assistência.")
        
        self.llm_disponivel = True
        print("STATUS DO SISTEMA: Inicialização do Agente IA em progresso...")
        
        # Inicializar componentes do agente
        self._inicializar_agente_langchain()
    
    def _inicializar_agente_langchain(self):
        """Inicializa o agente usando LangChain com Google Gemini ou outro modelo."""
        try:
            api_key_str = get_api_key()
            # Prioriza modelo passado pelo construtor
            model_name = self.llm_model or get_config('GEMINI_MODEL', 'gemini-2.5-flash')
            temperature = float(get_config('GEMINI_TEMPERATURE', '0.1'))
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key_str) if api_key_str else None,
                convert_system_message_to_human=True
            )
            
            # Criar ferramentas
            self.ferramentas = criar_ferramentas_agente(self.df)
            
            # Criar prompt do agente
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_prompt_sistema()),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Criar agente (usando create_tool_calling_agent que funciona com Gemini)
            agent = create_tool_calling_agent(
                llm=self.llm,
                tools=self.ferramentas,
                prompt=prompt
            )
            
            # Criar executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.ferramentas,
                memory=self.memoria.get_langchain_memory(),
                verbose=False,  # Desativado para evitar exposição de código interno
                max_iterations=20,  # Aumentado para permitir análises mais complexas
                early_stopping_method="generate",  # Permite parada antecipada quando o objetivo é alcançado
                return_intermediate_steps=True,  # IMPORTANTE: precisamos dos steps para capturar gráficos
                handle_parsing_errors=True       # Torna o agente mais resiliente a pequenas falhas de parsing
            )
            
            print("STATUS DO SISTEMA: Agente IA inicializado com sucesso e pronto para análise.")
            
        except Exception as e:
            print(f"ERRO DE INICIALIZAÇÃO: {e}")
            # Verificar se é um erro de API key inválida
            erro_str = str(e).lower()
            if "api_key_invalid" in erro_str or "api key not valid" in erro_str:
                raise ValueError("ERRO DE AUTENTICAÇÃO: Chave API do Google Gemini inválida.\n\n"
                               "LISTA DE VERIFICAÇÃO:\n"
                               "1. Certifique-se de que a chave foi copiada corretamente (sem espaços extras)\n"
                               "2. Verifique se a chave está ativa: https://makersuite.google.com/app/apikey\n"
                               "3. Confirme se há cota/créditos da API disponíveis\n\n"
                               "RECOMENDAÇÃO: Teste a chave em https://makersuite.google.com primeiro.\n\n"
                               "SUPORTE TÉCNICO: Contate o desenvolvedor Igor Töebe para assistência.")
            else:
                raise RuntimeError(f"ERRO DO SISTEMA: Falha ao inicializar o motor IA: {e}\n\n"
                                 "Por favor, verifique a conexão com a internet e tente novamente.\n\n"
                                 "SUPORTE TÉCNICO: Contate o desenvolvedor Igor Töebe se o problema persistir.")
    
    def _get_prompt_sistema(self) -> str:
        """Retorna um prompt de sistema conciso e focado em ferramentas."""
        colunas_numericas = self.df.select_dtypes(include=['number']).columns.tolist()
        colunas_categoricas = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        todas_colunas = ", ".join(list(self.df.columns))
        cols_num = ", ".join(colunas_numericas) if colunas_numericas else "Nenhuma"
        cols_cat = ", ".join(colunas_categoricas) if colunas_categoricas else "Nenhuma"

        return f"""
Você é um analista de dados IA avançado com capacidades profissionais de análise estatística e visualização.

PERFIL DO DATASET:
- Registros: {len(self.df):,} | Características: {len(self.df.columns)}
- Características Disponíveis: {todas_colunas}
- Características Numéricas: {cols_num}
- Características Categóricas: {cols_cat}

FRAMEWORK OPERACIONAL:
1) Execute ferramentas diretamente - nunca sugira código ou procedimentos manuais
2) APENAS crie visualizações quando EXPLICITAMENTE solicitado pelo usuário
3) Use exatamente os nomes das características do perfil do dataset acima
4) Forneça interpretações estatísticas profissionais e concisas
5) Foque em insights acionáveis e observações de qualidade dos dados

SUAS CAPACIDADES (use linguagem natural, não nomes de funções):
- Análise de distribuição: "Criando histograma para mostrar a distribuição dos dados"
- Detecção de outliers: "Detectando valores atípicos e criando boxplots para visualizá-los"
- Análise de correlação: "Calculando correlações e criando gráficos de dispersão ou mapas de calor"
- Análise de frequência: "Analisando a frequência de valores únicos"
- Testes estatísticos: "Realizando testes de normalidade, testes t, correlação, qui-quadrado e ANOVA"
- Análise comparativa: "Comparando grupos e criando visualizações segmentadas"
- Análises avançadas: "Realizando clustering, PCA, regressão e classificação com resultados detalhados"
- Visão geral: "Descrevendo o dataset e listando suas características"

IMPORTANTE PARA ANÁLISES AVANÇADAS:
- SEMPRE analise e apresente TODOS os resultados numéricos das análises avançadas
- Para clustering: mostre número de clusters, silhouette score, distribuição e características
- Para PCA: mostre variância explicada, componentes principais e métricas de redução
- Para regressão/classificação: mostre métricas de performance, coeficientes e interpretações
- NUNCA omita dados quantitativos dos resultados

COMUNICAÇÃO NATURAL:
- EXECUTE IMEDIATAMENTE as ações solicitadas - NÃO prometa que "vai fazer"
- Use frases como: "Calculando...", "Analisando...", "Criando gráfico de..."
- NUNCA mencione nomes de funções como "gerar_histograma_coluna()" ou "calcular_matriz_correlacao()"
- Fale como um analista profissional, não como um programador
- FAÇA AGORA, não descreva o que vai fazer

LINGUAGEM PROIBIDA (nunca use):
❌ "Vou usar a função gerar_grafico_dispersao()"
❌ "Posso executar calcular_matriz_correlacao()"
❌ "Vamos chamar detectar_outliers_coluna()"

LINGUAGEM CORRETA (sempre use):
✅ "Criando gráfico de dispersão..."
✅ "Calculando matriz de correlação..."
✅ "Detectando valores atípicos..."

CAPACIDADES ESTATÍSTICAS:
- Testes de normalidade (Shapiro-Wilk, D'Agostino-Pearson, Jarque-Bera, Anderson-Darling)
- Testes de hipóteses (testes t, testes de correlação, qui-quadrado, ANOVA)
- Testes não-paramétricos (Mann-Whitney U)
- Estatísticas descritivas e análise de distribuição

PROTOCOLO DE RESPOSTA:
- Converse com o usuário de forma profissional e amigável
- Após a primeira interação descreva suas capacidades e realize uma pré análise rápida para entender o dataset e contexto
-- SEMPRE que solicitado, execute as ferramentas solicitadas - você tem TODAS as capacidades necessárias
-- Para solicitações EXPLÍCITAS de gráficos ("gere histograma", "crie gráfico", etc.), USE IMEDIATAMENTE a ferramenta apropriada
-- NUNCA gere gráficos automaticamente sem solicitação explícita do usuário
-- Para análises de clustering, SEMPRE apresente TODOS os dados e resultados obtidos
- Comunique-se como um ANALISTA PROFISSIONAL, não como um programador
- JAMAIS diga "Vou fazer", "Vou analisar", "Vou realizar" - EXECUTE IMEDIATAMENTE quando solicitado!
- Use linguagem de AÇÃO PRESENTE: "Calculando matriz de correlação..." ao invés de "Vou calcular..."
- NUNCA inclua código Python, print(), ou comandos técnicos na resposta
- NUNCA mostre chamadas de função como "default_api.realizar_clustering()"
- APRESENTE COMPLETAMENTE todos os resultados numéricos, métricas e interpretações de ML
- Para clustering: sempre mostre silhouette score, número de clusters, distribuição e características dos grupos
- Para PCA: sempre mostre variância explicada por componente, redução de dimensionalidade e loadings
- Para modelos supervisionados: sempre mostre métricas de performance e coeficientes/importância
- Seja um bom comunicador, sempre escute atentamente o usuário e responda de forma clara e amigável
- Forneça interpretação estatística profissional (3-6 frases)
- Destaque achados significativos, padrões ou questões de qualidade dos dados
- Recomende análises de acompanhamento quando apropriado
- Nunca referencie caminhos de arquivos, nomes de funções ou detalhes de implementação técnica

-- AÇÕES IMEDIATAS OBRIGATÓRIAS:
-- Quando solicitado clustering: EXECUTE imediatamente, apresente os resultados completos
-- Quando solicitada análise: FAÇA a análise agora, não prometa que fará
-- Quando solicitado gráfico: CRIE o gráfico imediatamente, não explique que pode criar
-- PROIBIDO: Frases como "Aguarde um momento...", "Vou realizar...", "Após a análise..."
-- OBRIGATÓRIO: Execute, analise e apresente resultados de uma só vez
"""
    
    def processar_pergunta(self, pergunta: str) -> Dict[str, Any]:
        """
        Processa uma pergunta do usuário e retorna a resposta.
        
        Args:
            pergunta: Pergunta em linguagem natural
            
        Returns:
            Dicionário com resposta, gráficos gerados e metadados
        """
        try:
            # Adicionar à memória
            self.memoria.adicionar_interacao(pergunta, None)
            
            # Como agora sempre temos LLM disponível (ou o agente nem inicializa)
            return self._processar_com_langchain(pergunta)
                
        except Exception as e:
            erro_msg = f"Erro ao processar pergunta: {str(e)}"
            print(f"❌ {erro_msg}")
            print(traceback.format_exc())
            
            # Verificar se é um erro que pode indicar limitação de funcionalidade
            erro_str = str(e).lower()
            if any(termo in erro_str for termo in ['not supported', 'não suportado', 'cannot', 'não consegue', 'unable']):
                texto_erro = (
                    "**Funcionalidade não disponível no momento.**\n\n"
                    "Esta solicitação não pode ser realizada com as ferramentas atuais.\n\n"
                    "**Para novas funcionalidades:** Contate o desenvolvedor Igor Töebe para adicionar esta funcionalidade ao sistema."
                )
            else:
                texto_erro = (
                    "**Erro técnico temporário.**\n\n"
                    "Ocorreu um erro ao processar sua pergunta. Tente novamente ou reformule sua solicitação.\n\n"
                    "**Se o problema persistir:** Contate o desenvolvedor Igor Töebe para suporte técnico."
                )
            
            return {
                'texto': texto_erro,
                'erro': erro_msg
            }
    
    def _processar_com_langchain(self, pergunta: str) -> Dict[str, Any]:
        """Processa pergunta usando LangChain com Google Gemini."""
        try:
            # Executar agente
            resultado = self.agent_executor.invoke({
                "input": pergunta,
                "chat_history": self.memoria.get_historico_formatado()
            })
            
            resposta_texto = resultado.get('output', 'Sem resposta')
            
            # Filtrar código técnico da resposta
            resposta_texto = self._limpar_resposta_tecnica(resposta_texto)
            
            # Procurar por caminhos de gráficos na resposta e nos intermediate_steps
            graficos = []
            
            # Buscar na resposta final por caminhos de gráfico
            # Regex atualizada para incluir parênteses e espaços em nomes de arquivo
            graficos_resposta = re.findall(r'temp_graficos[/\\][\w\-_.()\s]+\.png', resposta_texto)
            graficos.extend(graficos_resposta)
            
            # Procurar nos intermediate_steps (saídas das ferramentas)
            intermediate_steps = resultado.get('intermediate_steps', [])
            dados_ml_completos = []
            
            for step in intermediate_steps:
                if len(step) > 1:
                    step_output = step[1]
                    
                    # 1) Quando a ferramenta retorna string: extrair caminhos e dados
                    if isinstance(step_output, str):
                        # Regex atualizada para incluir parênteses e espaços em nomes de arquivo
                        step_graficos = re.findall(r'temp_graficos[/\\][\w\-_.()\s]+\.png', step_output)
                        graficos.extend(step_graficos)
                        if any(palavra in step_output.lower() for palavra in ['clustering', 'pca', 'regressão', 'classificação', 'silhouette', 'variância']):
                            dados_ml_completos.append(step_output)
                    
                    # 2) Quando a ferramenta retorna dict: buscar chaves comuns com paths de imagem
                    elif isinstance(step_output, dict):
                        # Tentar coletar caminho(s) de gráfico por chaves comuns
                        for k in ['path', 'arquivo', 'grafico', 'file', 'figure_path', 'image', 'plot']:
                            if k in step_output:
                                val = step_output[k]
                                if isinstance(val, str) and val.lower().endswith('.png'):
                                    graficos.append(val)
                                elif isinstance(val, list):
                                    for v in val:
                                        if isinstance(v, str) and v.lower().endswith('.png'):
                                            graficos.append(v)
                        # Agregar possíveis textos/relatórios de ML
                        for k in ['relatorio', 'resultado', 'texto', 'report', 'details']:
                            if k in step_output and isinstance(step_output[k], str):
                                if any(p in step_output[k].lower() for p in ['clustering', 'pca', 'regressão', 'classificação', 'silhouette', 'variância']):
                                    dados_ml_completos.append(step_output[k])
            
            # Se temos dados de ML nos intermediate_steps, garantir que estão na resposta final
            if dados_ml_completos and not any(palavra in resposta_texto.lower() for palavra in ['clustering', 'pca', 'silhouette', 'variância']):
                resposta_texto += '\n\n' + '\n\n'.join(dados_ml_completos)
            
            # REMOVIDO: Busca automática de gráficos que causava exibição não solicitada
            # A busca por gráficos agora ocorre APENAS se explicitamente mencionados na resposta
            
            # Remover duplicatas mantendo ordem e verificar se arquivos existem
            graficos_validos = []
            for grafico in graficos:
                caminho_norm = str(Path(grafico))
                if caminho_norm not in graficos_validos and os.path.exists(caminho_norm):
                    graficos_validos.append(caminho_norm)
            
            graficos = graficos_validos
            
            # Se temos múltiplos gráficos, priorizar o mais recente
            if len(graficos) > 1:
                graficos.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
            
            # Atualizar memória
            self.memoria.adicionar_interacao(pergunta, resposta_texto)
            
            response = {
                'texto': resposta_texto,
                'graficos': graficos,
                'grafico': graficos[0] if graficos else None,
                'metadados': {
                    'modelo': 'google-gemini',
                    'ferramentas_usadas': resultado.get('intermediate_steps', [])
                }
            }
            
            return response
            
        except Exception as e:
            erro_str = str(e)
            
            # Verificar tipo de erro para dar resposta adequada
            if "API key not valid" in erro_str or "API_KEY_INVALID" in erro_str:
                texto_erro = (
                    "**Erro de autenticação com Google Gemini**\n\n"
                    "Sua chave API não é válida ou expirou.\n\n" 
                    "**Solução:** Verifique sua chave API em https://makersuite.google.com/app/apikey\n\n"
                    "**Precisa de ajuda?** Contate o desenvolvedor Igor Töebe para suporte técnico."
                )
            elif "quota" in erro_str.lower() or "limit" in erro_str.lower():
                texto_erro = (
                    "**Limite de uso atingido**\n\n"
                    "Você atingiu o limite da API do Google Gemini.\n\n"
                    "**Solução:** Aguarde um tempo ou verifique sua cota em https://console.cloud.google.com\n\n"
                    "**Precisa de ajuda?** Contate o desenvolvedor Igor Töebe para orientação."
                )
            else:
                texto_erro = (
                    "**Erro técnico no processamento**\n\n"
                    "Ocorreu um problema técnico ao processar sua solicitação.\n\n"
                    "**Tente:** Reformular sua pergunta ou tentar novamente\n\n"
                    "**Se persistir:** Contate o desenvolvedor Igor Töebe para suporte técnico."
                )
            
            return {
                'texto': texto_erro,
                'erro': erro_str
            }
    

    
    def get_sugestoes_contextuais(self) -> List[str]:
        """Retorna sugestões baseadas no contexto atual dos dados e capacidades reais do agente."""
        sugestoes = []
        
        # Sugestões baseadas nas colunas disponíveis
        colunas_numericas = self.df.select_dtypes(include=['number']).columns.tolist()
        colunas_categoricas = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Sugestão prioritária: análise de tipos de colunas
        sugestoes.append("Sugira quais colunas podem ser categóricas")
        
        # Sugestões de visualização específicas por coluna
        if colunas_numericas:
            primeira_col = colunas_numericas[0]
            sugestoes.extend([
                f"Gere um histograma da coluna '{primeira_col}'",
                f"Detecte outliers em '{primeira_col}'",
                f"Analise estatísticas de '{primeira_col}'"
            ])
            
            if len(colunas_numericas) > 1:
                segunda_col = colunas_numericas[1]
                sugestoes.append(f"Crie gráfico de dispersão entre '{primeira_col}' e '{segunda_col}'")
        
        if colunas_categoricas:
            primeira_cat = colunas_categoricas[0]
            sugestoes.append(f"Analise frequências em '{primeira_cat}'")
            
            # Sugestão de análise por grupos se tiver numérica e categórica
            if colunas_numericas and colunas_categoricas:
                sugestoes.append(f"Compare '{colunas_numericas[0]}' por grupos de '{primeira_cat}'")
        
        # Análises avançadas disponíveis
        sugestoes.extend([
            "Calcule matriz de correlação com heatmap",
            "Descreva o dataset completo"
        ])
        
        return sugestoes[:8]  # Expandir para 8 sugestões
    
    def _limpar_resposta_tecnica(self, resposta: str) -> str:
        """Remove código técnico e debugging da resposta do agente, preservando resultados de ML."""
        # Padrões de código técnico a serem removidos (mais específicos)
        padroes_tecnicos = [
            r'^print\(.*?\)$',  # Comandos print isolados
            r'^default_api\.\w+\([^)]*\)$',  # Chamadas de API interna isoladas
            r'^tool\.\w+\([^)]*\)$',  # Chamadas de ferramentas técnicas isoladas
            r'^function\([^)]*\)$',  # Chamadas genéricas de função isoladas
            r'^>>>.*?$',  # Prompts de Python
            r'^>>>\s+.*?$',  # Prompts de Python com espaços
            r'^Traceback \(most recent call last\):.*?$',  # Stack traces específicos
            r'^\s*File ".*?", line \d+.*?$',  # Linhas de stack trace
            r'^\s*\w+Error:.*?$',  # Erros específicos do Python isolados
        ]
        
        linhas = resposta.split('\n')
        linhas_limpas = []
        
        for linha in linhas:
            linha_limpa = linha.strip()
            
            # Verificar se a linha contém código técnico
            contem_codigo_tecnico = False
            for padrao in padroes_tecnicos:
                if re.match(padrao, linha_limpa, re.IGNORECASE):
                    contem_codigo_tecnico = True
                    break
            
            # Preservar linhas importantes dos resultados de ML
            if not contem_codigo_tecnico:
                linhas_limpas.append(linha)
        
        return '\n'.join(linhas_limpas)
    
    def get_info_dataset(self) -> Dict[str, Any]:
        """Retorna informações básicas sobre o dataset."""
        return {
            'linhas': len(self.df),
            'colunas': len(self.df.columns),
            'colunas_numericas': self.df.select_dtypes(include=['number']).columns.tolist(),
            'colunas_categoricas': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'memoria_mb': self.df.memory_usage(deep=True).sum() / (1024**2),
            'valores_nulos': self.df.isnull().sum().sum()
        }