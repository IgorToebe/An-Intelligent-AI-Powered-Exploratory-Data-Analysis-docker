"""
Utilitário para gerenciamento de configurações para Streamlit Cloud.

Este módulo fornece uma função unificada para acessar configurações
com a seguinte ordem de prioridade:
1. st.secrets (Streamlit Cloud)
2. Variáveis de ambiente (.env)
3. Valores padrão

Autor: Igor Töebe Lopes Farias
Versão: 1.0
"""

import os
import streamlit as st
from typing import Any, Optional


def get_config(key: str, default: Any = None) -> Any:
    """
    Busca configuração com prioridade correta para Streamlit Cloud.
    
    Ordem de prioridade:
    1. st.secrets (Streamlit Cloud)
    2. Variáveis de ambiente (.env)
    3. Valor padrão
    
    Args:
        key: Chave da configuração
        default: Valor padrão se não encontrado
        
    Returns:
        Valor da configuração
    """
    # 1. Tentar st.secrets primeiro (Streamlit Cloud)
    try:
        if hasattr(st, 'secrets'):
            # Tentar primeiro no nível superior (recomendado)
            if key in st.secrets:
                return st.secrets[key]
            # Tentar na seção [secrets] se configurado dessa forma
            elif 'secrets' in st.secrets and key in st.secrets['secrets']:
                return st.secrets['secrets'][key]
    except Exception:
        # st.secrets pode não estar disponível em desenvolvimento
        pass
    
    # 2. Tentar variáveis de ambiente (.env)
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value
    
    # 3. Retornar valor padrão
    return default


def get_api_key() -> Optional[str]:
    """
    Busca a chave da API Google Gemini com todas as fontes possíveis.
    
    Returns:
        Chave da API ou None se não encontrada
    """
    return get_config('GOOGLE_API_KEY')


def is_config_available() -> bool:
    """
    Verifica se as configurações básicas estão disponíveis.
    
    Returns:
        True se as configurações estão acessíveis
    """
    try:
        api_key = get_api_key()
        return api_key is not None and api_key.strip() != ""
    except Exception:
        return False


def get_all_configs() -> dict:
    """
    Retorna todas as configurações disponíveis.
    
    Returns:
        Dicionário com todas as configurações
    """
    configs = {
        # LLM
        'GOOGLE_API_KEY': get_config('GOOGLE_API_KEY'),
        'GEMINI_MODEL': get_config('GEMINI_MODEL', 'gemini-pro'),
        'GEMINI_TEMPERATURE': get_config('GEMINI_TEMPERATURE', '0.1'),
        'GEMINI_MAX_TOKENS': get_config('GEMINI_MAX_TOKENS', '2000'),
        
        # Agente
        'MEMORIA_LIMITE_INTERACOES': get_config('MEMORIA_LIMITE_INTERACOES', '50'),
        'AGENTE_MAX_ITERACOES': get_config('AGENTE_MAX_ITERACOES', '10'),
        'AGENTE_VERBOSE': get_config('AGENTE_VERBOSE', 'true'),
        
        # Processamento
        'MAX_ARQUIVO_SIZE_MB': get_config('MAX_ARQUIVO_SIZE_MB', '200'),
        'CHUNK_SIZE_PADRAO': get_config('CHUNK_SIZE_PADRAO', '10000'),
        'AUTO_OTIMIZAR_TIPOS': get_config('AUTO_OTIMIZAR_TIPOS', 'true'),
        
        # Visualização
        'GRAFICOS_DPI': get_config('GRAFICOS_DPI', '300'),
        'GRAFICOS_TEMP_DIR': get_config('GRAFICOS_TEMP_DIR', 'temp_graficos'),
        'GRAFICOS_FORMATO': get_config('GRAFICOS_FORMATO', 'png'),
        'GRAFICOS_ESTILO': get_config('GRAFICOS_ESTILO', 'darkgrid'),
        'GRAFICOS_PALETA': get_config('GRAFICOS_PALETA', 'husl'),
        
        # Interface
        'APP_TITULO': get_config('APP_TITULO', 'Agente de Análise Exploratória de Dados'),
        'APP_LAYOUT': get_config('APP_LAYOUT', 'wide'),
        'APP_TEMA': get_config('APP_TEMA', 'auto'),
        
        # Logs
        'LOG_LEVEL': get_config('LOG_LEVEL', 'INFO'),
        'LOG_FILE': get_config('LOG_FILE', ''),
        
        # Avançadas
        'MEMORIA_LONGO_PRAZO': get_config('MEMORIA_LONGO_PRAZO', 'false'),
        'INSIGHTS_DIR': get_config('INSIGHTS_DIR', 'memoria_insights'),
        'CACHE_ANALISES': get_config('CACHE_ANALISES', 'true'),
        'TIMEOUT_ANALISES': get_config('TIMEOUT_ANALISES', '300'),
        
        # Desenvolvimento
        'DEV_MODE': get_config('DEV_MODE', 'false'),
        'AUTO_RELOAD': get_config('AUTO_RELOAD', 'false'),
        'DEV_PORT': get_config('DEV_PORT', '8501'),
    }
    
    return configs


def print_config_status():
    """
    Imprime o status das configurações para debug.
    """
    print("=== STATUS DAS CONFIGURAÇÕES ===")
    
    # Verificar st.secrets
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            print("✅ st.secrets: Disponível")
        else:
            print("❌ st.secrets: Não disponível")
    except Exception as e:
        print(f"❌ st.secrets: Erro - {e}")
    
    # Verificar .env
    api_key_env = os.getenv('GOOGLE_API_KEY')
    if api_key_env:
        print("✅ .env: API Key encontrada")
    else:
        print("❌ .env: API Key não encontrada")
    
    # Status final
    if is_config_available():
        print("✅ STATUS FINAL: Configurações OK")
    else:
        print("❌ STATUS FINAL: Configurações faltando")
    
    print("================================")
