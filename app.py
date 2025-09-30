"""
Plataforma de Análise Exploratória Inteligente

Interface web moderna para análise exploratória de dados usando IA.
Permite upload de CSV e interação via linguagem natural com agente inteligente.

Autor: Igor Töebe Lopes Farias.
Versão: 2.0.1 - SSL Fix
"""

import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Nota: Para produção no Cloud Run, variáveis são definidas via --set-env-vars
# Para desenvolvimento local, use .env ou st.secrets

def converter_para_arrow_compativel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte DataFrame para tipos compatíveis com Arrow/Streamlit.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame com tipos compatíveis
        
    Note:
        Necessário para evitar problemas de serialização no Streamlit
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        dtype = str(df_copy[col].dtype)
        
        if dtype == 'float32':
            df_copy[col] = df_copy[col].astype('float64')
        elif dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
        elif any(t in dtype for t in ['int8', 'int16']):
            df_copy[col] = df_copy[col].astype('int64')
            
    return df_copy

# ========== IMPORTS DE BIBLIOTECAS ==========

# Bibliotecas de visualização
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    st.error(f"Bibliotecas de visualização indisponíveis: {e}")
    st.info("Execute: pip install matplotlib seaborn")

# ========== IMPORTS DOS MÓDULOS LOCAIS ==========

try:
    from processamento.carregador_dados import (
        carregar_csv, 
        obter_info_arquivo, 
        recomendar_estrategia_carregamento
    )
    from processamento.gcs_manager import (
        create_streamlit_file_uploader_with_gcs, 
        setup_gcs_environment
    )
    # AgenteEDA será importado quando necessário (lazy loading)
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    MODULES_AVAILABLE = False
    st.stop()


def get_agente_class():
    """Importa AgenteEDA apenas quando necessário (lazy loading)."""
    try:
        from agente.agente_core import AgenteEDA
        return AgenteEDA
    except ImportError as e:
        st.error(f"Erro ao carregar o módulo de IA: {e}")
        return None


# ========== CONFIGURAÇÃO DA APLICAÇÃO ==========

def configurar_pagina():
    """
    Configura a página Streamlit com design moderno e profissional.
    
    Aplica configurações de layout, tema e estilos CSS personalizados.
    """
    st.set_page_config(
        page_title="Plataforma de Análise Exploratória Inteligente",
        page_icon="▲",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Plataforma de Análise Exploratória com IA"
        }
    )
    
    # Configurar tema escuro no Streamlit
    st.markdown("""
    <script>
    var elements = window.parent.document.querySelectorAll('.stApp');
    var backgroundColor = '#1A1D23';
    for (var i = 0; i < elements.length; i++) {
        elements[i].style.backgroundColor = backgroundColor;
    }
    </script>
    """, unsafe_allow_html=True)
    
    # CSS customizado para interface profissional e futurística
    # Aplicar estilos CSS personalizados
    aplicar_estilos_css()


def aplicar_estilos_css():
    """
    Aplica estilos CSS personalizados para interface moderna.
    
    Define variáveis CSS, estiliza componentes e aplica tema dark moderno.
    """
    st.markdown("""
    <style>
    :root {
        --primary-color: #1a3079;
        --secondary-color: #50C878;
        --accent-color: #F5A623;
        --success-color: #27AE60;
        --warning-color: #E67E22;
        --error-color: #E74C3C;
        --background-main: #0F1419;
        --background-card: #1A1F2E;
        --background-elevated: #232A3C;
        --background-glass: rgba(26, 31, 46, 0.85);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-accent: #64FFDA;
        --text-muted: #64748B;
        --border-color: #334155;
        --border-subtle: #1E293B;
        --border-focus: #1a3079;
        --gradient-primary: linear-gradient(135deg, #1a3079 0%, #1a3079 100%);
        --gradient-card: linear-gradient(145deg, #1A1F2E 0%, #232A3C 100%);
        --gradient-glass: linear-gradient(145deg, rgba(26, 31, 46, 0.9) 0%, rgba(35, 42, 60, 0.8) 100%);
        --gradient-accent: linear-gradient(135deg, #1a3079 0%, #1a3079 100%);
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.16), 0 2px 4px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.15), 0 10px 10px rgba(0, 0, 0, 0.04);
        --blur-sm: blur(4px);
        --blur-md: blur(8px);
        --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Modern app layout with professional spacing */
    .main .block-container {
        padding: 3rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
        background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 50%, #0F1419 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 50%, #0F1419 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .stApp > header {
        background: rgba(15, 20, 25, 0.95);
        backdrop-filter: var(--blur-md);
        border-bottom: 1px solid var(--border-subtle);
    }
    
    /* Professional typography scale */
    body {
        font-feature-settings: "kern" 1, "liga" 1, "calt" 1;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Smooth scroll and accessibility */
    html {
        scroll-behavior: smooth;
    }
    
    *:focus-visible {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
        border-radius: 4px;
    }
    
    /* Professional animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(74, 144, 226, 0.3); }
        50% { box-shadow: 0 0 20px rgba(74, 144, 226, 0.5); }
    }
    
    .main-header {
        color: var(--text-primary);
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        text-align: center;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #F8FAFC 0%, #64FFDA 50%, #5B9BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 2px;
        background: var(--gradient-primary);
        border-radius: 1px;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.125rem;
        margin: 1.5rem 0 0.5rem 0;
        font-weight: 400;
        letter-spacing: 0.025em;
        line-height: 1.6;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .precision-badge {
        background: var(--gradient-glass);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 0.5rem 1.5rem;
        display: inline-block;
        margin: 1rem auto 3rem auto;
        color: var(--text-accent);
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        backdrop-filter: var(--blur-sm);
        box-shadow: var(--shadow-md);
        transition: var(--transition-normal);
        position: relative;
        overflow: hidden;
    }
    
    .precision-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(100, 255, 218, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .precision-badge:hover::before {
        left: 100%;
    }
    
    .precision-badge:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
        border-color: var(--text-accent);
    }
    
    .modern-card {
        background: var(--gradient-glass);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        transition: var(--transition-normal);
        box-shadow: var(--shadow-lg);
        backdrop-filter: var(--blur-sm);
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--text-accent), transparent);
        opacity: 0.5;
    }
    
    .modern-card:hover {
        border-color: var(--border-focus);
        box-shadow: var(--shadow-xl);
        transform: translateY(-2px) scale(1.001);
    }
    
    .modern-card h2, .modern-card h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
        line-height: 1.3;
    }
    
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 300;
        padding: 0.875rem 2rem;
        transition: var(--transition-normal);
        font-size: 0.875rem;
        letter-spacing: 1px;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
        font-family: inherit;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a3079 0%, #1a3079 100%);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-md);
    }
    
    .metric-card {
        background: var(--gradient-glass);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 0.75rem 0;
        transition: var(--transition-normal);
        box-shadow: var(--shadow-md);
        backdrop-filter: var(--blur-sm);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(74, 144, 226, 0.05), transparent);
        transition: left 0.8s ease;
    }
    
    .metric-card:hover::after {
        left: 100%;
    }
    
    .metric-card:hover {
        border-color: var(--border-focus);
        box-shadow: var(--shadow-lg);
        transform: translateY(-3px) scale(1.02);
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--text-accent);
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 0.8125rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Estilos para container de gráficos e títulos para maior legibilidade */
    .chart-container {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--border-subtle);
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.75rem 0;
    }

    .chart-container h4 {
        color: var(--text-primary) !important;
        margin-top: 0;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }

    .chart-container img {
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }
    
    .status-card {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-subtle);
        transition: all 0.2s ease;
    }
    
    .status-card:hover {
        border-color: var(--border-subtle);
        box-shadow: var(--shadow-elevated);
    }
    
    /* Streamlit specific modern adjustments */
    .stSelectbox > div > div {
        background: var(--gradient-glass);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        backdrop-filter: var(--blur-xs);
        transition: var(--transition-fast);
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--border-focus);
        box-shadow: var(--shadow-sm);
    }
    
    .stTextInput > div > div > input {
        background: var(--gradient-glass);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        backdrop-filter: var(--blur-xs);
        transition: var(--transition-fast);
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1), var(--shadow-sm);
        outline: none;
    }
    
    .stTextInput > div > div > input:hover:not(:focus) {
        border-color: var(--border-focus);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--background-card);
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #CBD5E1;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
        border-bottom-color: var(--primary-color);
    }
    
    .stDataFrame {
        background-color: var(--background-card);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
    }
    
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .stDataFrame th {
        background: var(--gradient-subtle);
        color: var(--text-primary);
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-bottom: 2px solid var(--border-subtle);
    }
    
    .stDataFrame td {
        padding: 0.625rem 1rem;
        border-bottom: 1px solid var(--border-color);
        transition: background-color 0.2s ease;
    }
    
    .stDataFrame tr:hover td {
        background-color: rgba(74, 144, 226, 0.05);
    }
    
    .stMetric {
        background-color: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 1rem;
    }
    
    .stSidebar {
        background-color: var(--background-elevated);
    }
    
    .stSidebar .stButton > button {
        background-color: #1a3079;
        color: white;
        border: none;
        width: 100%;
    }
    
    .stSidebar .stButton > button:hover {
        background-color: #1a3079;
        transform: translateY(-1px);
    }
    
    .chat-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background: var(--background-elevated);
        border-left: 3px solid var(--primary-color);
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: var(--text-primary);
    }
    
    .agent-message {
        background: var(--background-elevated);
        border-left: 3px solid var(--secondary-color);
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: var(--text-primary);
    }
    
    .chart-container {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Professional footer */
    .footer {
        margin-top: 6rem;
        padding: 3rem 0 2rem;
        border-top: 1px solid var(--border-color);
        text-align: center;
        background: var(--gradient-subtle);
        position: relative;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: var(--gradient-accent);
        border-radius: 2px;
    }
    
    .footer-text {
        color: var(--text-secondary);
        font-size: 0.9375rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        margin-top: 1rem;
    }
    
    .footer-brand {
        font-size: 1.125rem;
        color: var(--text-accent);
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .notification-warning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.08), rgba(255, 193, 7, 0.04));
        border: 1px solid rgba(255, 193, 7, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: var(--warning-color);
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.1);
        backdrop-filter: var(--blur-xs);
        position: relative;
        overflow: hidden;
    }
    
    .notification-warning::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--warning-color);
        border-radius: 0 2px 2px 0;
    }
    
    .notification-info {
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.08), rgba(74, 144, 226, 0.04));
        border: 1px solid rgba(74, 144, 226, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.1);
        backdrop-filter: var(--blur-xs);
        position: relative;
        overflow: hidden;
    }
    
    .notification-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--primary-color);
        border-radius: 0 2px 2px 0;
    }
    
    .notification-error {
        background: var(--background-elevated);
        border: 1px solid var(--error-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--error-color);
    }
    </style>
    """, unsafe_allow_html=True)


# ========== INICIALIZAÇÃO E ESTADO ==========

def inicializar_sessao():
    """
    Inicializa variáveis de estado da sessão.
    
    Define valores padrão para todas as variáveis de estado necessárias
    para o funcionamento da aplicação.
    """
    # Configurações de sessão
    # Tentar obter API key do Streamlit secrets (produção) ou .env (desenvolvimento)
    api_key_default = None
    try:
        # Primeiro tentar st.secrets (Streamlit Cloud)
        api_key_default = st.secrets.get("GOOGLE_API_KEY", None)
    except (FileNotFoundError, AttributeError, KeyError):
        # Se não houver secrets, tentar variável de ambiente
        api_key_default = os.getenv('GOOGLE_API_KEY')
    
    session_defaults = {
        'sessao_inicializada': True,
        'df_carregado': None,
        'info_arquivo': None,
        'agente': None,
        'historico_chat': [],
        'arquivo_nome': None,
        'api_key': api_key_default,
        'llm_model': 'gemini-2.5-flash'  # Modelo LLM padrão
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ========== INTERFACE DE DADOS ==========

def mostrar_informacoes_arquivo(info_arquivo: Dict[str, Any]):
    """
    Exibe informações detalhadas sobre o arquivo carregado.
    
    Args:
        info_arquivo: Dicionário com metadados do arquivo
    """
    st.markdown("### Informações do Arquivo")
    
    # Métricas principais em colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tamanho", f"{info_arquivo['tamanho_mb']} MB")
        st.metric("Colunas", info_arquivo['num_colunas'])
    
    with col2:
        st.metric("Linhas (estimadas)", f"{info_arquivo['linhas_estimadas']:,}")
    
    with col3:
        if st.button("Ver Amostra dos Dados"):
            _exibir_amostra_dados()


def _exibir_amostra_dados():
    """
    Função auxiliar para exibir amostra dos dados carregados.
    
    Mostra preview do DataFrame e informações sobre tipos de dados.
    """
    if st.session_state.df_carregado is not None:
        df = st.session_state.df_carregado
        
        # Preview dos dados
        st.markdown("#### Primeiras 5 linhas:")
        df_display = converter_para_arrow_compativel(df.head())
        st.dataframe(df_display, use_container_width=True)
        
        # Informações dos tipos de dados
        st.markdown("#### Tipos de dados:")
        tipos_df = pd.DataFrame({
            'Coluna': df.columns.astype(str),
            'Tipo': df.dtypes.astype(str),
            'Nulos': df.isnull().sum().astype(int),
            'Únicos': df.nunique().astype(int)
        }).astype(str)
        
        st.dataframe(tipos_df, use_container_width=True)


# ========== UPLOAD DE ARQUIVOS ==========

def carregar_arquivo():
    """
    Interface para upload e carregamento de arquivos CSV.
    
    Permite upload, análise de metadados e carregamento otimizado
    de datasets para análise exploratória.
    """
    # Header da seção de upload
    st.markdown("""
    <div class="modern-card">
        <h2 style="margin-top: 0; color: var(--text-primary); font-weight: 300; font-size: 1.5rem;">
            Pré-processamento e Engenharia de Variáveis
        </h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem; font-size: 0.9rem;">
            Carregue seu dataset (CSV) para limpeza, transformação, análise de qualidade e engenharia de
            variáveis — preparado para análises estatísticas e modelagem.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configurar ambiente GCS e verificar se está funcionando
    gcs_configured = setup_gcs_environment()
    
    if not gcs_configured:
        st.error("🚨 Google Cloud Storage não está configurado!")
        st.info("📋 Configure as seguintes variáveis de ambiente:")
        st.code("""
GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
GCS_BUCKET_NAME=i2a2-eda-uploads
        """)
        st.stop()
    
    # Widget de upload otimizado para arquivos grandes
    df, blob_name = create_streamlit_file_uploader_with_gcs()
    
    if df is not None:
        # Processar DataFrame carregado
        _processar_dataframe_carregado(df, blob_name)


def _processar_dataframe_carregado(df: pd.DataFrame, blob_name: Optional[str] = None):
    """
    Processa o DataFrame carregado (via upload tradicional ou GCS).
    
    Args:
        df: DataFrame carregado
        blob_name: Nome do blob no GCS (se aplicável)
    """
    try:
        # Converter para tipos compatíveis com Arrow
        df = converter_para_arrow_compativel(df)
        
        # Simular informações de arquivo para compatibilidade
        info_arquivo = {
            'tamanho_mb': len(df) * len(df.columns) * 8 / (1024 * 1024),  # Estimativa
            'num_linhas': len(df),
            'num_colunas': len(df.columns),
            'memoria_estimada_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Mostrar informações do DataFrame
        st.success(f"✅ **Dataset carregado com sucesso!**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Linhas", f"{info_arquivo['num_linhas']:,}")
        with col2:
            st.metric("📋 Colunas", info_arquivo['num_colunas'])
        with col3:
            st.metric("💾 Memória", f"{info_arquivo['memoria_estimada_mb']:.1f} MB")
        with col4:
            fonte = "☁️ GCS" if blob_name else "📁 Local"
            st.metric("📂 Fonte", fonte)
        
        # Salvar na sessão
        st.session_state.df_carregado = df
        st.session_state.info_arquivo = info_arquivo
        st.session_state.arquivo_carregado = True
        
        # Mostrar prévia do dataset
        st.markdown("### 👀 Prévia do Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Botão para iniciar análise
        st.markdown("---")
        if st.button("🚀 Iniciar Análise Exploratória", type="primary", use_container_width=True):
            # Verificar API key
            if not st.session_state.get('api_key'):
                st.error("⚠️ Configure sua API key do Google Gemini nas configurações para continuar.")
                return
            
            # Inicializar agente
            with st.spinner("🤖 Inicializando agente de IA..."):
                AgenteEDA = get_agente_class()
                if AgenteEDA:
                    st.session_state.agente = AgenteEDA(
                        df,
                        api_key=st.session_state.api_key,
                        llm_model=st.session_state.get('llm_model', None)
                    )
                    st.session_state.analise_iniciada = True
                    st.rerun()
        
    except Exception as e:
        st.error(f"❌ Erro ao processar dados: {e}")
        st.error("Verifique se o arquivo está no formato correto (CSV com cabeçalhos)")


def _processar_upload_arquivo(arquivo_upload):
    """
    Processa o arquivo enviado pelo usuário.
    
    Args:
        arquivo_upload: Objeto do arquivo enviado via Streamlit
    """
    # Salvar temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(arquivo_upload.getvalue())
        caminho_temp = tmp_file.name
    
    try:
        # Analisar arquivo
        with st.spinner("Analisando arquivo..."):
            info_arquivo = obter_info_arquivo(caminho_temp)
            recomendacao = recomendar_estrategia_carregamento(caminho_temp)
        
        # Mostrar estratégia de otimização
        st.success(f"**Estratégia**: {recomendacao['justificativa']}")
        
        # Configurações de carregamento
        col1, col2 = st.columns(2)
        
        with col1:
            otimizar_memoria = st.checkbox(
                "Habilitar Otimização de Tipos de Dados", 
                value=recomendacao['otimizar_memoria'],
                help="Converte tipos de dados para reduzir uso de memória"
            )
        
        with col2:
            chunksize = None
            if recomendacao['chunksize']:
                usar_chunks = st.checkbox(
                    f"Carregamento em Blocos ({recomendacao['chunksize']:,} linhas)",
                    value=True,
                    help="Processa arquivos grandes em blocos eficientes de memória"
                )
                chunksize = recomendacao['chunksize'] if usar_chunks else None
        
        # Botão de carregamento
        st.markdown("---")
        if st.button("Inicializar Processamento de Dados", type="primary", use_container_width=True):
            _carregar_e_inicializar_dados(caminho_temp, arquivo_upload.name, 
                                        info_arquivo, chunksize, otimizar_memoria)
                    
    except Exception as e:
        st.error(f"Erro ao analisar arquivo: {str(e)}")
    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(caminho_temp)
        except:
            pass


def _carregar_e_inicializar_dados(caminho_temp, nome_arquivo, info_arquivo, 
                                  chunksize, otimizar_memoria):
    """
    Carrega os dados e inicializa o agente.
    
    Args:
        caminho_temp: Caminho do arquivo temporário
        nome_arquivo: Nome original do arquivo
        info_arquivo: Metadados do arquivo
        chunksize: Tamanho dos blocos para carregamento
        otimizar_memoria: Se deve otimizar uso de memória
    """
    try:
        # Carregar dados
        with st.spinner("Carregando dados..."):
            df = carregar_csv(caminho_temp, chunksize=chunksize, 
                            otimizar_memoria=otimizar_memoria)
        
        # Salvar no estado da sessão
        st.session_state.df_carregado = df
        st.session_state.info_arquivo = info_arquivo
        st.session_state.arquivo_nome = nome_arquivo
        
        # Inicializar agente IA
        with st.spinner("Inicializando agente IA..."):
            AgenteEDA = get_agente_class()
            if AgenteEDA:
                st.session_state.agente = AgenteEDA(df, api_key=st.session_state.api_key)
        
        # Feedback de sucesso
        memoria_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.success(f"""
        **Dataset Carregado com Sucesso**
        
        Registros: {len(df):,} | Características: {len(df.columns)} | Memória: {memoria_mb:.1f} MB
        """)
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        if st.checkbox("Mostrar detalhes técnicos"):
            st.code(traceback.format_exc())


def executar_com_progresso_ml(agente, pergunta: str):
    """Executa operação de ML com indicador de progresso avançado."""
    pergunta_lower = pergunta.lower()
    
    # Detectar se é operação de ML pesada
    operacoes_ml = {
        'clustering': ['cluster', 'agrupamento', 'kmeans', 'dbscan'],
        'pca': ['pca', 'componentes principais', 'redução'],
        'modelo': ['regressão', 'regressao', 'modelo', 'treinar', 'prever', 'classificação', 'classificacao']
    }
    
    operacao_detectada = None
    for tipo, palavras in operacoes_ml.items():
        if any(palavra in pergunta_lower for palavra in palavras):
            operacao_detectada = tipo
            break
    
    if operacao_detectada:
        # Criar placeholder para progresso
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown("### Processamento de Análises Avançadas")
            
            if operacao_detectada == 'clustering':
                st.info("**Executando Clustering:** Normalizando dados → Aplicando algoritmo → Calculando métricas")
            elif operacao_detectada == 'pca':
                st.info("**Executando PCA:** Preparando matriz → Calculando componentes → Analisando variância")
            elif operacao_detectada == 'modelo':
                st.info("**Treinando Modelo:** Dividindo dados → Treinando → Validando → Calculando métricas")
            
            # Barra de progresso simulada
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simular progresso enquanto processa
            import time
            import threading
            
            resultado = [None]  # Lista para armazenar resultado (referência mutável)
            
            def processar():
                resultado[0] = agente.processar_pergunta(pergunta)
            
            # Iniciar processamento em thread separada
            thread = threading.Thread(target=processar)
            thread.start()
            
            # Simular progresso
            etapas = [
                "Preparando dados...",
                "Aplicando algoritmo...",
                "Calculando métricas...",
                "Gerando visualizações...",
                "Finalizando análise..."
            ]
            
            for i, etapa in enumerate(etapas):
                if not thread.is_alive():
                    break
                status_text.text(etapa)
                progress_bar.progress((i + 1) * 20)
                time.sleep(0.8)
            
            # Aguardar conclusão
            while thread.is_alive():
                time.sleep(0.1)
            
            progress_bar.progress(100)
            status_text.text("Processamento concluído!")
            time.sleep(0.5)
            
        # Limpar placeholder
        progress_placeholder.empty()
        return resultado[0]
    else:
        # Operação normal sem barra de progresso especial
        return agente.processar_pergunta(pergunta)


def enviar_pergunta_ao_agente(pergunta: str):
    """Envia uma pergunta para o agente e atualiza o histórico do chat."""
    if not st.session_state.agente or not pergunta.strip():
        return

    # Adicionar pergunta ao histórico
    st.session_state.historico_chat.append(("usuario", pergunta))

    try:
        # Detectar tipo de operação para spinner mais específico
        pergunta_lower = pergunta.lower()
        
        if any(palavra in pergunta_lower for palavra in ['cluster', 'agrupamento', 'kmeans', 'dbscan']):
            spinner_text = "Executando análise de clustering... Isso pode levar alguns segundos"
        elif any(palavra in pergunta_lower for palavra in ['pca', 'componentes principais', 'redução']):
            spinner_text = "Calculando componentes principais... Processando"
        elif any(palavra in pergunta_lower for palavra in ['regressão', 'regressao', 'modelo', 'treinar', 'prever']):
            spinner_text = "Treinando modelo (análises avançadas)... Aguarde"
        elif any(palavra in pergunta_lower for palavra in ['classificação', 'classificacao', 'classificar']):
            spinner_text = "Executando classificação... Processando dados"
        elif any(palavra in pergunta_lower for palavra in ['correlação', 'correlacao', 'heatmap']):
            spinner_text = "Calculando correlações... Quase pronto"
        elif any(palavra in pergunta_lower for palavra in ['gráfico', 'grafico', 'plot', 'visualização']):
            spinner_text = "Gerando visualização... Criando gráfico"
        else:
            spinner_text = "🤔 Agente analisando... Processando sua solicitação"
        
        # Usar progresso avançado para ML ou spinner normal para outras operações
        if any(palavra in pergunta_lower for palavra in ['cluster', 'agrupamento', 'kmeans', 'dbscan', 'pca', 'componentes principais', 'regressão', 'regressao', 'modelo', 'treinar', 'classificação', 'classificacao']):
            resposta = executar_com_progresso_ml(st.session_state.agente, pergunta)
        else:
            with st.spinner(spinner_text):
                resposta = st.session_state.agente.processar_pergunta(pergunta)

        # Processar resposta sem debug desnecessário

        # Processar resposta
        if isinstance(resposta, dict):
            # Primeiro adicionar a resposta de texto
            if 'texto' in resposta:
                st.session_state.historico_chat.append(("agente", resposta['texto']))
            
            # Processar gráficos - garantir que todos sejam exibidos sem duplicação
            graficos_encontrados = []
            
            if 'grafico' in resposta and resposta['grafico']:
                graficos_encontrados.append(resposta['grafico'])
            
            if 'graficos' in resposta and resposta['graficos']:
                for grafico in resposta['graficos']:
                    if grafico not in graficos_encontrados:
                        graficos_encontrados.append(grafico)
            
            # REMOVIDO: Detecção automática de gráficos que causava exibição não solicitada
            # Os gráficos agora são exibidos APENAS quando explicitamente retornados pelo agente
            
            # Verificar quais gráficos já estão no histórico para evitar duplicação
            graficos_existentes = [item[1] for item in st.session_state.historico_chat if item[0] == "grafico"]
            
            # Adicionar apenas gráficos novos ao histórico
            for grafico in graficos_encontrados:
                if grafico and os.path.exists(grafico) and grafico not in graficos_existentes:
                    st.session_state.historico_chat.append(("grafico", grafico))
            
            if 'erro' in resposta:
                st.session_state.historico_chat.append(("erro", resposta['erro']))
        else:
            st.session_state.historico_chat.append(("agente", str(resposta)))

    except Exception as e:
        erro_msg = f"Erro ao processar pergunta: {str(e)}"
        st.session_state.historico_chat.append(("erro", erro_msg))
        st.error(erro_msg)
    
    st.rerun()


def interface_chat():
    """Interface de chat moderna com o agente."""
    st.markdown("""
    <div class="modern-card">
        <h2 style="margin-top: 0; color: var(--text-primary); font-weight: 300; font-size: 1.5rem;">Neural Analytics - Análise Inteligente</h2>
        <p style="color: var(--text-secondary); margin-bottom: 0; font-size: 0.9rem;">
            Interação em linguagem natural com recursos avançados de análise estatística.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.agente is None:
        st.markdown("""
        <div class="notification-warning">
            <strong>Dataset Necessário</strong><br>
            Por favor, carregue um dataset CSV para iniciar a análise inteligente.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Mostrar histórico apenas se houver mensagens
    if st.session_state.historico_chat:
        # Container para o histórico de chat
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Mostrar histórico
        for i, (tipo, conteudo) in enumerate(st.session_state.historico_chat):
            if tipo == "usuario":
                st.markdown(f"""
                <div class="user-message">
                    <strong>Consulta:</strong> {conteudo}
                </div>
                """, unsafe_allow_html=True)
            elif tipo == "agente":
                st.markdown(f"""
                <div class="agent-message">
                    <strong>Análise IA:</strong><br>{conteudo}
                </div>
                """, unsafe_allow_html=True)
            elif tipo == "grafico":
                st.markdown("""
                <div class="chart-container">
                    <h4 style="color: var(--text-primary); margin-top: 0; font-weight: 600; font-size: 1.05rem;">Visualização Gerada</h4>
                """, unsafe_allow_html=True)
                try:
                    if os.path.exists(conteudo):
                        st.image(conteudo, caption=f"Gráfico: {os.path.basename(conteudo)}", use_column_width=True)
                    else:
                        st.markdown("""
                        <div class="notification-error">
                            <strong>Erro de Visualização</strong><br>
                            Arquivo de gráfico não encontrado: {}
                        </div>
                        """.format(conteudo), unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="notification-error">
                        <strong>Erro de Exibição</strong><br>
                        {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif tipo == "erro":
                st.markdown(f"""
                <div class="notification-error">
                    <strong>Erro do Sistema</strong><br>
                    {conteudo}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input para nova mensagem
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            pergunta = st.text_input(
                "Consulta em Linguagem Natural:",
                placeholder="Exemplo: Gere um histograma da coluna 'idade' ou Teste a normalidade dos dados de vendas",
                label_visibility="collapsed"
            )
        
        with col2:
            enviar = st.form_submit_button("Enviar", type="primary")
    
    # Processar mensagem
    if enviar and pergunta.strip():
        enviar_pergunta_ao_agente(pergunta)
    
    # Sugestões de análise inteligentes
    if st.session_state.df_carregado is not None:
        st.markdown("""
        <div class="modern-card">
            <h3 style="margin-top: 0; color: var(--text-primary);">Sugestões de Análise Inteligente</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Análise Exploratória**")
            sugestoes1 = [
                "Descreva a estrutura e qualidade dos dados",
                "Detecte outliers e anomalias nos dados",
                "Calcule matriz de correlação com análise",
                "Realize clustering automático para descobrir padrões",
                "Analise normalidade de todas as variáveis numéricas"
            ]
            
            for sugestao in sugestoes1:
                if st.button(sugestao, key=f"sug1_{sugestao}"):
                    enviar_pergunta_ao_agente(sugestao)
        
        with col2:
            st.markdown("**Análise Estatística**")
            # Sugestões baseadas nas colunas
            colunas_numericas = st.session_state.df_carregado.select_dtypes(include=['number']).columns.tolist()
            colunas_categoricas = st.session_state.df_carregado.select_dtypes(include=['object']).columns.tolist()
            
            if colunas_numericas:
                primeira_numerica = colunas_numericas[0]
                sugestoes2 = [
                    f"Realize clustering K-means no dataset completo",
                    f"Teste normalidade da variável '{primeira_numerica}'",
                    f"Gere histograma de '{primeira_numerica}' para ver distribuição",
                    f"Otimize o número de clusters automaticamente"
                ]
                
                # Adicionar sugestão de análise avançada se há duas colunas numéricas
                if len(colunas_numericas) > 1:
                    sugestoes2.append(f"Gere gráfico de dispersão entre '{primeira_numerica}' e '{colunas_numericas[1]}'")
                
                # Adicionar sugestão de teste estatístico se há coluna categórica
                if colunas_categoricas:
                    sugestoes2.append(f"Teste Mann-Whitney: '{primeira_numerica}' por grupos de '{colunas_categoricas[0]}'")
                
                for sugestao in sugestoes2:
                    if st.button(sugestao, key=f"sug2_{sugestao}"):
                        enviar_pergunta_ao_agente(sugestao)


def sidebar_informacoes():
    """Barra lateral moderna com informações do sistema."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 2rem 0;">
            <h1 style="color: var(--text-primary); margin: 0; font-size: 1.25rem; font-weight: 300; letter-spacing: 1px;">Analytical Platform</h1>
            <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 500;">
                Precision System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Informações do arquivo carregado
        if st.session_state.df_carregado is not None:
            st.markdown("""
            <div class="status-card">
                <h4 style="color: var(--text-primary); margin-top: 0; font-size: 1.25rem; font-weight: 300; letter-spacing: 1px; text-align: center;">Current Dataset</h4>
            """, unsafe_allow_html=True)
            
            # Métricas do dataset dentro do card
            registros = len(st.session_state.df_carregado)
            caracteristicas = len(st.session_state.df_carregado.columns)
            memory_mb = st.session_state.df_carregado.memory_usage(deep=True).sum() / (1024**2)
            
            st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                    <div style="text-align: center; padding: 0.75rem; background: rgba(26, 49, 121, 0.1); border-radius: 8px; border: 1px solid rgba(26, 49, 121, 0.2);">
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-accent); margin-bottom: 0.25rem;">{registros:,}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Registros</div>
                    </div>
                    <div style="text-align: center; padding: 0.75rem; background: rgba(26, 49, 121, 0.1); border-radius: 8px; border: 1px solid rgba(26, 49, 121, 0.2);">
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-accent); margin-bottom: 0.25rem;">{caracteristicas}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Características</div>
                    </div>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(26, 49, 121, 0.1); border-radius: 8px; border: 1px solid rgba(26, 49, 121, 0.2); margin: 0.5rem 0 1rem 0;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-accent); margin-bottom: 0.25rem;">{memory_mb:.1f} MB</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Memória</div>
                </div>
                <div style="background: rgba(46, 134, 171, 0.1); padding: 0.5rem; border-radius: 6px; margin: 1rem 0;">
                    <small style="color: var(--text-secondary);">
                        <strong>Arquivo:</strong> {st.session_state.arquivo_nome}
                    </small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Botão para limpar dados - centralizado
            st.markdown("<div style='text-align: center; margin: 1.5rem 0;'>", unsafe_allow_html=True)
            if st.button("Reiniciar Sistema", type="secondary", use_container_width=True):
                st.session_state.df_carregado = None
                st.session_state.info_arquivo = None
                st.session_state.agente = None
                st.session_state.historico_chat = []
                st.session_state.arquivo_nome = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Campo de API key com exibição mascarada
        api_key_stored = st.session_state.get('api_key', '')
        api_key_display = ''
        if api_key_stored:
            # Mostrar apenas os primeiros 4 e últimos 2 caracteres
            if len(api_key_stored) > 6:
                api_key_display = api_key_stored[:4] + '*' * (len(api_key_stored) - 6) + api_key_stored[-2:]
            else:
                api_key_display = '*' * len(api_key_stored)
        
        st.markdown("<div style='text-align: center; margin: 1rem 0;'><small style='color: var(--text-secondary); font-weight: 500;'>Configuração da API</small></div>", unsafe_allow_html=True)
        api_key = st.text_input(
            "Intelligence Engine API Key", 
            value=api_key_display,
            help="Enter your Google Gemini API key for advanced AI analysis",
            label_visibility="collapsed"
        )
        if api_key:
            # Verifica se a API key mudou (ignorar se for a versão mascarada)
            api_key_mudou = False
            
            # Se a chave digitada não contém asteriscos, é uma nova chave
            if '*' not in api_key:
                api_key_mudou = ('api_key' not in st.session_state or 
                               st.session_state.api_key != api_key)
                st.session_state.api_key = api_key
            # Se contém asteriscos, usar a chave armazenada
            elif 'api_key' in st.session_state:
                api_key = st.session_state.api_key
            st.markdown("""
            <div style="background: var(--background-elevated); border: 1px solid var(--success-color); padding: 0.5rem; border-radius: 4px; color: var(--success-color);">
                <small><strong>API Active</strong></small>
            </div>
            """, unsafe_allow_html=True)
            
            # Espaço proporcional entre seções
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            
            # Recria o agente se a API key mudou e há dados carregados
            if api_key_mudou and st.session_state.df_carregado is not None:
                try:
                    AgenteEDA = get_agente_class()
                    if AgenteEDA:
                        st.session_state.agente = AgenteEDA(st.session_state.df_carregado, api_key=st.session_state.api_key)
                    # Verificar se o agente está usando LLM
                    if hasattr(st.session_state.agente, 'llm_disponivel') and st.session_state.agente.llm_disponivel:
                        st.markdown("""
                        <div style="background: var(--background-elevated); border: 1px solid var(--success-color); padding: 0.5rem; border-radius: 4px;">
                            <small style="color: var(--success-color);"><strong>Intelligence Agent Online</strong></small>
                        </div>
                        """, unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.markdown(f"""
                    <div style="background: rgba(231, 76, 60, 0.15); padding: 0.5rem; border-radius: 6px; color: var(--error-color);">
                        <small><strong>Erro de Configuração:</strong> {str(e)}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Espaço proporcional entre seções
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
        
        # Status do sistema
        if hasattr(st.session_state, 'agente') and st.session_state.agente:
            if hasattr(st.session_state.agente, 'llm_disponivel'):
                if st.session_state.agente.llm_disponivel:
                    st.markdown("""
                    <div style="background: var(--background-elevated); border: 1px solid var(--success-color); padding: 0.5rem; border-radius: 4px;">
                        <small style="color: var(--success-color);"><strong>Status:</strong> Intelligence System Active</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(243, 156, 18, 0.15); padding: 0.5rem; border-radius: 6px;">
                        <small style="color: var(--warning-color);"><strong>Status:</strong> Modo Básico</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        if st.button("Limpar Sessão", type="secondary", use_container_width=True):
            st.session_state.historico_chat = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Ações de visualização
        st.markdown("<div style='text-align: center; margin: 1rem 0;'><small style='color: var(--text-secondary); font-weight: 500;'>Gerenciador de Visualização</small></div>", unsafe_allow_html=True)
        
        from visualizacao.gerador_graficos import arquivar_graficos_temporarios
        st.markdown("<div style='text-align: center; margin: 1rem 0;'>", unsafe_allow_html=True)
        if st.button("Arquivar Gráficos Atuais", type="secondary", use_container_width=True):
            msg = arquivar_graficos_temporarios()
            st.success(msg)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Informações técnicas
        st.markdown("<div style='text-align: center; margin: 1rem 0;'><small style='color: var(--text-secondary); font-weight: 500;'>Technology Stack</small></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="status-card">
            <div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.7; text-align: center;">
                <strong>Intelligence Engine:</strong> Google Gemini API<br>
                <strong>Orchestration:</strong> LangChain Framework<br>
                <strong>Análises Avançadas:</strong> Scikit-learn<br>
                <strong>Data Processing:</strong> Pandas + NumPy<br>
                <strong>Visualization:</strong> Matplotlib + Seaborn<br>
                <strong>Statistics:</strong> SciPy.stats<br>
                <strong>Interface:</strong> Streamlit Framework<br>
                <strong>Algorithms:</strong> K-Means, DBSCAN, Random Forest, SVM, KNN
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Comandos disponíveis
        with st.expander("Comandos Disponíveis", expanded=False):
            st.markdown("""
            **Visualização de Dados:**
            - "Gere histograma da coluna [nome]" / "Histograma agrupado por [categoria]"
            - "Crie boxplot para detectar outliers em [coluna]"
            - "Gráfico de dispersão entre [col1] e [col2]"
            - "Matriz de correlação com heatmap"
            
            **Análise Estatística:**
            - "Teste normalidade da coluna [coluna]" (Shapiro-Wilk, Anderson-Darling)
            - "Teste t de uma amostra para [coluna]"
            - "Teste t independente entre [grupo1] e [grupo2]"
            - "Teste Mann-Whitney para [coluna] entre grupos"
            - "Teste qui-quadrado de independência"
            - "Análise de correlação entre [col1] e [col2]"
            
            **Análises Avançadas:**
            - "Realize clustering K-means com [n] clusters"
            - "Aplique DBSCAN para detecção de outliers"
            - "Otimize número de clusters automaticamente"
            - "Análise detalhada dos clusters formados"
            - "Compare clusters pela variável [coluna]"
            
            **Exploração de Dados:**
            - "Descreva estrutura completa do dataset"
            - "Análise de valores ausentes e duplicados"
            - "Estatísticas descritivas detalhadas"
            - "Detecção automática de padrões anômalos"
            """, help="Comandos em linguagem natural para análise exploratória avançada")


# ========== SELEÇÃO DE MODELO LLM ========== 
llm_model = st.sidebar.selectbox(
    'Selecione o modelo de IA (LLM) para análise:',
    [
        'gemini-2.5-flash',
        'gemini-1.5-pro',
        'openai-gpt-4',
        'openai-gpt-3.5-turbo'
    ],
    index=0,
    help='Escolha o modelo de linguagem para as respostas do agente.'
)
st.session_state['llm_model'] = llm_model


def main():
    """Função principal da aplicação."""
    configurar_pagina()
    inicializar_sessao()
    
    # Cabeçalho elegante e refinado
    st.markdown('<h1 class="main-header">Plataforma de Análise Exploratória Inteligente</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center;"><span class="precision-badge">Neural Analytics</span></div>', 
                unsafe_allow_html=True)
    
    st.markdown('<p class="subtitle">Análise Exploratória de Dados powered by IA</p>', 
                unsafe_allow_html=True)
    
    st.markdown('<p class="subtitle">Desenvolvido por Igor Töebe Lopes Farias</p>', 
                unsafe_allow_html=True)
    
    # Status do sistema
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">IA</div>
            <div class="metric-label">Agente</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">ML</div>
            <div class="metric-label">Análise</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">NLP</div>
            <div class="metric-label">Interface</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">STATS</div>
            <div class="metric-label">Estatística</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    sidebar_informacoes()
    
    # Conteúdo principal
    if st.session_state.df_carregado is None:
        # Página de upload
        carregar_arquivo()
    else:
        # Interface principal com dados carregados
        
        # Espaço acima das abas de navegação
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Overview", "Data Explorer", "Visualizations"])
        
        with tab1:
            interface_chat()
        
        with tab2:
            if st.session_state.info_arquivo:
                mostrar_informacoes_arquivo(st.session_state.info_arquivo)
        
        with tab3:
            st.markdown("""
            <div class="modern-card">
                <h2 style="margin-top: 0; color: var(--text-primary);">Explorador de Dados</h2>
                <p style="color: var(--text-secondary); margin-bottom: 0;">
                    Interactive dataset examination and statistical overview
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Controles de visualização
            col1, col2 = st.columns([2, 1])
            with col1:
                num_linhas = st.selectbox("Records to display:", [5, 10, 20, 50, 100], index=1)
            with col2:
                st.metric("Total Records", f"{len(st.session_state.df_carregado):,}")
            
            # Mostrar dados
            df_display = converter_para_arrow_compativel(st.session_state.df_carregado.head(num_linhas))
            st.dataframe(df_display, use_container_width=True)
            
            # Informações analíticas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="modern-card">
                    <h3 style="margin-top: 0; color: var(--text-primary);">Resumo Estatístico</h3>
                </div>
                """, unsafe_allow_html=True)
                # Converter describe() para tipos compatíveis com Arrow
                stats_df = converter_para_arrow_compativel(st.session_state.df_carregado.describe())
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="modern-card">
                    <h3 style="margin-top: 0; color: var(--text-primary);">Informações das Colunas</h3>
                </div>
                """, unsafe_allow_html=True)
                info_colunas = pd.DataFrame({
                    'Tipo de Dados': st.session_state.df_carregado.dtypes.astype(str),
                    'Contagem de Nulos': st.session_state.df_carregado.isnull().sum().astype(int),
                    'Valores Únicos': st.session_state.df_carregado.nunique().astype(int)
                })
                # Converter todas as colunas para tipos compatíveis com Arrow
                info_colunas = info_colunas.astype(str)
                st.dataframe(info_colunas, use_container_width=True)

        with tab4:
            st.markdown("""
            <div class="modern-card">
                <h2 style="margin-top: 0; color: var(--text-primary);">Galeria de Visualizações</h2>
                <p style="color: var(--text-secondary); margin-bottom: 0;">
                    Gráficos gerados e visualizações estatísticas
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            graficos_dir = Path("temp_graficos")
            arquivados_dir = graficos_dir / "arquivados"

            colA, colB = st.columns(2)
            with colA:
                st.markdown("""
                <div class="status-card">
                    <h4 style="color: var(--text-primary); margin-top: 0;">Visualizações Recentes</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if graficos_dir.exists():
                    arquivos = sorted(graficos_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if arquivos:
                        for p in arquivos[:15]:  # Reduzir para performance
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.image(str(p), use_column_width=True)
                            st.markdown(f"<div style='color: var(--text-secondary); font-size:0.95rem; margin-top:0.4rem;'>" + p.name + "</div>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="notification-warning">
                            <strong>Nenhum Gráfico Recente</strong><br>
                            Gere visualizações usando a aba Análise com IA
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="notification-warning">
                        <strong>Diretório de Gráficos Não Encontrado</strong><br>
                        Os gráficos aparecerão aqui após a primeira geração
                    </div>
                    """, unsafe_allow_html=True)

            with colB:
                st.markdown("""
                <div class="status-card">
                    <h4 style="color: var(--text-primary); margin-top: 0;">Coleções Arquivadas</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if arquivados_dir.exists():
                    # Listar pastas por data
                    datas = sorted([d for d in arquivados_dir.iterdir() if d.is_dir()], reverse=True)
                    if datas:
                        data_sel = st.selectbox("Selecione a data do arquivo:", [d.name for d in datas])
                        if data_sel:
                            pasta = arquivados_dir / data_sel
                            arquivos = sorted(pasta.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if arquivos:
                                for p in arquivos[:10]:  # Limitar para performance
                                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                    st.image(str(p), use_column_width=True)
                                    st.markdown(f"<div style='color: var(--text-secondary); font-size:0.95rem; margin-top:0.4rem;'>" + p.name + "</div>", unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="notification-warning">
                                    <strong>Nenhum Gráfico Encontrado</strong><br>
                                    Esta coleção de arquivo está vazia
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="notification-warning">
                            <strong>Nenhum Arquivo Disponível</strong><br>
                            Use 'Arquivar Gráficos Atuais' para criar coleções
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="notification-warning">
                        <strong>Sistema de Arquivo Não Inicializado</strong><br>
                        Os arquivos aparecerão após o primeiro uso
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
    # Professional footer
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">Neural Analytics AI Platform</div>
        <div class="footer-text">Sistema de Análise e Visualização Inteligente de Dados criado por Igor Töebe</div>
    </div>
    """, unsafe_allow_html=True)