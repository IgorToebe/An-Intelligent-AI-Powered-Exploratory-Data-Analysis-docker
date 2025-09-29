"""
Alternativa ao file_uploader para evitar erro 413 no Cloud Run
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import requests
from io import StringIO
from processamento.gcs_manager import GCSManager


def create_manual_file_uploader():
    """
    Cria uma interface manual de upload que evita completamente o st.file_uploader
    para arquivos grandes, usando signed URLs do GCS.
    """
    st.subheader("📂 Upload de Arquivo CSV - Versão Manual")
    
    # Inicializar GCS Manager
    gcs_manager = GCSManager()
    
    if not gcs_manager.is_available():
        st.error("❌ Google Cloud Storage não está configurado")
        return None, None
    
    st.success("🚀 Método Manual - Sem Limitações de Tamanho")
    
    # Opções de upload
    upload_method = st.radio(
        "Escolha o método de upload:",
        [
            "📤 Upload via URL do GCS (Recomendado para arquivos grandes)",
            "📄 Cole o conteúdo CSV diretamente",
            "🔗 Upload via URL externa"
        ]
    )
    
    if upload_method and upload_method.startswith("📤"):
        return _handle_gcs_signed_url_upload(gcs_manager)
    elif upload_method and upload_method.startswith("📄"):
        return _handle_text_paste_upload()
    elif upload_method and upload_method.startswith("🔗"):
        return _handle_url_upload()
    
    return None, None


def _handle_gcs_signed_url_upload(gcs_manager):
    """Gera signed URL para upload direto no GCS"""
    st.markdown("### 📤 Upload via Signed URL")
    
    filename = st.text_input("Nome do arquivo (ex: dados.csv):", value="dados.csv")
    
    if st.button("🔗 Gerar Link de Upload"):
        if filename:
            upload_info = gcs_manager.generate_signed_upload_url(filename)
            if upload_info:
                st.success("✅ Link de upload gerado!")
                st.markdown(f"**🔗 URL para upload:**")
                st.code(upload_info['signed_url'])
                
                st.markdown("**📋 Instruções:**")
                st.markdown("""
                1. Copie a URL acima
                2. Use um cliente HTTP (curl, Postman, etc.) para fazer upload:
                ```bash
                curl -X PUT -H "Content-Type: text/csv" --data-binary @seu_arquivo.csv "URL_COPIADA"
                ```
                3. Após o upload, digite o nome do blob abaixo para processar
                """)
                
                st.session_state.last_blob_name = upload_info['blob_name']
    
    # Verificar se arquivo foi enviado
    if 'last_blob_name' in st.session_state:
        blob_name = st.text_input(
            "Nome do blob no GCS:", 
            value=st.session_state.last_blob_name
        )
        
        if st.button("📥 Carregar arquivo do GCS"):
            with st.spinner("Baixando e processando..."):
                df = gcs_manager.download_file_as_dataframe(blob_name)
                if df is not None:
                    st.success(f"✅ Arquivo carregado: {len(df)} linhas")
                    return df, blob_name
                else:
                    st.error("❌ Erro ao carregar arquivo")
    
    return None, None


def _handle_text_paste_upload():
    """Permite colar conteúdo CSV diretamente"""
    st.markdown("### 📄 Cole o Conteúdo CSV")
    
    csv_content = st.text_area(
        "Cole o conteúdo do arquivo CSV aqui:",
        height=300,
        placeholder="col1,col2,col3\nvalor1,valor2,valor3\n..."
    )
    
    if csv_content and st.button("📊 Processar CSV"):
        try:
            df = pd.read_csv(StringIO(csv_content))
            st.success(f"✅ CSV processado: {len(df)} linhas × {len(df.columns)} colunas")
            return df, "manual_paste"
        except Exception as e:
            st.error(f"❌ Erro ao processar CSV: {e}")
    
    return None, None


def _handle_url_upload():
    """Permite upload via URL externa"""
    st.markdown("### 🔗 Upload via URL Externa")
    
    url = st.text_input("URL do arquivo CSV:")
    
    if url and st.button("📥 Baixar e Processar"):
        try:
            with st.spinner("Baixando arquivo..."):
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                df = pd.read_csv(StringIO(response.text))
                st.success(f"✅ Arquivo baixado: {len(df)} linhas × {len(df.columns)} colunas")
                return df, f"url_download_{hash(url)}"
                
        except Exception as e:
            st.error(f"❌ Erro ao baixar arquivo: {e}")
    
    return None, None


def create_hybrid_uploader():
    """
    Versão híbrida que tenta file_uploader pequeno primeiro,
    depois oferece alternativas para arquivos grandes
    """
    st.subheader("📂 Upload de Arquivo CSV")
    
    # Tentar upload pequeno primeiro (< 1MB)
    st.markdown("### 📤 Upload Direto (apenas arquivos pequenos < 1MB)")
    
    uploaded_file = st.file_uploader(
        "Arquivo CSV pequeno:",
        type=['csv'],
        help="Para arquivos > 1MB, use as opções abaixo"
    )
    
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb < 1:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado: {len(df)} linhas × {len(df.columns)} colunas")
                return df, f"small_upload_{uploaded_file.name}"
            except Exception as e:
                st.error(f"❌ Erro ao processar: {e}")
        else:
            st.warning(f"⚠️ Arquivo muito grande ({file_size_mb:.1f} MB). Use o método manual abaixo.")
    
    # Oferecer método manual para arquivos grandes
    st.markdown("---")
    st.markdown("### 🚀 Para Arquivos Grandes (> 1MB)")
    
    return create_manual_file_uploader()