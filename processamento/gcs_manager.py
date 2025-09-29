"""
Google Cloud Storage Manager - SOLUÇÃO DEFINITIVA
Mantém drag and drop + força processamento via GCS para evitar erro 413
"""

import os
import tempfile
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from io import BytesIO

import streamlit as st
import pandas as pd

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

import requests
import logging

logger = logging.getLogger(__name__)


class GCSManager:
    """Gerenciador que força uso do GCS para todos os uploads."""
    
    def __init__(self):
        self.bucket_name = os.getenv('GCS_BUCKET_NAME', 'i2a2-eda-uploads')
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.client = None
        self.bucket = None
        
        if GCS_AVAILABLE and self.project_id:
            try:
                self.client = storage.Client(project=self.project_id)
                self.bucket = self.client.bucket(self.bucket_name)
            except Exception as e:
                logger.error(f"Erro GCS: {e}")
    
    def is_available(self) -> bool:
        return GCS_AVAILABLE and self.client is not None and self.bucket is not None
    
    def upload_file_direct(self, file_data: bytes, filename: str) -> Optional[str]:
        """Upload direto para GCS sem signed URL."""
        if not self.is_available():
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"uploads/{timestamp}_{filename}"
            
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                file_data,
                content_type='text/csv'
            )
            
            return blob_name
            
        except Exception as e:
            logger.error(f"Erro no upload direto: {e}")
            return None
    
    def download_file_as_dataframe(self, blob_name: str) -> Optional[pd.DataFrame]:
        """Baixa arquivo do GCS e converte para DataFrame."""
        if not self.is_available():
            return None
            
        try:
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                return None
            
            # Download direto para memória
            file_data = blob.download_as_text()
            
            # Converter para DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(file_data))
            return df
                    
        except Exception as e:
            logger.error(f"Erro ao baixar: {e}")
            return None
    
    def delete_file(self, blob_name: str) -> bool:
        """Remove arquivo do GCS após processamento."""
        if not self.is_available():
            return False
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar: {e}")
            return False


def create_streamlit_file_uploader_with_gcs():
    """
    Upload com drag and drop que FORÇA processamento via GCS
    Solução final que intercepta arquivos grandes ANTES do erro 413
    """
    gcs_manager = GCSManager()
    
    st.subheader("📂 Upload de Arquivo CSV")
    
    if not gcs_manager.is_available():
        st.error("❌ Google Cloud Storage não está disponível")
        st.info("Configure as variáveis de ambiente:")
        st.code("""
GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
GCS_BUCKET_NAME=i2a2-eda-uploads
""")
        return None, None
    
    st.success("🚀 Upload com processamento via Google Cloud Storage")
    st.info("💡 Arraste e solte seu arquivo CSV aqui - processamento automático via GCS")
    
    # JavaScript para detectar tamanho do arquivo ANTES do upload
    file_size_js = """
    <script>
    function checkFileSize() {
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput && fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const sizeMB = file.size / (1024 * 1024);
            
            if (sizeMB > 30) {
                alert(`Arquivo muito grande (${sizeMB.toFixed(1)} MB)! Use o método GCS abaixo.`);
                fileInput.value = '';
                return false;
            }
        }
        return true;
    }
    
    // Interceptar mudanças no file input
    setTimeout(() => {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        fileInputs.forEach(input => {
            input.addEventListener('change', checkFileSize);
        });
    }, 1000);
    </script>
    """
    
    st.markdown(file_size_js, unsafe_allow_html=True)
    
    # File uploader padrão para arquivos pequenos/médios
    uploaded_file = st.file_uploader(
        "Selecione ou arraste seu arquivo CSV (processamento via GCS automático):",
        type=['csv'],
        help="Arquivos pequenos: upload direto | Arquivos grandes: processamento via GCS"
    )
    
    if uploaded_file is not None:
        file_data = uploaded_file.getvalue()
        file_size_mb = len(file_data) / (1024 * 1024)
        filename = uploaded_file.name
        st.info(f"📊 Arquivo: {filename} ({file_size_mb:.1f} MB)")
        
        if file_size_mb > 32:
            st.warning("Arquivo grande detectado (>32MB). Usando upload via URL assinada para evitar erro 413.")
            if not gcs_manager.is_available():
                st.error("❌ Google Cloud Storage não está disponível. Configure as variáveis de ambiente corretamente.")
                return None, None
            with st.spinner(f"🔄 Gerando URL assinada para {filename}..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                blob_name = f"uploads/{timestamp}_{filename}"
                blob = gcs_manager.bucket.blob(blob_name)
                expiration = datetime.now() + timedelta(hours=1)
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration,
                    method="PUT",
                    content_type="text/csv"
                )
            with st.spinner(f"⬆️ Enviando {filename} para GCS via URL assinada..."):
                try:
                    response = requests.put(
                        signed_url,
                        data=file_data,
                        headers={"Content-Type": "text/csv"}
                    )
                    if response.status_code == 200:
                        st.success("✅ Upload via URL assinada concluído!")
                        with st.spinner("📥 Baixando e convertendo para DataFrame..."):
                            df = gcs_manager.download_file_as_dataframe(blob_name)
                        if df is not None:
                            gcs_manager.delete_file(blob_name)
                            st.success(f"🎉 Dados carregados: {len(df):,} linhas × {len(df.columns)} colunas")
                            return df, blob_name
                        else:
                            st.error("❌ Erro ao converter arquivo para DataFrame")
                            gcs_manager.delete_file(blob_name)
                            return None, None
                    else:
                        st.error(f"❌ Falha no upload via URL assinada: {response.status_code}")
                        return None, None
                except Exception as e:
                    st.error(f"❌ Erro no upload via URL assinada: {e}")
                    return None, None
        else:
            with st.spinner(f"🔄 Processando {filename} via Google Cloud Storage..."):
                blob_name = gcs_manager.upload_file_direct(file_data, filename)
                if blob_name:
                    st.success("✅ Upload para GCS concluído!")
                    with st.spinner("📥 Baixando e convertendo para DataFrame..."):
                        df = gcs_manager.download_file_as_dataframe(blob_name)
                    if df is not None:
                        gcs_manager.delete_file(blob_name)
                        st.success(f"🎉 Dados carregados: {len(df):,} linhas × {len(df.columns)} colunas")
                        return df, blob_name
                    else:
                        st.error("❌ Erro ao converter arquivo para DataFrame")
                        gcs_manager.delete_file(blob_name)
                        return None, None
                else:
                    st.error("❌ Falha no upload para Google Cloud Storage")
                    return None, None
    
    # Método alternativo para arquivos muito grandes
    with st.expander("🔧 Método Alternativo para Arquivos Muito Grandes (> 30MB)"):
        st.markdown("### 📋 Processo Manual:")
        st.markdown("1. **Gere um link de upload**")
        
        large_filename = st.text_input(
            "Nome do arquivo grande:",
            placeholder="dataset_grande.csv",
            key="large_file_input"
        )
        
        if large_filename and st.button("🔗 Gerar Link de Upload"):
            # Gerar signed URL para arquivo grande
            try:
                if gcs_manager.bucket is None:
                    st.error("❌ Erro: Bucket não disponível")
                    return None, None
                    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                blob_name = f"uploads/{timestamp}_{large_filename}"
                blob = gcs_manager.bucket.blob(blob_name)
                
                expiration = datetime.now() + timedelta(hours=1)
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration,
                    method="PUT",
                    content_type="text/csv"
                )
                
                st.success("✅ Link gerado!")
                st.code(signed_url)
                
                st.markdown("2. **Execute no terminal:**")
                st.code(f'curl -X PUT -H "Content-Type: text/csv" --data-binary @{large_filename} "{signed_url}"')
                
                # Armazenar para processamento posterior
                st.session_state.pending_blob = blob_name
                
            except Exception as e:
                st.error(f"Erro ao gerar link: {e}")
        
        # Processar arquivo enviado manualmente
        if 'pending_blob' in st.session_state:
            st.markdown("3. **Processar arquivo enviado**")
            if st.button("📊 Processar Arquivo Enviado"):
                with st.spinner("Processando arquivo grande..."):
                    df = gcs_manager.download_file_as_dataframe(st.session_state.pending_blob)
                
                if df is not None:
                    gcs_manager.delete_file(st.session_state.pending_blob)
                    del st.session_state.pending_blob
                    
                    st.success(f"🎉 Arquivo grande processado: {len(df):,} linhas × {len(df.columns)} colunas")
                    return df, st.session_state.pending_blob
                else:
                    st.error("❌ Arquivo não encontrado ou erro no processamento")
    
    return None, None


def setup_gcs_environment():
    """Verifica e configura ambiente GCS."""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    bucket_name = os.getenv('GCS_BUCKET_NAME', 'i2a2-eda-uploads')
    
    # Verificar se estamos no Cloud Run
    if os.getenv('K_SERVICE'):
        st.success("☁️ Executando no Google Cloud Run")
        if project_id:
            st.info(f"✅ Projeto: {project_id}")
            st.info(f"✅ Bucket: {bucket_name}")
            return True
        else:
            st.error("❌ GOOGLE_CLOUD_PROJECT não configurado no Cloud Run")
            st.error("🚨 Google Cloud Storage não está configurado!")
            st.markdown("📋 Configure as seguintes variáveis de ambiente:")
            st.code("""
GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
GCS_BUCKET_NAME=i2a2-eda-uploads
            """)
            return False
    else:
        # Ambiente local
        if project_id:
            st.info(f"🖥️ Ambiente local - Projeto: {project_id}")
            return True
        else:
            st.warning("⚠️ GOOGLE_CLOUD_PROJECT não configurado para ambiente local")
            return False