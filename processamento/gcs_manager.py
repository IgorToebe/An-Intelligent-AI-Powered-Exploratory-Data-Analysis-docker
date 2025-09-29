"""
Google Cloud Storage Manager - Versão Cloud Run
Especificamente otimizada para contornar o limite de 32MB do Cloud Run
usando APENAS signed URLs para arquivos grandes (150MB+)
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

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
    """Gerenciador otimizado para Cloud Run - apenas arquivos grandes via GCS."""
    
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
        return GCS_AVAILABLE and self.client is not None
    
    def generate_signed_upload_url(self, filename: str) -> Optional[Dict[str, Any]]:
        """Gera signed URL para upload direto ao GCS."""
        if not self.is_available():
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"uploads/{timestamp}_{filename}"
            blob = self.bucket.blob(blob_name)
            
            # URL válida por 1 hora
            expiration = datetime.now() + timedelta(hours=1)
            
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="PUT",
                content_type="text/csv"
            )
            
            return {
                'signed_url': signed_url,
                'blob_name': blob_name,
                'expires_at': expiration.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar signed URL: {e}")
            return None
    
    def download_file_as_dataframe(self, blob_name: str) -> Optional[pd.DataFrame]:
        """Baixa arquivo do GCS e converte para DataFrame."""
        if not self.is_available():
            return None
            
        try:
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                return None
            
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                try:
                    df = pd.read_csv(tmp_file.name)
                    return df
                finally:
                    os.unlink(tmp_file.name)
                    
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
        except:
            return False


def create_streamlit_file_uploader_with_gcs():
    """
    Interface Cloud Run - APENAS signed URLs para arquivos 150MB+
    Elimina completamente st.file_uploader para evitar erro 413
    """
    gcs_manager = GCSManager()
    
    st.subheader("📂 Upload de Arquivo CSV (150MB+)")
    
    if not gcs_manager.is_available():
        st.error("❌ Google Cloud Storage não configurado")
        st.code("Configure: GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9")
        return None, None
    
    # MÉTODO ÚNICO: Signed URL do GCS
    st.success("☁️ Google Cloud Run - Upload via Signed URL")
    st.info("💡 Método otimizado para arquivos grandes (150MB+)")
    
    # Gerar signed URL
    filename = st.text_input(
        "Nome do arquivo CSV:",
        placeholder="meus_dados.csv",
        help="Digite o nome do arquivo que você vai enviar"
    )
    
    if filename and st.button("🔗 Gerar Link de Upload"):
        upload_info = gcs_manager.generate_signed_upload_url(filename)
        
        if upload_info:
            st.success("✅ Link de upload gerado!")
            
            # Instruções claras
            st.markdown("### 📋 Instruções de Upload:")
            st.markdown("1. **Copie o link abaixo**")
            st.code(upload_info['signed_url'])
            
            st.markdown("2. **Execute o comando no terminal:**")
            st.code(f'curl -X PUT -H "Content-Type: text/csv" --data-binary @{filename} "{upload_info["signed_url"]}"')
            
            st.markdown("3. **Ou use PowerShell:**")
            st.code(f'Invoke-RestMethod -Uri "{upload_info["signed_url"]}" -Method Put -InFile "{filename}" -ContentType "text/csv"')
            
            st.markdown("4. **Após o upload, digite o nome do blob:**")
            
            # Armazenar info no session state
            st.session_state.last_blob_name = upload_info['blob_name']
            st.session_state.upload_filename = filename
    
    # Processar arquivo após upload
    if 'last_blob_name' in st.session_state:
        st.markdown("### 📥 Processar Arquivo Enviado")
        
        blob_name = st.text_input(
            "Nome do blob no GCS:",
            value=st.session_state.last_blob_name,
            help="Nome gerado automaticamente após o upload"
        )
        
        if st.button("📊 Carregar e Processar Dados"):
            with st.spinner("Baixando e processando arquivo..."):
                df = gcs_manager.download_file_as_dataframe(blob_name)
            
            if df is not None:
                # Limpar arquivo após carregar
                gcs_manager.delete_file(blob_name)
                
                # Limpar session state
                if 'last_blob_name' in st.session_state:
                    del st.session_state.last_blob_name
                if 'upload_filename' in st.session_state:
                    del st.session_state.upload_filename
                
                st.success(f"🎉 Arquivo processado: {len(df):,} linhas × {len(df.columns)} colunas")
                return df, blob_name
            else:
                st.error("❌ Erro ao processar arquivo ou arquivo não encontrado")
    
    # Alternativa: URL pública
    st.markdown("---")
    st.markdown("### 🌐 Alternativa: Arquivo via URL")
    st.info("Para arquivos já hospedados online")
    
    csv_url = st.text_input("URL do arquivo CSV:", placeholder="https://example.com/data.csv")
    
    if csv_url and st.button("📥 Baixar da URL"):
        try:
            with st.spinner("Baixando arquivo da URL..."):
                response = requests.get(csv_url, timeout=120)
                response.raise_for_status()
                
                # Verificar se é CSV
                content_type = response.headers.get('content-type', '')
                if 'csv' not in content_type and not csv_url.endswith('.csv'):
                    st.warning("⚠️ Arquivo pode não ser CSV válido")
                
                # Processar diretamente se pequeno, ou via GCS se grande
                size_mb = len(response.content) / (1024 * 1024)
                
                if size_mb > 30:  # > 30MB via GCS
                    st.info(f"📊 Arquivo grande ({size_mb:.1f} MB) - processando via GCS")
                    
                    filename = csv_url.split('/')[-1] or 'arquivo_url.csv'
                    
                    # Upload para GCS
                    upload_info = gcs_manager.generate_signed_upload_url(filename)
                    if upload_info:
                        # Upload direto via requests
                        upload_response = requests.put(
                            upload_info['signed_url'],
                            data=response.content,
                            headers={'Content-Type': 'text/csv'},
                            timeout=300
                        )
                        
                        if upload_response.status_code == 200:
                            df = gcs_manager.download_file_as_dataframe(upload_info['blob_name'])
                            if df is not None:
                                gcs_manager.delete_file(upload_info['blob_name'])
                                st.success(f"✅ Processado: {len(df)} linhas × {len(df.columns)} colunas")
                                return df, f"url_{filename}"
                else:
                    # Arquivo pequeno - processar diretamente
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    st.success(f"✅ Processado: {len(df)} linhas × {len(df.columns)} colunas")
                    return df, "url_small"
                    
        except Exception as e:
            st.error(f"❌ Erro ao processar URL: {e}")
    
    return None, None


def setup_gcs_environment():
    """Verifica configuração GCS para Cloud Run."""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    bucket_name = os.getenv('GCS_BUCKET_NAME', 'i2a2-eda-uploads')
    
    if os.getenv('K_SERVICE'):  # Cloud Run
        st.success("☁️ Google Cloud Run detectado")
        if project_id:
            st.info(f"🗂️ Projeto: {project_id}")
            st.info(f"🪣 Bucket: {bucket_name}")
            return True
        else:
            st.error("❌ GOOGLE_CLOUD_PROJECT não configurado")
            return False
    else:
        st.info("🖥️ Ambiente local detectado")
        return project_id is not None