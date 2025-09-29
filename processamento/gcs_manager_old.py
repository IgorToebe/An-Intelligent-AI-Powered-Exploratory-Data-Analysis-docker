"""
Google Cloud Storage Manager para Upload de Arquivos Grandes

Este módulo gerencia uploads de arquivos grandes (150MB+) usando signed URLs
para contornar o limite de 32MB do Cloud Run.
"""

import os
import json
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import pandas as pd

try:
    from google.cloud import storage
    from google.auth import default
    import google.auth.exceptions
    GCS_AVAILABLE = True
    StorageClient = storage.Client
    Bucket = storage.Bucket
    Blob = storage.Blob
except ImportError:
    GCS_AVAILABLE = False
    storage = None
    StorageClient = None
    Bucket = None
    Blob = None

import requests
import logging

logger = logging.getLogger(__name__)


class GCSManager:
    """Gerenciador de arquivos no Google Cloud Storage."""
    
    def __init__(self, bucket_name: Optional[str] = None, project_id: Optional[str] = None):
        """
        Inicializa o gerenciador GCS.
        
        Args:
            bucket_name: Nome do bucket GCS
            project_id: ID do projeto Google Cloud
        """
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'i2a2-eda-uploads')
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.client = None  # type: Optional[Any]
        self.bucket = None  # type: Optional[Any]
        
        # Configurações de upload
        self.upload_timeout = 300  # 5 minutos
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializa o cliente GCS."""
        if not GCS_AVAILABLE:
            logger.warning("Google Cloud Storage não disponível")
            return False
            
        try:
            # Tentar autenticação automática (funciona no Cloud Run)
            if GCS_AVAILABLE and storage:
                self.client = storage.Client(project=self.project_id)
                self.bucket = self.client.bucket(self.bucket_name)
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar GCS: {e}")
            return False
    
    def is_available(self) -> bool:
        """Verifica se o GCS está disponível e configurado."""
        return GCS_AVAILABLE and self.client is not None
    
    def generate_signed_upload_url(
        self, 
        filename: str, 
        content_type: str = "text/csv",
        expiration_minutes: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Gera uma signed URL para upload direto ao GCS.
        
        Args:
            filename: Nome do arquivo
            content_type: Tipo MIME do arquivo
            expiration_minutes: Minutos até expirar a URL
            
        Returns:
            Dict com informações da signed URL ou None se erro
        """
        if not self.is_available() or not self.bucket:
            return None
            
        try:
            # Gerar nome único para o arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"uploads/{timestamp}_{filename}"
            
            blob = self.bucket.blob(blob_name)
            
            # Configurar metadados
            blob.metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'original_filename': filename,
                'content_type': content_type
            }
            
            # Gerar signed URL para upload
            expiration = datetime.now() + timedelta(minutes=expiration_minutes)
            
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="PUT",
                content_type=content_type,
                headers={'x-goog-content-length-range': '0,209715200'}  # Max 200MB
            )
            
            return {
                'signed_url': signed_url,
                'blob_name': blob_name,
                'bucket_name': self.bucket_name,
                'expires_at': expiration.isoformat(),
                'content_type': content_type,
                'upload_headers': {
                    'Content-Type': content_type,
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar signed URL: {e}")
            return None
    
    def download_file_as_dataframe(self, blob_name: str) -> Optional[pd.DataFrame]:
        """
        Baixa um arquivo do GCS e converte para DataFrame.
        
        Args:
            blob_name: Nome do blob no GCS
            
        Returns:
            DataFrame do pandas ou None se erro
        """
        if not self.is_available() or not self.bucket:
            return None
            
        try:
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                logger.error(f"Arquivo {blob_name} não encontrado no GCS")
                return None
            
            # Baixar para arquivo temporário
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                
                # Ler como DataFrame
                try:
                    df = pd.read_csv(tmp_file.name)
                    return df
                except Exception as e:
                    logger.error(f"Erro ao ler CSV: {e}")
                    return None
                finally:
                    # Limpar arquivo temporário
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            logger.error(f"Erro ao baixar arquivo do GCS: {e}")
            return None
    
    def delete_file(self, blob_name: str) -> bool:
        """
        Deleta um arquivo do GCS.
        
        Args:
            blob_name: Nome do blob no GCS
            
        Returns:
            True se sucesso, False se erro
        """
        if not self.is_available() or not self.bucket:
            return False
            
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar arquivo: {e}")
            return False
    
    def list_user_files(self, prefix: str = "uploads/") -> list:
        """
        Lista arquivos do usuário no GCS.
        
        Args:
            prefix: Prefixo para filtrar arquivos
            
        Returns:
            Lista de informações dos arquivos
        """
        if not self.is_available() or not self.client:
            return []
            
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            
            files = []
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created,
                    'updated': blob.updated,
                    'content_type': blob.content_type,
                    'metadata': blob.metadata or {}
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Erro ao listar arquivos: {e}")
            return []


def upload_large_file_to_gcs(file_data: bytes, filename: str, gcs_manager: GCSManager) -> Optional[str]:
    """
    Upload de arquivo grande usando signed URL com streaming.
    
    Args:
        file_data: Dados do arquivo em bytes
        filename: Nome do arquivo
        gcs_manager: Instância do GCSManager
        
    Returns:
        Nome do blob no GCS ou None se erro
    """
    try:
        # Verificar tamanho
        file_size_mb = len(file_data) / (1024 * 1024)
        logger.info(f"Iniciando upload de {filename} ({file_size_mb:.1f} MB)")
        
        # Gerar signed URL
        upload_info = gcs_manager.generate_signed_upload_url(filename)
        if not upload_info:
            st.error("❌ Erro ao gerar URL de upload")
            return None
        
        # Fazer upload usando método mais robusto
        headers = upload_info['upload_headers']
        
        # Adicionar headers específicos para evitar erro 413
        headers.update({
            'Content-Length': str(len(file_data)),
            'X-Goog-Content-Length-Range': f'0,{len(file_data)}'
        })
        
        # Upload direto sem streaming para simplificar
        st.info(f"🔄 Enviando {file_size_mb:.1f} MB para GCS...")
        
        response = requests.put(
            upload_info['signed_url'],
            data=file_data,
            headers=headers,
            timeout=900  # 15 minutos timeout
        )
        
        logger.info(f"Upload response: {response.status_code}")
        
        if response.status_code == 200:
            st.success(f"✅ Arquivo {filename} enviado com sucesso!")
            logger.info(f"Upload successful: {upload_info['blob_name']}")
            return upload_info['blob_name']
        elif response.status_code == 413:
            st.error("❌ Arquivo muito grande. Verifique configuração do servidor.")
            logger.error(f"Erro 413 - Payload too large para {filename}")
            return None
        else:
            st.error(f"❌ Erro no upload: {response.status_code} - {response.text}")
            logger.error(f"Upload error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Erro durante upload: {e}")
        logger.error(f"Erro no upload para GCS: {e}")
        return None


def create_streamlit_file_uploader_with_gcs():
    """
    Versão que intercepta arquivos ANTES do processamento do Streamlit.
    Usa JavaScript para detectar o tamanho e redirecionar para GCS imediatamente.
    """
    import streamlit.components.v1 as components
    
    # Inicializar GCS Manager
    gcs_manager = GCSManager()
    
    st.subheader("📂 Upload de Arquivo CSV")
    
    if not gcs_manager.is_available():
        st.error("❌ Google Cloud Storage é obrigatório para esta aplicação")
        st.info("📋 Configure as variáveis de ambiente:")
        st.code("""
GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
GCS_BUCKET_NAME=i2a2-eda-uploads
""")
        return None, None
    
    st.success("🚀 Upload via Google Cloud Storage (interceptação automática)")
    
    # Interface JavaScript para interceptar arquivos grandes
    upload_component = f"""
    <div id="file-upload-container">
        <div style="border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 10px 0;">
            <p>� Arraste e solte seu arquivo CSV aqui</p>
            <input type="file" id="file-input" accept=".csv" style="margin: 10px;"/>
            <div id="upload-status" style="margin-top: 10px;"></div>
        </div>
    </div>
    
    <script>
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const container = document.getElementById('file-upload-container');
    
    // Handle drag and drop
    container.addEventListener('dragover', (e) => {{
        e.preventDefault();
        container.style.backgroundColor = '#f0f8ff';
    }});
    
    container.addEventListener('dragleave', (e) => {{
        e.preventDefault();
        container.style.backgroundColor = '';
    }});
    
    container.addEventListener('drop', (e) => {{
        e.preventDefault();
        container.style.backgroundColor = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {{
            handleFile(files[0]);
        }}
    }});
    
    fileInput.addEventListener('change', (e) => {{
        if (e.target.files.length > 0) {{
            handleFile(e.target.files[0]);
        }}
    }});
    
    function handleFile(file) {{
        const sizeMB = file.size / (1024 * 1024);
        uploadStatus.innerHTML = `
            <p>📊 Arquivo: ${{file.name}} (${{sizeMB.toFixed(1)}} MB)</p>
            <p>🔄 Preparando upload via GCS...</p>
        `;
        
        // Sempre usar GCS para evitar erro 413
        if (sizeMB > 0) {{
            uploadStatus.innerHTML += '<p>☁️ Enviando via Google Cloud Storage...</p>';
            
            // Converter para base64 e enviar para Streamlit
            const reader = new FileReader();
            reader.onload = function(e) {{
                const base64Data = e.target.result.split(',')[1];
                
                // Armazenar no session state via JavaScript
                window.parent.postMessage({{
                    type: 'file_upload',
                    filename: file.name,
                    size: file.size,
                    data: base64Data
                }}, '*');
                
                uploadStatus.innerHTML = '<p>✅ Arquivo preparado para processamento GCS!</p>';
            }};
            reader.readAsDataURL(file);
        }}
    }}
    </script>
    """
    
    # Renderizar componente
    components.html(upload_component, height=200)
    
    # Verificar se há dados de arquivo no session state via JavaScript callback
    if 'uploaded_file_data' in st.session_state:
        file_info = st.session_state.uploaded_file_data
        
        st.info(f"📊 Processando: {file_info['filename']} ({file_info['size_mb']:.1f} MB)")
        
        with st.spinner("Enviando para GCS..."):
            import base64
            file_data = base64.b64decode(file_info['data'])
            
            blob_name = upload_large_file_to_gcs(
                file_data,
                file_info['filename'],
                gcs_manager
            )
        
        if blob_name:
            with st.spinner("Baixando e processando do GCS..."):
                df = gcs_manager.download_file_as_dataframe(blob_name)
            
            if df is not None:
                gcs_manager.delete_file(blob_name)
                st.success(f"🎉 Dados carregados: {len(df):,} linhas × {len(df.columns)} colunas")
                
                # Limpar session state
                del st.session_state.uploaded_file_data
                return df, blob_name
            else:
                st.error("❌ Erro ao processar dados")
                del st.session_state.uploaded_file_data
        else:
            st.error("❌ Falha no upload")
            del st.session_state.uploaded_file_data
    
    # Fallback: file_uploader tradicional apenas para arquivos MUITO pequenos
    st.markdown("---")
    st.markdown("### 📄 Alternativa para Arquivos Pequenos (< 1MB)")
    
    uploaded_file = st.file_uploader(
        "Upload tradicional (apenas arquivos pequenos):",
        type=['csv'],
        help="Para arquivos > 1MB, use o método acima"
    )
    
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb < 1:
            st.info(f"📊 Arquivo pequeno: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            # Mesmo arquivos pequenos vão para GCS para consistência
            with st.spinner("Enviando arquivo pequeno para GCS..."):
                blob_name = upload_large_file_to_gcs(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    gcs_manager
                )
            
            if blob_name:
                with st.spinner("Processando..."):
                    df = gcs_manager.download_file_as_dataframe(blob_name)
                
                if df is not None:
                    gcs_manager.delete_file(blob_name)
                    st.success(f"✅ Dados carregados: {len(df)} linhas × {len(df.columns)} colunas")
                    return df, blob_name
        else:
            st.warning(f"⚠️ Arquivo muito grande ({file_size_mb:.1f} MB). Use o método de drag and drop acima.")
    
    return None, None


# Configurações de ambiente para GCS
def setup_gcs_environment():
    """Configura e verifica variáveis de ambiente necessárias para GCS."""
    
    # Verificar variáveis de ambiente obrigatórias
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    bucket_name = os.getenv('GCS_BUCKET_NAME', 'i2a2-eda-uploads')
    
    # Tentar obter do secrets.toml se não estiver nas env vars
    if not project_id:
        try:
            project_id = st.secrets.get("GOOGLE_CLOUD_PROJECT")
            bucket_name = st.secrets.get("GCS_BUCKET_NAME", "i2a2-eda-uploads")
            
            if project_id:
                st.info(f"📋 Usando configuração do secrets.toml: {project_id}")
            
        except Exception:
            pass
    
    # Verificar se estamos no Cloud Run
    if os.getenv('K_SERVICE'):  # Variável presente no Cloud Run
        st.success("☁️ Executando no Google Cloud Run - autenticação automática")
        if not project_id:
            st.error("❌ GOOGLE_CLOUD_PROJECT não configurado no Cloud Run")
            st.error("🚨 Google Cloud Storage não está configurado!")
            st.markdown("📋 Configure as seguintes variáveis de ambiente:")
            st.code("""
GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
GCS_BUCKET_NAME=i2a2-eda-uploads
        """)
            return False
        else:
            st.info(f"🗂️ Projeto: {project_id}")
            st.info(f"🪣 Bucket: {bucket_name}")
            return True
    
    # Para desenvolvimento local
    if not project_id:
        st.warning("""
        ⚠️ Configuração GCS faltando para desenvolvimento local:
        
        **Opção 1: Variáveis de ambiente**
        ```
        set GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9
        set GCS_BUCKET_NAME=i2a2-eda-uploads
        ```
        
        **Opção 2: Autenticação manual**
        ```
        gcloud auth application-default login
        gcloud config set project groovy-rope-471520-c9
        ```
        """)
        return False
    
    # Verificar credenciais
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        # Tentar usar credenciais padrão
        try:
            from google.auth import default
            credentials, _ = default()
            st.success("✅ Credenciais Google Cloud encontradas")
            return True
        except Exception as e:
            st.warning(f"⚠️ Problema com credenciais: {e}")
            st.info("Execute: `gcloud auth application-default login`")
            return False
    
    return True