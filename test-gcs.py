#!/usr/bin/env python3
"""
Script de teste para verificar configuração do Google Cloud Storage
Execute este script para testar se o GCS está funcionando corretamente.
"""

import os
import sys
import tempfile
from pathlib import Path

# Adicionar o diretório do projeto ao path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

try:
    from processamento.gcs_manager import GCSManager
    print("✅ Importação do GCSManager bem-sucedida")
except ImportError as e:
    print(f"❌ Erro ao importar GCSManager: {e}")
    sys.exit(1)

def test_gcs_configuration():
    """Testa a configuração do GCS"""
    
    print("🧪 Testando configuração do Google Cloud Storage...")
    print("-" * 50)
    
    # Verificar variáveis de ambiente
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    bucket_name = os.getenv('GCS_BUCKET_NAME', 'i2a2-eda-uploads')
    
    print(f"📋 Project ID: {project_id or 'NÃO CONFIGURADO'}")
    print(f"🪣 Bucket Name: {bucket_name}")
    print(f"☁️ Cloud Run: {'SIM' if os.getenv('K_SERVICE') else 'NÃO'}")
    print()
    
    # Inicializar GCS Manager
    try:
        gcs_manager = GCSManager(bucket_name=bucket_name, project_id=project_id)
        print("✅ GCSManager inicializado")
    except Exception as e:
        print(f"❌ Erro ao inicializar GCSManager: {e}")
        return False
    
    # Verificar disponibilidade
    if not gcs_manager.is_available():
        print("❌ GCS não está disponível")
        print("💡 Possíveis soluções:")
        print("   1. Execute: gcloud auth application-default login")
        print("   2. Configure: GOOGLE_CLOUD_PROJECT=groovy-rope-471520-c9")
        print("   3. Configure: GCS_BUCKET_NAME=i2a2-eda-uploads")
        return False
    
    print("✅ GCS está disponível")
    
    # Testar geração de signed URL
    try:
        upload_info = gcs_manager.generate_signed_upload_url("test-file.csv")
        if upload_info:
            print("✅ Signed URL gerada com sucesso")
            print(f"🔗 URL expires at: {upload_info['expires_at']}")
        else:
            print("❌ Falha ao gerar signed URL")
            return False
    except Exception as e:
        print(f"❌ Erro ao gerar signed URL: {e}")
        return False
    
    # Testar upload pequeno
    try:
        print("\n🧪 Testando upload de arquivo pequeno...")
        test_data = b"col1,col2,col3\n1,2,3\n4,5,6\n"
        
        # Simular upload
        import requests
        response = requests.put(
            upload_info['signed_url'],
            data=test_data,
            headers=upload_info['upload_headers'],
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Upload de teste bem-sucedido")
            
            # Testar download
            df = gcs_manager.download_file_as_dataframe(upload_info['blob_name'])
            if df is not None:
                print(f"✅ Download bem-sucedido: {len(df)} linhas")
                
                # Limpar arquivo de teste
                gcs_manager.delete_file(upload_info['blob_name'])
                print("✅ Arquivo de teste removido")
            else:
                print("❌ Falha no download")
                return False
        else:
            print(f"❌ Falha no upload: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste de upload: {e}")
        return False
    
    print("\n🎉 Todos os testes passaram! GCS está funcionando corretamente.")
    return True

def main():
    """Função principal"""
    
    print("🔍 Teste de Configuração do Google Cloud Storage")
    print("=" * 60)
    
    # Verificar se estamos no diretório correto
    if not Path("app.py").exists():
        print("❌ Execute este script no diretório raiz do projeto")
        sys.exit(1)
    
    # Executar testes
    success = test_gcs_configuration()
    
    if success:
        print("\n✅ SUCESSO: GCS está configurado e funcionando!")
        print("🚀 Sua aplicação pode processar arquivos grandes via GCS")
    else:
        print("\n❌ FALHA: GCS não está funcionando corretamente")
        print("📋 Verifique a documentação em GCS-LARGE-FILES.md")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()