"""
Módulo de Carregamento e Processamento de Dados
Responsável por carregar arquivos CSV de forma otimizada e eficiente.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os


def otimizar_tipos_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Otimiza os tipos de dados do DataFrame para reduzir uso de memória.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame com tipos otimizados
    """
    df_otimizado = df.copy()
    
    for coluna in df_otimizado.columns:
        tipo_original = df_otimizado[coluna].dtype
        
        # Tentar converter para numérico se possível
        if tipo_original == 'object':
            # Tentar datetime primeiro
            try:
                df_otimizado[coluna] = pd.to_datetime(df_otimizado[coluna])
                continue
            except:
                pass
                
            # Tentar numérico
            try:
                df_otimizado[coluna] = pd.to_numeric(df_otimizado[coluna], errors='coerce')
            except:
                pass
        
        # Otimizar tipos numéricos (usando tipos compatíveis com Arrow)
        if pd.api.types.is_integer_dtype(df_otimizado[coluna]):
            # Usar sempre int64 para compatibilidade com Arrow
            df_otimizado[coluna] = df_otimizado[coluna].astype(np.int64)
                
        elif pd.api.types.is_float_dtype(df_otimizado[coluna]):
            # Usar sempre float64 para compatibilidade com Arrow
            df_otimizado[coluna] = df_otimizado[coluna].astype(np.float64)
    
    return df_otimizado


def carregar_csv(caminho_arquivo: str, 
                chunksize: Optional[int] = None,
                otimizar_memoria: bool = True,
                encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Carrega um arquivo CSV de forma otimizada.
    
    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        chunksize: Tamanho do chunk para leitura em pedaços (None para carregar tudo)
        otimizar_memoria: Se deve otimizar tipos de dados para economizar memória
        encoding: Encoding do arquivo
        
    Returns:
        DataFrame com os dados carregados
        
    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o arquivo não puder ser lido
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    try:
        # Tentar diferentes encodings se UTF-8 falhar
        encodings_para_tentar = [encoding, 'latin-1', 'cp1252', 'iso-8859-1']
        
        df = None
        for enc in encodings_para_tentar:
            try:
                if chunksize:
                    # Leitura em chunks
                    chunks = []
                    for chunk in pd.read_csv(caminho_arquivo, 
                                           chunksize=chunksize, 
                                           encoding=enc):
                        if otimizar_memoria:
                            chunk = otimizar_tipos_dados(chunk)
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    # Leitura completa
                    df = pd.read_csv(caminho_arquivo, encoding=enc)
                
                print(f"Arquivo carregado com encoding: {enc}")
                break
                
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Não foi possível ler o arquivo com nenhum encoding testado")
        
        # Otimizar tipos se solicitado e não foi feito em chunks
        if otimizar_memoria and not chunksize:
            df = otimizar_tipos_dados(df)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Erro ao carregar arquivo CSV: {str(e)}")


def obter_info_arquivo(caminho_arquivo: str) -> Dict[str, Any]:
    """
    Obtém informações básicas sobre o arquivo CSV sem carregá-lo completamente.
    
    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        
    Returns:
        Dicionário com informações do arquivo
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    # Obter tamanho do arquivo
    tamanho_bytes = os.path.getsize(caminho_arquivo)
    tamanho_mb = tamanho_bytes / (1024 * 1024)
    
    # Ler apenas as primeiras linhas para obter informações sobre colunas
    try:
        amostra = pd.read_csv(caminho_arquivo, nrows=5)
        num_colunas = len(amostra.columns)
        nomes_colunas = list(amostra.columns)
        
        # Estimar número de linhas (aproximado)
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            primeira_linha = f.readline()
            tamanho_linha_media = len(primeira_linha)
        
        linhas_estimadas = tamanho_bytes // tamanho_linha_media
        
    except Exception as e:
        return {
            'erro': str(e),
            'tamanho_mb': tamanho_mb
        }
    
    return {
        'tamanho_mb': round(tamanho_mb, 2),
        'linhas_estimadas': linhas_estimadas,
        'num_colunas': num_colunas,
        'nomes_colunas': nomes_colunas,
        'tipos_amostra': dict(amostra.dtypes.astype(str))
    }


def recomendar_estrategia_carregamento(caminho_arquivo: str) -> Dict[str, Any]:
    """
    Recomenda a melhor estratégia de carregamento baseada no tamanho do arquivo.
    
    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        
    Returns:
        Dicionário com recomendações
    """
    info = obter_info_arquivo(caminho_arquivo)
    
    if 'erro' in info:
        return {'erro': info['erro']}
    
    tamanho_mb = info['tamanho_mb']
    recomendacao = {}
    
    if tamanho_mb < 50:
        recomendacao = {
            'estrategia': 'carregamento_completo',
            'chunksize': None,
            'otimizar_memoria': True,
            'justificativa': 'Arquivo pequeno, pode ser carregado completamente na memória'
        }
    elif tamanho_mb < 200:
        recomendacao = {
            'estrategia': 'carregamento_otimizado',
            'chunksize': None,
            'otimizar_memoria': True,
            'justificativa': 'Arquivo médio, carregamento completo com otimização de tipos'
        }
    else:
        chunk_size = max(1000, min(10000, int(100000 / info['num_colunas'])))
        recomendacao = {
            'estrategia': 'carregamento_em_chunks',
            'chunksize': chunk_size,
            'otimizar_memoria': True,
            'justificativa': f'Arquivo grande ({tamanho_mb:.1f}MB), usar chunks de {chunk_size} linhas'
        }
    
    recomendacao['info_arquivo'] = info
    return recomendacao