"""
Módulo de Análise de Dados
Contém funções para realizar análises estatísticas, identificação de padrões e detecção de anomalias.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def descrever_dados(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Retorna uma descrição completa dos dados.
    
    Args:
        df: DataFrame a ser analisado
        
    Returns:
        Dicionário com informações descritivas dos dados
    """
    from io import StringIO
    
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    info_string = info_buffer.getvalue()
    
    # Análise de valores nulos
    valores_nulos = df.isnull().sum()
    percentual_nulos = (valores_nulos / len(df)) * 100
    
    # Análise de tipos de dados
    tipos_dados = df.dtypes.value_counts()
    
    # Estatísticas básicas para colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'forma_dados': {
            'linhas': len(df),
            'colunas': len(df.columns),
            'tamanho_memoria_mb': df.memory_usage(deep=True).sum() / (1024**2)
        },
        'info_detalhada': info_string,
        'estatisticas_descritivas': df.describe().to_dict(),
        'tipos_dados': tipos_dados.to_dict(),
        'valores_nulos': {
            'contagem': valores_nulos.to_dict(),
            'percentual': percentual_nulos.to_dict()
        },
        'colunas_por_tipo': {
            'numericas': colunas_numericas,
            'categoricas': colunas_categoricas
        }
    }


def calcular_estatisticas(df: pd.DataFrame, coluna: str) -> Dict[str, Any]:
    """
    Calcula estatísticas detalhadas para uma coluna específica.
    
    Args:
        df: DataFrame
        coluna: Nome da coluna
        
    Returns:
        Dicionário com estatísticas da coluna
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    serie = df[coluna].dropna()
    
    if len(serie) == 0:
        return {'erro': 'Coluna contém apenas valores nulos'}
    
    resultado = {
        'nome_coluna': coluna,
        'tipo_dados': str(df[coluna].dtype),
        'valores_totais': len(df[coluna]),
        'valores_nulos': df[coluna].isnull().sum(),
        'valores_unicos': df[coluna].nunique()
    }
    
    # Estatísticas para dados numéricos
    if pd.api.types.is_numeric_dtype(serie):
        resultado.update({
            'estatisticas_numericas': {
                'minimo': float(serie.min()),
                'maximo': float(serie.max()),
                'media': float(serie.mean()),
                'mediana': float(serie.median()),
                'moda': serie.mode().tolist()[0] if not serie.mode().empty else None,
                'desvio_padrao': float(serie.std()) if pd.api.types.is_number(serie.std()) else 0.0,
                'variancia': float(serie.var()) if pd.api.types.is_number(serie.var()) else 0.0,
                'coeficiente_variacao': float(serie.std() / serie.mean()) if serie.mean() != 0 else 0.0,
                'assimetria': float(stats.skew(serie)),
                'curtose': float(stats.kurtosis(serie)),
                'percentis': {
                    'p25': float(serie.quantile(0.25)),
                    'p50': float(serie.quantile(0.50)),
                    'p75': float(serie.quantile(0.75)),
                    'p90': float(serie.quantile(0.90)),
                    'p95': float(serie.quantile(0.95)),
                    'p99': float(serie.quantile(0.99))
                }
            }
        })
    
    # Estatísticas para dados categóricos
    else:
        valores_mais_frequentes = serie.value_counts().head(10)
        resultado.update({
            'estatisticas_categoricas': {
                'valores_mais_frequentes': valores_mais_frequentes.to_dict(),
                'categoria_mais_comum': valores_mais_frequentes.index[0] if not valores_mais_frequentes.empty else None,
                'frequencia_mais_comum': int(valores_mais_frequentes.iloc[0]) if not valores_mais_frequentes.empty else None
            }
        })
    
    return resultado


def encontrar_valores_frequentes(df: pd.DataFrame, coluna: str, top_n: int = 20) -> Dict[str, Any]:
    """
    Encontra os valores mais frequentes em uma coluna.
    
    Args:
        df: DataFrame
        coluna: Nome da coluna
        top_n: Número de valores mais frequentes a retornar
        
    Returns:
        Dicionário com análise de frequência
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    # Contar valores
    contagem_valores = df[coluna].value_counts().head(top_n)
    percentual_valores = (df[coluna].value_counts(normalize=True) * 100).head(top_n)
    
    # Análise de distribuição
    valores_unicos = df[coluna].nunique()
    total_valores = len(df[coluna].dropna())
    
    return {
        'coluna': coluna,
        'valores_unicos': valores_unicos,
        'total_valores_validos': total_valores,
        'valores_mais_frequentes': {
            'contagem': contagem_valores.to_dict(),
            'percentual': percentual_valores.to_dict()
        },
        'distribuicao': {
            'entropia': float(-np.sum(percentual_valores/100 * np.log2(percentual_valores/100 + 1e-10))),
            'concentracao_top10': float(percentual_valores.head(10).sum()),
            'eh_uniforme': valores_unicos == total_valores  # Todos valores únicos
        }
    }


def detectar_outliers_iqr(df: pd.DataFrame, coluna: str) -> Dict[str, Any]:
    """
    Detecta outliers usando o método do Intervalo Interquartil (IQR).
    
    Args:
        df: DataFrame
        coluna: Nome da coluna numérica
        
    Returns:
        Dicionário com informações sobre outliers
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[coluna]):
        raise ValueError(f"Coluna '{coluna}' deve ser numérica para detecção de outliers")
    
    serie = df[coluna].dropna()
    
    if len(serie) == 0:
        return {'erro': 'Coluna contém apenas valores nulos'}
    
    # Calcular quartis e IQR
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    
    # Limites para outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers_inferiores = serie[serie < limite_inferior]
    outliers_superiores = serie[serie > limite_superior]
    outliers_todos = pd.concat([outliers_inferiores, outliers_superiores])
    
    # Outliers extremos (3 * IQR)
    limite_extremo_inferior = Q1 - 3 * IQR
    limite_extremo_superior = Q3 + 3 * IQR
    outliers_extremos = serie[(serie < limite_extremo_inferior) | (serie > limite_extremo_superior)]
    
    return {
        'coluna': coluna,
        'quartis': {
            'Q1': float(Q1),
            'Q2_mediana': float(serie.median()),
            'Q3': float(Q3),
            'IQR': float(IQR)
        },
        'limites_outliers': {
            'inferior': float(limite_inferior),
            'superior': float(limite_superior)
        },
        'outliers': {
            'total': len(outliers_todos),
            'percentual': float(len(outliers_todos) / len(serie) * 100),
            'inferiores': {
                'quantidade': len(outliers_inferiores),
                'valores': outliers_inferiores.tolist()[:10]  # Máximo 10 para não sobrecarregar
            },
            'superiores': {
                'quantidade': len(outliers_superiores),
                'valores': outliers_superiores.tolist()[:10]  # Máximo 10 para não sobrecarregar
            },
            'extremos': {
                'quantidade': len(outliers_extremos),
                'valores': outliers_extremos.tolist()[:10]  # Máximo 10 para não sobrecarregar
            }
        },
        'estatisticas_sem_outliers': {
            'media': float(serie[(serie >= limite_inferior) & (serie <= limite_superior)].mean()),
            'desvio_padrao': float(serie[(serie >= limite_inferior) & (serie <= limite_superior)].std())
        }
    }


def calcular_correlacao(df: pd.DataFrame, metodo: str = 'pearson') -> Dict[str, Any]:
    """
    Calcula a matriz de correlação para variáveis numéricas.
    
    Args:
        df: DataFrame
        metodo: Método de correlação ('pearson', 'spearman', 'kendall')
        
    Returns:
        Dicionário com matriz de correlação e análises
    """
    # Selecionar apenas colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])
    
    if df_numerico.empty:
        return {'erro': 'Nenhuma coluna numérica encontrada para calcular correlação'}
    
    # Validar método de correlação
    metodos_validos = ['pearson', 'kendall', 'spearman']
    if metodo not in metodos_validos:
        metodo = 'pearson'
    
    # Calcular matriz de correlação  
    from typing import Literal
    metodo_typed: Literal['pearson', 'kendall', 'spearman'] = metodo  # type: ignore
    matriz_correlacao = df_numerico.corr(method=metodo_typed)
    
    # Encontrar correlações mais fortes (excluindo diagonal)
    correlacoes_sem_diagonal = matriz_correlacao.where(
        ~np.eye(matriz_correlacao.shape[0], dtype=bool)
    ).stack().dropna()
    
    # Ordenar por valor absoluto da correlação
    correlacoes_ordenadas = correlacoes_sem_diagonal.abs().sort_values(ascending=False)
    
    # Pares com correlação forte (|r| > 0.7)
    correlacoes_fortes = correlacoes_ordenadas[correlacoes_ordenadas > 0.7]
    
    # Pares com correlação moderada (0.3 < |r| <= 0.7)
    correlacoes_moderadas = correlacoes_ordenadas[
        (correlacoes_ordenadas > 0.3) & (correlacoes_ordenadas <= 0.7)
    ]
    
    return {
        'metodo': metodo,
        'colunas_analisadas': df_numerico.columns.tolist(),
        'matriz_correlacao': matriz_correlacao.to_dict(),
        'correlacoes_mais_fortes': correlacoes_ordenadas.head(20).to_dict(),
        'analise_correlacoes': {
            'correlacoes_fortes': {
                'quantidade': len(correlacoes_fortes),
                'pares': correlacoes_fortes.to_dict()
            },
            'correlacoes_moderadas': {
                'quantidade': len(correlacoes_moderadas),
                'pares': dict(list(correlacoes_moderadas.items())[:10])  # Top 10
            }
        },
        'estatisticas_matriz': {
            'correlacao_media': float(correlacoes_sem_diagonal.mean()) if pd.api.types.is_number(correlacoes_sem_diagonal.mean()) else 0.0,
            'correlacao_maxima': float(correlacoes_sem_diagonal.max()) if pd.api.types.is_number(correlacoes_sem_diagonal.max()) else 0.0,
            'correlacao_minima': float(correlacoes_sem_diagonal.min()) if pd.api.types.is_number(correlacoes_sem_diagonal.min()) else 0.0,
            'desvio_padrao_correlacoes': float(correlacoes_sem_diagonal.std()) if pd.api.types.is_number(correlacoes_sem_diagonal.std()) else 0.0
        }
    }


def detectar_padroes_temporais(df: pd.DataFrame, coluna_data: str, coluna_valor: str) -> Dict[str, Any]:
    """
    Detecta padrões temporais em uma série temporal.
    
    Args:
        df: DataFrame
        coluna_data: Nome da coluna com datas
        coluna_valor: Nome da coluna com valores
        
    Returns:
        Dicionário com análise temporal
    """
    if coluna_data not in df.columns or coluna_valor not in df.columns:
        raise ValueError("Colunas especificadas não encontradas no DataFrame")
    
    # Converter coluna de data
    df_temp = df.copy()
    try:
        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data])
    except:
        raise ValueError(f"Não foi possível converter '{coluna_data}' para datetime")
    
    if not pd.api.types.is_numeric_dtype(df_temp[coluna_valor]):
        raise ValueError(f"Coluna '{coluna_valor}' deve ser numérica")
    
    # Ordenar por data
    df_temp = df_temp.sort_values(coluna_data)
    
    # Análises temporais
    df_temp['ano'] = df_temp[coluna_data].dt.year
    df_temp['mes'] = df_temp[coluna_data].dt.month
    df_temp['dia_semana'] = df_temp[coluna_data].dt.dayofweek
    df_temp['hora'] = df_temp[coluna_data].dt.hour
    
    # Estatísticas por período
    stats_anuais = df_temp.groupby('ano')[coluna_valor].agg(['mean', 'std', 'count']).to_dict()
    stats_mensais = df_temp.groupby('mes')[coluna_valor].agg(['mean', 'std', 'count']).to_dict()
    stats_dia_semana = df_temp.groupby('dia_semana')[coluna_valor].agg(['mean', 'std', 'count']).to_dict()
    
    return {
        'periodo_analise': {
            'data_inicio': df_temp[coluna_data].min().isoformat(),
            'data_fim': df_temp[coluna_data].max().isoformat(),
            'total_registros': len(df_temp)
        },
        'tendencia': {
            'correlacao_temporal': float(df_temp[coluna_valor].corr(
                pd.to_numeric(df_temp[coluna_data])
            )) if len(df_temp) > 1 else None
        },
        'sazonalidade': {
            'por_ano': stats_anuais,
            'por_mes': stats_mensais,
            'por_dia_semana': stats_dia_semana
        }
    }


def analisar_distribuicao(df: pd.DataFrame, coluna: str) -> Dict[str, Any]:
    """
    Analisa a distribuição de uma variável numérica.
    
    Args:
        df: DataFrame
        coluna: Nome da coluna numérica
        
    Returns:
        Dicionário com análise de distribuição
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[coluna]):
        raise ValueError(f"Coluna '{coluna}' deve ser numérica para análise de distribuição")
    
    serie = df[coluna].dropna()
    
    if len(serie) == 0:
        return {'erro': 'Coluna contém apenas valores nulos'}
    
    # Testes de normalidade
    if len(serie) >= 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(serie.sample(min(5000, len(serie))))
        except:
            shapiro_stat, shapiro_p = None, None
        
        try:
            ks_stat, ks_p = stats.kstest(serie, 'norm', args=(serie.mean(), serie.std()))
        except:
            ks_stat, ks_p = None, None
    else:
        shapiro_stat, shapiro_p = None, None
        ks_stat, ks_p = None, None
    
    return {
        'coluna': coluna,
        'estatisticas_distribuicao': {
            'assimetria': float(stats.skew(serie)),
            'curtose': float(stats.kurtosis(serie)),
            'amplitude': float(serie.max() - serie.min()),
            'coeficiente_variacao': float(serie.std() / serie.mean()) if serie.mean() != 0 else None
        },
        'testes_normalidade': {
            'shapiro_wilk': {
                'estatistica': float(shapiro_stat) if shapiro_stat else None,
                'p_valor': float(shapiro_p) if shapiro_p else None,
                'eh_normal': bool(shapiro_p > 0.05) if shapiro_p else None
            },
            'kolmogorov_smirnov': {
                'estatistica': float(ks_stat) if ks_stat else None,
                'p_valor': float(ks_p) if ks_p else None,
                'eh_normal': bool(ks_p > 0.05) if ks_p else None
            }
        },
        'interpretacao': {
            'formato_distribuicao': (
                'Simétrica' if abs(stats.skew(serie)) < 0.5 else
                'Assimétrica à direita' if stats.skew(serie) > 0 else
                'Assimétrica à esquerda'
            ),
            'achatamento': (
                'Normal' if -0.5 <= stats.kurtosis(serie) <= 0.5 else
                'Leptocúrtica (mais pontuda)' if stats.kurtosis(serie) > 0.5 else
                'Platicúrtica (mais achatada)'
            )
        }
    }