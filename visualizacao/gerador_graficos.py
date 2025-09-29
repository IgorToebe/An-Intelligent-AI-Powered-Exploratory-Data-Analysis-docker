"""
Módulo de Geração de Gráficos
Responsável por criar visualizações usando Matplotlib e Seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Literal
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configurações globais para os gráficos
plt.style.use('default')
sns.set_palette("husl")

# Diretório para salvar gráficos
GRAFICOS_DIR = Path("temp_graficos")
GRAFICOS_DIR.mkdir(exist_ok=True)


def configurar_estilo_grafico(estilo: str = 'whitegrid', paleta: str = 'husl', 
                             tamanho_figura: Tuple[int, int] = (10, 6)):
    """
    Configura o estilo global dos gráficos.
    
    Args:
        estilo: Estilo do seaborn ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        paleta: Paleta de cores
        tamanho_figura: Tamanho padrão das figuras (largura, altura)
    """
    # Validar estilo
    estilos_validos: List[Literal['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']] = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
    if estilo not in estilos_validos:
        estilo = 'whitegrid'
    
    sns.set_style(estilo)  # type: ignore
    sns.set_palette(paleta)
    plt.rcParams['figure.figsize'] = tamanho_figura
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def gerar_histograma(df: pd.DataFrame, coluna: str, bins: int = 30, 
                    densidade: bool = False, titulo: Optional[str] = None,
                    salvar_como: Optional[str] = None) -> str:
    """
    Cria um histograma para a distribuição de uma coluna.
    
    Args:
        df: DataFrame
        coluna: Nome da coluna
        bins: Número de bins do histograma
        densidade: Se True, mostra densidade; se False, mostra frequência
        titulo: Título personalizado
        salvar_como: Nome do arquivo (sem extensão)
        
    Returns:
        Caminho do arquivo salvo
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[coluna]):
        raise ValueError(f"Coluna '{coluna}' deve ser numérica para histograma")
    
    # Remover valores nulos
    dados = df[coluna].dropna()
    
    if len(dados) == 0:
        raise ValueError(f"Coluna '{coluna}' não contém valores válidos")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Criar histograma
    n, bins_edges, patches = ax.hist(dados, bins=bins, density=densidade, 
                                   alpha=0.7, color='skyblue', edgecolor='black')
    
    # Adicionar linha de densidade se solicitado
    if densidade:
        # Calcular e plotar curva de densidade
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(dados)
            x_range = np.linspace(dados.min(), dados.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Densidade estimada')
            ax.legend()
        except:
            pass
    
    # Configurar título e labels
    if titulo is None:
        titulo = f'Distribuição de {coluna}'
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel(coluna, fontsize=12)
    ax.set_ylabel('Densidade' if densidade else 'Frequência', fontsize=12)
    
    # Adicionar estatísticas no gráfico
    media = dados.mean()
    mediana = dados.median()
    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    ax.legend()
    
    # Melhorar aparência
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar arquivo
    if salvar_como is None:
        salvar_como = f"histograma_{coluna}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    caminho_arquivo = GRAFICOS_DIR / f"{salvar_como}.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(caminho_arquivo)


def gerar_dispersao(df: pd.DataFrame, coluna1: str, coluna2: str,
                   cor_por: Optional[str] = None, tamanho_por: Optional[str] = None,
                   titulo: Optional[str] = None, salvar_como: Optional[str] = None) -> str:
    """
    Cria um gráfico de dispersão entre duas variáveis.
    
    Args:
        df: DataFrame
        coluna1: Nome da primeira coluna (eixo x)
        coluna2: Nome da segunda coluna (eixo y)
        cor_por: Coluna para colorir os pontos
        tamanho_por: Coluna para variar o tamanho dos pontos
        titulo: Título personalizado
        salvar_como: Nome do arquivo (sem extensão)
        
    Returns:
        Caminho do arquivo salvo
    """
    if coluna1 not in df.columns or coluna2 not in df.columns:
        raise ValueError("Uma ou ambas as colunas não foram encontradas no DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[coluna1]) or not pd.api.types.is_numeric_dtype(df[coluna2]):
        raise ValueError("Ambas as colunas devem ser numéricas para gráfico de dispersão")
    
    # Filtrar dados válidos
    colunas_analisar = [coluna1, coluna2]
    if cor_por:
        colunas_analisar.append(cor_por)
    if tamanho_por:
        colunas_analisar.append(tamanho_por)
    
    df_filtrado = df[colunas_analisar].dropna()
    
    if len(df_filtrado) == 0:
        raise ValueError("Não há dados válidos para todas as colunas especificadas")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Preparar argumentos para scatter
    scatter_args = {
        'x': df_filtrado[coluna1],
        'y': df_filtrado[coluna2],
        'alpha': 0.7
    }
    
    if cor_por and cor_por in df_filtrado.columns:
        if pd.api.types.is_numeric_dtype(df_filtrado[cor_por]):
            scatter_args['c'] = df_filtrado[cor_por]
            scatter_args['cmap'] = 'viridis'
        else:
            # Para variáveis categóricas, usar seaborn
            sns.scatterplot(data=df_filtrado, x=coluna1, y=coluna2, hue=cor_por, ax=ax)
            
    if tamanho_por and tamanho_por in df_filtrado.columns and pd.api.types.is_numeric_dtype(df_filtrado[tamanho_por]):
        # Normalizar tamanhos entre 20 e 200
        tamanhos = df_filtrado[tamanho_por]
        tamanhos_norm = 20 + (tamanhos - tamanhos.min()) / (tamanhos.max() - tamanhos.min()) * 180
        scatter_args['s'] = tamanhos_norm
    
    # Criar scatter plot se não usou seaborn
    if not (cor_por and not pd.api.types.is_numeric_dtype(df_filtrado[cor_por])):
        scatter = ax.scatter(**scatter_args)
        
        # Adicionar colorbar se necessário
        if cor_por and pd.api.types.is_numeric_dtype(df_filtrado[cor_por]):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(cor_por, fontsize=12)
    
    # Calcular e plotar linha de regressão
    from scipy.stats import linregress
    try:
        # Garantir que r_value seja um float antes de usar
        result = linregress(df_filtrado[coluna1], df_filtrado[coluna2])
        slope, intercept, r_value, p_value, std_err = result
        
        if isinstance(r_value, (int, float)):
            x_line = np.array([df_filtrado[coluna1].min(), df_filtrado[coluna1].max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', linewidth=2, 
                    label=f'R² = {r_value**2:.3f}')
            ax.legend()
    except Exception:
        pass
    
    # Configurar título e labels
    if titulo is None:
        titulo = f'Dispersão: {coluna1} vs {coluna2}'
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel(coluna1, fontsize=12)
    ax.set_ylabel(coluna2, fontsize=12)
    
    # Melhorar aparência
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar arquivo
    if salvar_como is None:
        salvar_como = f"dispersao_{coluna1}_{coluna2}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    caminho_arquivo = GRAFICOS_DIR / f"{salvar_como}.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(caminho_arquivo)


def gerar_heatmap_correlacao(df: pd.DataFrame, metodo: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
                           apenas_numericas: bool = True, titulo: Optional[str] = None,
                           salvar_como: Optional[str] = None) -> str:
    """
    Visualiza a matriz de correlação como um heatmap.
    
    Args:
        df: DataFrame
        metodo: Método de correlação ('pearson', 'spearman', 'kendall')
        apenas_numericas: Se deve considerar apenas colunas numéricas
        titulo: Título personalizado
        salvar_como: Nome do arquivo (sem extensão)
        
    Returns:
        Caminho do arquivo salvo
    """
    # Selecionar colunas apropriadas
    if apenas_numericas:
        df_corr = df.select_dtypes(include=[np.number])
    else:
        df_corr = df

    # Validação extra de segurança para o método
    if metodo not in ('pearson', 'spearman', 'kendall'):
        raise ValueError("Método de correlação inválido. Use 'pearson', 'spearman' ou 'kendall'.")
    
    if df_corr.empty:
        raise ValueError("Nenhuma coluna numérica encontrada para calcular correlação")
    
    if len(df_corr.columns) < 2:
        raise ValueError("Pelo menos duas colunas numéricas são necessárias para correlação")
    
    # Calcular matriz de correlação
    matriz_correlacao = df_corr.corr(method=metodo)
    
    # Configurar tamanho da figura baseado no número de variáveis
    n_vars = len(matriz_correlacao.columns)
    fig_size = max(8, min(16, n_vars * 0.8))
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Criar máscara para o triângulo superior
    mask = np.triu(np.ones_like(matriz_correlacao, dtype=bool))
    
    # Gerar heatmap
    sns.heatmap(matriz_correlacao, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={"shrink": .8},
                mask=mask,
                ax=ax)
    
    # Configurar título
    if titulo is None:
        titulo = f'Matriz de Correlação ({metodo.title()})'
    
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
    
    # Melhorar aparência
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Salvar arquivo
    if salvar_como is None:
        salvar_como = f"heatmap_correlacao_{metodo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    caminho_arquivo = GRAFICOS_DIR / f"{salvar_como}.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(caminho_arquivo)


def gerar_boxplot(df: pd.DataFrame, coluna: str, agrupar_por: Optional[str] = None,
                 titulo: Optional[str] = None, salvar_como: Optional[str] = None) -> str:
    """
    Cria um boxplot para identificar outliers e distribuição.
    
    Args:
        df: DataFrame
        coluna: Nome da coluna numérica
        agrupar_por: Coluna categórica para agrupar
        titulo: Título personalizado
        salvar_como: Nome do arquivo (sem extensão)
        
    Returns:
        Caminho do arquivo salvo
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[coluna]):
        raise ValueError(f"Coluna '{coluna}' deve ser numérica para boxplot")
    
    # Filtrar dados válidos
    if agrupar_por:
        if agrupar_por not in df.columns:
            raise ValueError(f"Coluna de agrupamento '{agrupar_por}' não encontrada")
        df_filtrado = df[[coluna, agrupar_por]].dropna()
    else:
        df_filtrado = df[[coluna]].dropna()
    
    if len(df_filtrado) == 0:
        raise ValueError("Não há dados válidos para criar o boxplot")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Criar boxplot
    if agrupar_por:
        sns.boxplot(data=df_filtrado, x=agrupar_por, y=coluna, ax=ax)
        plt.xticks(rotation=45)
    else:
        sns.boxplot(data=df_filtrado, y=coluna, ax=ax)
    
    # Configurar título
    if titulo is None:
        if agrupar_por:
            titulo = f'Boxplot de {coluna} por {agrupar_por}'
        else:
            titulo = f'Boxplot de {coluna}'
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    # Melhorar aparência
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar arquivo
    if salvar_como is None:
        if agrupar_por:
            salvar_como = f"boxplot_{coluna}_{agrupar_por}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            salvar_como = f"boxplot_{coluna}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    caminho_arquivo = GRAFICOS_DIR / f"{salvar_como}.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(caminho_arquivo)


def gerar_grafico_barras(df: pd.DataFrame, coluna: str, top_n: int = 15,
                        horizontal: bool = False, titulo: Optional[str] = None,
                        salvar_como: Optional[str] = None) -> str:
    """
    Cria um gráfico de barras para variáveis categóricas.
    
    Args:
        df: DataFrame
        coluna: Nome da coluna categórica
        top_n: Número de categorias mais frequentes a mostrar
        horizontal: Se True, cria gráfico horizontal
        titulo: Título personalizado
        salvar_como: Nome do arquivo (sem extensão)
        
    Returns:
        Caminho do arquivo salvo
    """
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    # Contar valores
    contagem_valores = df[coluna].value_counts().head(top_n)
    
    if len(contagem_valores) == 0:
        raise ValueError(f"Coluna '{coluna}' não contém valores válidos")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 8) if horizontal else (10, 6))
    
    # Criar gráfico de barras
    if horizontal:
        contagem_valores.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_xlabel('Frequência', fontsize=12)
        ax.set_ylabel(coluna, fontsize=12)
    else:
        contagem_valores.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel(coluna, fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    # Configurar título
    if titulo is None:
        titulo = f'Distribuição de {coluna} (Top {top_n})'
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    # Adicionar valores nas barras
    for i, v in enumerate(contagem_valores.values):
        if horizontal:
            ax.text(v + max(contagem_valores.values) * 0.01, i, str(v), 
                   va='center', fontsize=10)
        else:
            ax.text(i, v + max(contagem_valores.values) * 0.01, str(v), 
                   ha='center', va='bottom', fontsize=10)
    
    # Melhorar aparência
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar arquivo
    if salvar_como is None:
        salvar_como = f"barras_{coluna}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    caminho_arquivo = GRAFICOS_DIR / f"{salvar_como}.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(caminho_arquivo)


def gerar_grafico_linha(df: pd.DataFrame, coluna_x: str, coluna_y: str,
                       agrupar_por: Optional[str] = None, titulo: Optional[str] = None,
                       salvar_como: Optional[str] = None) -> str:
    """
    Cria um gráfico de linha temporal ou de série.
    
    Args:
        df: DataFrame
        coluna_x: Nome da coluna do eixo x (geralmente temporal)
        coluna_y: Nome da coluna do eixo y (valores)
        agrupar_por: Coluna para criar múltiplas linhas
        titulo: Título personalizado
        salvar_como: Nome do arquivo (sem extensão)
        
    Returns:
        Caminho do arquivo salvo
    """
    if coluna_x not in df.columns or coluna_y not in df.columns:
        raise ValueError("Uma ou ambas as colunas não foram encontradas no DataFrame")
    
    # Filtrar dados válidos
    colunas_necessarias = [coluna_x, coluna_y]
    if agrupar_por:
        colunas_necessarias.append(agrupar_por)
    
    df_filtrado = df[colunas_necessarias].dropna()
    
    if len(df_filtrado) == 0:
        raise ValueError("Não há dados válidos para criar o gráfico")
    
    # Tentar converter coluna_x para datetime se parecer ser temporal
    try:
        df_filtrado[coluna_x] = pd.to_datetime(df_filtrado[coluna_x])
        is_temporal = True
    except:
        is_temporal = False
    
    # Ordenar por coluna_x
    df_filtrado = df_filtrado.sort_values(coluna_x)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Criar gráfico de linha
    if agrupar_por:
        for grupo in df_filtrado[agrupar_por].unique():
            dados_grupo = df_filtrado[df_filtrado[agrupar_por] == grupo]
            ax.plot(dados_grupo[coluna_x], dados_grupo[coluna_y], 
                   label=str(grupo), linewidth=2, marker='o', markersize=4)
        ax.legend()
    else:
        ax.plot(df_filtrado[coluna_x], df_filtrado[coluna_y], 
               linewidth=2, marker='o', markersize=4, color='blue')
    
    # Configurar título e labels
    if titulo is None:
        titulo = f'{coluna_y} por {coluna_x}'
    
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel(coluna_x, fontsize=12)
    ax.set_ylabel(coluna_y, fontsize=12)
    
    # Formatar eixo x se for temporal
    if is_temporal:
        plt.xticks(rotation=45)
        fig.autofmt_xdate()
    
    # Melhorar aparência
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar arquivo
    if salvar_como is None:
        salvar_como = f"linha_{coluna_x}_{coluna_y}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    caminho_arquivo = GRAFICOS_DIR / f"{salvar_como}.png"
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(caminho_arquivo)


def limpar_graficos_temporarios():
    """[Desativado] Não remove mais gráficos temporários."""
    return "Limpeza desativada: nenhum gráfico foi removido."


def arquivar_graficos_temporarios() -> str:
    """Move os gráficos PNG atuais para uma pasta de arquivo por data.

    Estrutura de destino: temp_graficos/arquivados/YYYY-MM-DD/

    Returns:
        Mensagem com resumo da operação.
    """
    try:
        destino_raiz = GRAFICOS_DIR / "arquivados" / pd.Timestamp.now().strftime('%Y-%m-%d')
        destino_raiz.mkdir(parents=True, exist_ok=True)

        movimentados = 0
        for arquivo in GRAFICOS_DIR.glob("*.png"):
            try:
                destino = destino_raiz / arquivo.name
                # Evitar sobrescrita: se existir, gerar nome único
                if destino.exists():
                    base = arquivo.stem
                    ext = arquivo.suffix
                    idx = 1
                    while True:
                        candidato = destino_raiz / f"{base}_{idx}{ext}"
                        if not candidato.exists():
                            destino = candidato
                            break
                        idx += 1
                arquivo.rename(destino)
                movimentados += 1
            except Exception:
                # Continua tentando os demais
                pass

        return f"📦 Arquivamento concluído. Arquivos movidos: {movimentados}. Pasta: {destino_raiz}"
    except Exception as e:
        return f"❌ Erro ao arquivar gráficos: {e}"


# Configurar estilo padrão ao importar o módulo
configurar_estilo_grafico()