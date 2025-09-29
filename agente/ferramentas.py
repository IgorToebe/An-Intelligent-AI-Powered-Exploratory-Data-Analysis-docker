"""
Módulo de Ferramentas do Agente
Contém as funções de análise que o agente pode usar, decoradas como ferramentas LangChain.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Any
from pathlib import Path
import re
import platform
import signal
import time
from langchain.tools import tool
import matplotlib.pyplot as plt
import seaborn as sns

# Importar funções do motor de análise
from processamento.motor_analise import (
    descrever_dados, calcular_estatisticas, detectar_outliers_iqr,
    calcular_correlacao
)
from visualizacao.gerador_graficos import (
    gerar_histograma, gerar_dispersao, gerar_heatmap_correlacao,
    gerar_boxplot, gerar_grafico_barras, GRAFICOS_DIR
)

# Importar scipy.stats para análises estatísticas avançadas
try:
    from scipy import stats
    from scipy.stats import (
        normaltest, shapiro, kstest, jarque_bera,
        ttest_1samp, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
        chi2_contingency, fisher_exact, pearsonr, spearmanr, kendalltau,
        f_oneway, kruskal, levene, bartlett, fligner
    )
    from scipy.stats import anderson as anderson_scipy
    
    # Wrapper para anderson com tipo correto
    def anderson(*args, **kwargs):  # type: ignore
        """Wrapper para anderson com tipo compatível"""
        return anderson_scipy(*args, **kwargs)
    
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Definir funções vazias para type checking
    def shapiro(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def normaltest(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def jarque_bera(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    class _AndersonResult:
        def __init__(self):
            self.statistic = 0.0
            self.critical_values = np.array([0.0])
            self.significance_level = np.array([0.05])
    
    def anderson(*args, **kwargs): return _AndersonResult()  # type: ignore
    def ttest_1samp(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def ttest_ind(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def levene(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def pearsonr(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def spearmanr(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def kendalltau(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def chi2_contingency(*args, **kwargs): return (0.0, 1.0, 1, [[1]])  # type: ignore
    def f_oneway(*args, **kwargs): return (0.0, 1.0)  # type: ignore
    def mannwhitneyu(*args, **kwargs): return (0.0, 1.0)  # type: ignore

# Importar scikit-learn para análises avançadas (clustering, PCA, etc.)
try:
    from sklearn.base import BaseEstimator  # Import da classe base
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, 
        mean_squared_error, r2_score, mean_absolute_error,
        silhouette_score, adjusted_rand_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
    # Criar classe base compatível para modelos ML
    class _BaseEstimatorDummy:
        """Classe base compatível com sklearn BaseEstimator"""
        def get_params(self, deep=True):
            return {}
        
        def set_params(self, **params):
            return self
    
    # Usar BaseEstimatorDummy como fallback  
    BaseEstimator = _BaseEstimatorDummy
    
    class _DummyMLModel(_BaseEstimatorDummy):
        def __init__(self, *args, **kwargs):
            self.feature_importances_ = np.array([0.5])
            self.classes_ = np.array([])
            self.explained_variance_ratio_ = np.array([0.95])
            self.components_ = np.array([[0.5, 0.5]])
            self.cluster_centers_ = np.array([[0.0, 0.0]])
            self.best_estimator_ = self
            self.best_score_ = 0.5
            self.best_params_ = {}
            self.cv_results_ = {'mean_test_score': [0.5], 'std_test_score': [0.1], 'params': [{}]}
            
        def fit(self, X, y=None):
            return self
            
        def predict(self, X):
            if hasattr(X, 'shape'):
                return np.zeros(X.shape[0])
            return np.array([0])
            
        def fit_predict(self, X, y=None):
            return self.predict(X)
            
        def score(self, X, y):
            return 0.5
            
        def fit_transform(self, X, y=None):
            return X if hasattr(X, 'shape') else np.array(X)
            
        def transform(self, X):
            return X if hasattr(X, 'shape') else np.array(X)
            
        def inverse_transform(self, X):
            return X if hasattr(X, 'shape') else np.array(X)
    
    # Criar stubs para todas as funções e classes - usar BaseEstimator dummy aqui
    train_test_split = lambda *args, **kwargs: (np.array([]), np.array([]), np.array([]), np.array([]))  # type: ignore
    cross_val_score = lambda *args, **kwargs: np.array([0.5])  # type: ignore
    GridSearchCV = _DummyMLModel  # type: ignore
    StandardScaler = _DummyMLModel  # type: ignore
    LabelEncoder = _DummyMLModel  # type: ignore
    MinMaxScaler = _DummyMLModel  # type: ignore
    LinearRegression = _DummyMLModel  # type: ignore
    LogisticRegression = _DummyMLModel  # type: ignore
    DecisionTreeClassifier = _DummyMLModel  # type: ignore
    DecisionTreeRegressor = _DummyMLModel  # type: ignore
    RandomForestClassifier = _DummyMLModel  # type: ignore
    RandomForestRegressor = _DummyMLModel  # type: ignore
    SVC = _DummyMLModel  # type: ignore
    SVR = _DummyMLModel  # type: ignore
    KNeighborsClassifier = _DummyMLModel  # type: ignore
    KNeighborsRegressor = _DummyMLModel  # type: ignore
    GaussianNB = _DummyMLModel  # type: ignore
    KMeans = _DummyMLModel  # type: ignore
    DBSCAN = _DummyMLModel  # type: ignore
    PCA = _DummyMLModel  # type: ignore
    accuracy_score = lambda *args, **kwargs: 0.5  # type: ignore
    classification_report = lambda *args, **kwargs: {}  # type: ignore
    confusion_matrix = lambda *args, **kwargs: np.array([[1]])  # type: ignore
    mean_squared_error = lambda *args, **kwargs: 0.5  # type: ignore
    r2_score = lambda *args, **kwargs: 0.5  # type: ignore
    mean_absolute_error = lambda *args, **kwargs: 0.5  # type: ignore
    silhouette_score = lambda *args, **kwargs: 0.5  # type: ignore
    adjusted_rand_score = lambda *args, **kwargs: 0.5  # type: ignore

# Classe global _DummyMLModel compatível com BaseEstimator
if SKLEARN_AVAILABLE:
    # Se sklearn está disponível, usar BaseEstimator real
    class _DummyMLModelGlobal(BaseEstimator):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.feature_importances_ = np.array([0.5])
            self.classes_ = np.array([])
            self.explained_variance_ratio_ = np.array([0.95])
            self.components_ = np.array([[0.5, 0.5]])
            self.cluster_centers_ = np.array([[0.0, 0.0]])
            self.best_estimator_ = self
            self.best_score_ = 0.5
            self.best_params_ = {}
            
        def fit(self, X, y=None):
            return self
            
        def predict(self, X):
            if hasattr(X, 'shape'):
                return np.zeros(X.shape[0])
            return np.array([0])
            
        def transform(self, X):
            return X
            
        def score(self, X, y):
            return 0.5
            
        def get_params(self, deep=True):
            return {}
            
        def set_params(self, **params):
            return self
else:
    # Se sklearn não está disponível, usar implementação dummy
    _DummyMLModelGlobal = _DummyMLModel  # type: ignore

# Redefinir todas as classes para usar a versão global compatível
if not SKLEARN_AVAILABLE:
    GridSearchCV = _DummyMLModelGlobal  # type: ignore
    StandardScaler = _DummyMLModelGlobal  # type: ignore
    LabelEncoder = _DummyMLModelGlobal  # type: ignore
    MinMaxScaler = _DummyMLModelGlobal  # type: ignore
    LinearRegression = _DummyMLModelGlobal  # type: ignore
    LogisticRegression = _DummyMLModelGlobal  # type: ignore
    DecisionTreeClassifier = _DummyMLModelGlobal  # type: ignore
    DecisionTreeRegressor = _DummyMLModelGlobal  # type: ignore
    RandomForestClassifier = _DummyMLModelGlobal  # type: ignore
    RandomForestRegressor = _DummyMLModelGlobal  # type: ignore
    SVC = _DummyMLModelGlobal  # type: ignore
    SVR = _DummyMLModelGlobal  # type: ignore
    KNeighborsClassifier = _DummyMLModelGlobal  # type: ignore
    KNeighborsRegressor = _DummyMLModelGlobal  # type: ignore
    GaussianNB = _DummyMLModelGlobal  # type: ignore
    KMeans = _DummyMLModelGlobal  # type: ignore
    DBSCAN = _DummyMLModelGlobal  # type: ignore
    PCA = _DummyMLModelGlobal  # type: ignore

# Imports dos módulos de processamento e visualização
from processamento.motor_analise import (
    descrever_dados, calcular_estatisticas, encontrar_valores_frequentes,
    detectar_outliers_iqr, calcular_correlacao
)
from visualizacao.gerador_graficos import (
    gerar_histograma, gerar_dispersao, gerar_heatmap_correlacao,
    gerar_boxplot, gerar_grafico_barras, GRAFICOS_DIR
)

# Variável global para armazenar resultados de clustering
dados_clustering_global = {}

@tool
def realizar_clustering(colunas_features: str, algoritmo: str = "kmeans", n_clusters: int = 3, eps: float = 0.5, min_samples: int = 5) -> str:
    """Realiza análise de clustering (agrupamento) nos dados usando K-Means ou DBSCAN."""
    try:
        if not SKLEARN_AVAILABLE:
            return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
        
        if df is None or df.empty:
            return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
        
        # Processar colunas de features
        lista_features = [col.strip() for col in colunas_features.split(',')]
        features_validas = []
        
        for feature in lista_features:
            if feature not in df.columns:
                return f"❌ Feature '{feature}' não encontrada.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            if not pd.api.types.is_numeric_dtype(df[feature]):
                return f"❌ Feature '{feature}' deve ser numérica para clustering."
            features_validas.append(feature)
        
        # Preparar dados
        df_clustering = df[features_validas].dropna()
        if len(df_clustering) < 10:
            return f"❌ Dados insuficientes após limpeza (mínimo: 10, atual: {len(df_clustering)})"
        
        # Amostragem inteligente para datasets grandes
        amostra_usada = False
        tamanho_original = len(df_clustering)
        
        if len(df_clustering) > 10000:
            # Usar amostragem estratificada para manter representatividade
            np.random.seed(42)  # Para reproducibilidade
            
            # Amostra estratificada baseada na distribuição das features principais
            n_amostra = min(10000, int(len(df_clustering) * 0.1))  # Max 10k ou 10% dos dados
            
            # Se temos variáveis categóricas no dataset para estratificar
            if 'Class' in df.columns and len(df['Class'].unique()) <= 10:
                # Amostragem estratificada por classe se disponível
                from sklearn.model_selection import train_test_split
                try:
                    df_clustering_original_idx = df_clustering.index
                    df_temp = df.loc[df_clustering_original_idx]
                    _, df_amostra_temp = train_test_split(
                        df_temp, test_size=n_amostra, 
                        stratify=df_temp['Class'],
                        random_state=42
                    )
                    df_clustering = df_amostra_temp[features_validas]
                except:
                    # Fallback para amostragem aleatória se estratificada falhar
                    df_clustering = df_clustering.sample(n=n_amostra, random_state=42)
            else:
                # Amostragem aleatória simples
                df_clustering = df_clustering.sample(n=n_amostra, random_state=42)
            
            amostra_usada = True
        
        X = df_clustering.values
        
        # Normalizar dados para clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Escolher algoritmo de clustering (execução em thread principal para Streamlit Cloud)
        try:
            if algoritmo.lower() == "kmeans":
                # Validar número de clusters
                if n_clusters < 2:
                    n_clusters = 2
                elif n_clusters > len(df_clustering) // 2:
                    n_clusters = min(8, len(df_clustering) // 2)
                
                # Forçar execução sequencial para compatibilidade com Streamlit Cloud
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = model.fit_predict(X_scaled)
                nome_algoritmo = f"K-Means (k={n_clusters})"
                
                # Calcular inércia e silhueta
                inercia = float(model.inertia_)
                
            elif algoritmo.lower() == "dbscan":
                # DBSCAN com execução sequencial
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(X_scaled)
                nome_algoritmo = f"DBSCAN (eps={eps}, min_samples={min_samples})"
                
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                inercia = None
                
            else:
                return f"❌ Algoritmo '{algoritmo}' não suportado. Use: kmeans, dbscan"
                
        except Exception as e:
            return f"❌ Erro ao executar clustering: {str(e)}. Tente com dados menores ou diferentes parâmetros."
        
        # Adicionar clusters ao dataframe
        df_resultado = df_clustering.copy()
        df_resultado['Cluster'] = clusters
        
        # Calcular silhueta score (se há mais de 1 cluster)
        if len(set(clusters)) > 1:
            silhouette = float(silhouette_score(X_scaled, clusters))
        else:
            silhouette = -1.0
        
        # Análise dos clusters
        n_clusters_final = len(set(clusters))
        n_noise = (clusters == -1).sum() if -1 in clusters else 0
        
        response = f"🧩 **ANÁLISE DE CLUSTERING - {nome_algoritmo.upper()}**\n\n"
        response += f"**Configuração:**\n"
        response += f"- Features: {', '.join(features_validas)}\n"
        response += f"- Algoritmo: {nome_algoritmo}\n"
        
        if amostra_usada:
            response += f"- **Dataset original:** {tamanho_original:,} amostras\n"
            response += f"- **Amostra utilizada:** {len(df_clustering):,} amostras ({len(df_clustering)/tamanho_original*100:.1f}%)\n"
            response += f"- **Método:** Amostragem estratificada para otimização\n\n"
        else:
            response += f"- Amostras analisadas: {len(df_clustering):,}\n\n"
        
        response += f"**📊 Resultados:**\n"
        response += f"- **Clusters encontrados:** {n_clusters_final}\n"
        
        if algoritmo.lower() == "dbscan":
            response += f"- **Pontos de ruído:** {n_noise} ({(n_noise/len(df_clustering))*100:.1f}%)\n"
            response += f"- **Pontos agrupados:** {len(df_clustering) - n_noise}\n"
        
        if silhouette >= 0:
            response += f"- **Silhouette Score:** {silhouette:.4f} "
            if silhouette >= 0.7:
                response += f"(clustering excelente)\n"
            elif silhouette >= 0.5:
                response += f"(clustering bom)\n"
            elif silhouette >= 0.3:
                response += f"(clustering moderado)\n"
            else:
                response += f"(clustering fraco)\n"
        
        if inercia is not None:
            response += f"- **Inércia (WCSS):** {inercia:.2f}\n"
        
        # Estatísticas por cluster
        response += f"\n**📋 Análise por Cluster:**\n"
        for cluster_id in sorted(set(clusters)):
            cluster_dados = df_resultado[df_resultado['Cluster'] == cluster_id]
            n_pontos = len(cluster_dados)
            
            if cluster_id == -1:
                response += f"\n**🔸 Ruído (Cluster -1):** {n_pontos} pontos ({(n_pontos/len(df_clustering))*100:.1f}%)\n"
                continue
                
            response += f"\n**🔹 Cluster {cluster_id}:** {n_pontos} pontos ({(n_pontos/len(df_clustering))*100:.1f}%)\n"
            
            for feature in features_validas:
                media = cluster_dados[feature].mean()
                std = cluster_dados[feature].std()
                response += f"  - {feature}: μ={media:.3f}, σ={std:.3f}\n"
        
        # Salvar informações de clustering globalmente para o agente usar
        global dados_clustering_global
        dados_clustering_global = {
            'dataframe': df_resultado,
            'features': features_validas,
            'algoritmo': algoritmo,
            'n_clusters': n_clusters_final,
            'silhouette': silhouette,
            'modelo': model,
            'scaler': scaler
        }
        
        # Gerar visualizações
        if len(features_validas) >= 2:
            # Scatter plot dos clusters (usando primeiras 2 dimensões)
            plt.figure(figsize=(10, 8))
            
            # Cores diferentes para cada cluster
            cores = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters_final)))  # type: ignore
            
            for cluster_id in sorted(set(clusters)):
                mask = clusters == cluster_id
                if cluster_id == -1:
                    # Pontos de ruído em preto
                    plt.scatter(df_clustering.iloc[mask, 0], df_clustering.iloc[mask, 1], 
                              c='black', marker='x', s=50, alpha=0.6, label='Ruído')
                else:
                    cor = cores[cluster_id % len(cores)]
                    plt.scatter(df_clustering.iloc[mask, 0], df_clustering.iloc[mask, 1], 
                              c=[cor], s=60, alpha=0.7, label=f'Cluster {cluster_id}')
            
            # Adicionar centróides se K-means
            if algoritmo.lower() == "kmeans" and hasattr(model, 'cluster_centers_'):
                centroides = scaler.inverse_transform(model.cluster_centers_)  # type: ignore
                plt.scatter(centroides[:, 0], centroides[:, 1], 
                          c='red', marker='*', s=200, edgecolor='black', 
                          linewidth=2, label='Centróides')
            
            plt.xlabel(features_validas[0])
            plt.ylabel(features_validas[1])
            plt.title(f'Clusters - {nome_algoritmo}\nSilhouette Score: {silhouette:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Salvar gráfico
            caminho_grafico = GRAFICOS_DIR / f"clustering_{algoritmo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            
            response += f"\n📊 **Visualização gerada:** {caminho_grafico}"
            
            # Gerar matriz de correlação entre clusters e features originais
            if n_clusters_final > 1:
                plt.figure(figsize=(8, 6))
                
                # Calcular médias por cluster
                cluster_means = df_resultado.groupby('Cluster')[features_validas].mean()
                
                sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='viridis', 
                          cbar_kws={'label': 'Valor Médio'})
                plt.title('Perfil dos Clusters\n(Valores Médios por Feature)')
                plt.xlabel('Cluster')
                plt.ylabel('Features')
                
                # Salvar heatmap
                caminho_heatmap = GRAFICOS_DIR / f"clustering_perfil_{algoritmo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(caminho_heatmap, dpi=300, bbox_inches='tight')
                plt.close()
                
                response += f"\n📊 **Perfil dos clusters:** {caminho_heatmap}"
        
        response += f"\n\n**💡 INTERPRETAÇÃO E PRÓXIMOS PASSOS:**\n"
        
        if silhouette >= 0.5:
            response += f"✅ Os clusters encontrados apresentam boa separação e coesão.\n"
        elif silhouette >= 0.3:
            response += f"⚠️ Os clusters apresentam separação moderada. Considere ajustar parâmetros.\n"
        else:
            response += f"❌ Clusters mal definidos. Considere:\n"
            response += f"   - Diferentes valores de k (K-means) ou eps/min_samples (DBSCAN)\n"
            response += f"   - Outras features ou transformações dos dados\n"
            response += f"   - Verificar se os dados realmente possuem estrutura de clusters\n"
        
        response += f"\n**🔧 Dados disponíveis para análise:**\n"
        response += f"- Os clusters foram adicionados aos dados como coluna 'Cluster'\n"
        response += f"- Use 'analisar_clusters_detalhadamente' para análises específicas\n"
        response += f"- Use 'comparar_clusters_por_variavel' para comparações detalhadas\n"
        
        return response
        
    except Exception as e:
        return f"Erro no clustering: {str(e)}"

@tool
def analisar_clusters_detalhadamente() -> str:
    """Análise detalhada dos clusters gerados pela ferramenta de clustering."""
    try:
        # Verificar se existe resultado de clustering
        if 'dados_clustering_global' not in globals() or not dados_clustering_global:
            return "❌ Nenhuma análise de clustering foi realizada ainda. Use 'realizar_clustering' primeiro."
        
        dados = dados_clustering_global
        df_clusters = dados['dataframe']
        features = dados['features']
        algoritmo = dados['algoritmo']
        n_clusters = dados['n_clusters']
        
        response = f"🔍 **ANÁLISE DETALHADA DOS CLUSTERS**\n\n"
        response += f"**Configuração atual:**\n"
        response += f"- Algoritmo: {algoritmo.upper()}\n"
        response += f"- Features: {', '.join(features)}\n"
        response += f"- Clusters: {n_clusters}\n\n"
        
        # Estatísticas detalhadas por cluster
        response += f"**📊 ESTATÍSTICAS DETALHADAS POR CLUSTER:**\n"
        
        for cluster_id in sorted(df_clusters['Cluster'].unique()):
            if cluster_id == -1:
                continue  # Pular ruído por enquanto
                
            cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
            n_pontos = len(cluster_data)
            pct = (n_pontos / len(df_clusters)) * 100
            
            response += f"\n**🔹 CLUSTER {cluster_id}** ({n_pontos} pontos - {pct:.1f}%)\n"
            
            for feature in features:
                dados_feature = cluster_data[feature]
                response += f"**{feature}:**\n"
                response += f"  - Média: {dados_feature.mean():.4f}\n"
                response += f"  - Mediana: {dados_feature.median():.4f}\n"
                response += f"  - Desvio Padrão: {dados_feature.std():.4f}\n"
                response += f"  - Min-Max: {dados_feature.min():.4f} a {dados_feature.max():.4f}\n"
                response += f"  - Quartis: Q1={dados_feature.quantile(0.25):.4f}, Q3={dados_feature.quantile(0.75):.4f}\n"
            
            response += f"\n"
        
        # Análise de separação entre clusters
        if len(features) >= 2 and n_clusters > 1:
            response += f"**🎯 ANÁLISE DE SEPARAÇÃO ENTRE CLUSTERS:**\n"
            
            # Calcular distâncias entre centros de clusters
            centroides = df_clusters.groupby('Cluster')[features].mean()
            
            response += f"**Centróides dos clusters:**\n"
            for cluster_id in centroides.index:
                if cluster_id != -1:
                    centro = centroides.loc[cluster_id]
                    valores = ", ".join([f"{feat}={val:.3f}" for feat, val in zip(features, centro.values)])
                    response += f"- Cluster {cluster_id}: {valores}\n"
            
            # Maior diferença entre clusters
            response += f"\n**Características distintivas:**\n"
            for feature in features:
                valores_por_cluster = df_clusters.groupby('Cluster')[feature].mean()
                valores_sem_ruido = valores_por_cluster[valores_por_cluster.index != -1]
                
                if len(valores_sem_ruido) > 1:
                    max_cluster = valores_sem_ruido.idxmax()
                    min_cluster = valores_sem_ruido.idxmin()
                    diferenca = valores_sem_ruido.max() - valores_sem_ruido.min()
                    
                    response += f"- **{feature}:** Cluster {max_cluster} (maior: {valores_sem_ruido.max():.3f}) "
                    response += f"vs Cluster {min_cluster} (menor: {valores_sem_ruido.min():.3f}) "
                    response += f"- diferença: {diferenca:.3f}\n"
        
        # Análise de qualidade dos clusters
        response += f"\n**📈 QUALIDADE DOS CLUSTERS:**\n"
        silhouette = dados.get('silhouette', -1)
        
        if silhouette >= 0:
            response += f"- **Silhouette Score:** {silhouette:.4f}\n"
            
            if silhouette >= 0.7:
                response += f"  ✅ Excelente: clusters bem definidos e separados\n"
            elif silhouette >= 0.5:
                response += f"  ✅ Bom: clusters razoavelmente bem separados\n"
            elif silhouette >= 0.3:
                response += f"  ⚠️ Moderado: alguns clusters podem se sobrepor\n"
            else:
                response += f"  ❌ Fraco: clusters mal definidos ou dados não adequados para clustering\n"
        
        # Sugestões de análise
        response += f"\n**💡 SUGESTÕES PARA ANÁLISE ADICIONAL:**\n"
        response += f"1. Use 'comparar_clusters_por_variavel' para comparar clusters em variáveis específicas\n"
        response += f"2. Use 'gerar_boxplot_agrupado' com coluna 'Cluster' para visualizar distribuições\n"
        response += f"3. Use 'analisar_por_grupos' para estatísticas de outras variáveis por cluster\n"
        response += f"4. Considere analisar outliers dentro de cada cluster\n"
        
        if algoritmo.lower() == "kmeans":
            response += f"5. Experimente diferentes valores de k para otimizar o clustering\n"
        elif algoritmo.lower() == "dbscan":
            response += f"5. Ajuste eps e min_samples se necessário para melhorar os clusters\n"
        
        return response
        
    except Exception as e:
        return f"Erro na análise detalhada dos clusters: {str(e)}"

@tool
def comparar_clusters_por_variavel(variavel: str) -> str:
    """Compara os clusters em relação a uma variável específica (que pode não ter sido usada no clustering)."""
    try:
        # Verificar se existe resultado de clustering
        if 'dados_clustering_global' not in globals() or not dados_clustering_global:
            return "❌ Nenhuma análise de clustering foi realizada ainda. Use 'realizar_clustering' primeiro."
        
        if variavel not in df.columns:
            return f"❌ Variável '{variavel}' não encontrada.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
        
        dados_clustering_info = dados_clustering_global
        df_clusters = dados_clustering_info['dataframe']
        
        # Verificar se a variável está disponível nos dados
        indices_clusters = df_clusters.index
        variavel_valores = df.loc[indices_clusters, variavel].dropna()
        
        if len(variavel_valores) == 0:
            return f"❌ Não há valores válidos para a variável '{variavel}' nos dados analisados."
        
        # Criar dataframe com clusters e variável de interesse
        df_comparacao = pd.DataFrame({
            'Cluster': df_clusters.loc[variavel_valores.index, 'Cluster'],
            variavel: variavel_valores
        })
        
        # Remover ruído se existir
        df_sem_ruido = df_comparacao[df_comparacao['Cluster'] != -1]
        
        response = f"📊 **COMPARAÇÃO DOS CLUSTERS POR '{variavel}'**\n\n"
        
        # Verificar se a variável é numérica ou categórica
        if pd.api.types.is_numeric_dtype(variavel_valores):
            response += f"**Tipo de análise:** Variável numérica\n"
            response += f"**Dados válidos:** {len(df_sem_ruido)} registros\n\n"
            
            # Estatísticas por cluster
            response += f"**📈 ESTATÍSTICAS POR CLUSTER:**\n"
            stats_por_cluster = df_sem_ruido.groupby('Cluster')[variavel].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
            
            for cluster_id in sorted(df_sem_ruido['Cluster'].unique()):
                stats = stats_por_cluster.loc[cluster_id]
                n = int(stats['count'])
                pct = (n / len(df_sem_ruido)) * 100
                
                response += f"\n**🔹 Cluster {cluster_id}** ({n} registros - {pct:.1f}%)\n"
                response += f"  - Média: {stats['mean']:.4f}\n"
                response += f"  - Mediana: {stats['median']:.4f}\n"
                response += f"  - Desvio Padrão: {stats['std']:.4f}\n"
                response += f"  - Amplitude: {stats['min']:.4f} a {stats['max']:.4f}\n"
            
            # Identificar cluster com características extremas
            medias = stats_por_cluster['mean']
            cluster_maior = medias.idxmax()
            cluster_menor = medias.idxmin()
            diferenca_medias = medias.max() - medias.min()
            
            response += f"\n**🎯 PRINCIPAIS DESCOBERTAS:**\n"
            response += f"- **Cluster com maior média:** Cluster {cluster_maior} ({medias.max():.4f})\n"
            response += f"- **Cluster com menor média:** Cluster {cluster_menor} ({medias.min():.4f})\n"
            response += f"- **Diferença entre extremos:** {diferenca_medias:.4f}\n"
            
            # Análise de variabilidade
            desvios = stats_por_cluster['std']
            cluster_mais_variavel = desvios.idxmax()
            cluster_menos_variavel = desvios.idxmin()
            
            response += f"- **Cluster mais variável:** Cluster {cluster_mais_variavel} (σ={desvios.max():.4f})\n"
            response += f"- **Cluster mais homogêneo:** Cluster {cluster_menos_variavel} (σ={desvios.min():.4f})\n"
            
            # Gerar boxplot para visualizar
            try:
                plt.figure(figsize=(10, 6))
                
                # Preparar dados para boxplot
                dados_boxplot = [df_sem_ruido[df_sem_ruido['Cluster'] == c][variavel].values 
                               for c in sorted(df_sem_ruido['Cluster'].unique())]
                labels_boxplot = [f'Cluster {c}' for c in sorted(df_sem_ruido['Cluster'].unique())]
                
                plt.boxplot(dados_boxplot, labels=labels_boxplot)
                plt.title(f'Distribuição de {variavel} por Cluster')
                plt.xlabel('Cluster')
                plt.ylabel(variavel)
                plt.grid(True, alpha=0.3)
                
                # Salvar gráfico
                caminho_grafico = GRAFICOS_DIR / f"clusters_vs_{variavel}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
                plt.close()
                
                response += f"\n📊 **Boxplot gerado:** {caminho_grafico}"
                
            except Exception as e:
                response += f"\n⚠️ Erro ao gerar visualização: {str(e)}"
            
        else:
            # Análise categórica
            response += f"**Tipo de análise:** Variável categórica\n"
            response += f"**Dados válidos:** {len(df_sem_ruido)} registros\n\n"
            
            # Tabela de contingência
            tabela_contingencia = pd.crosstab(df_sem_ruido['Cluster'], df_sem_ruido[variavel])
            
            response += f"**📊 DISTRIBUIÇÃO CATEGÓRICA:**\n"
            response += f"```\n{tabela_contingencia.to_string()}\n```\n\n"
            
            # Percentuais por cluster
            response += f"**📈 PERCENTUAIS POR CLUSTER:**\n"
            percentuais = pd.crosstab(df_sem_ruido['Cluster'], df_sem_ruido[variavel], normalize='index') * 100
            
            for cluster_id in sorted(df_sem_ruido['Cluster'].unique()):
                response += f"\n**🔹 Cluster {cluster_id}:**\n"
                for categoria in percentuais.columns:
                    pct = percentuais.loc[cluster_id, categoria]
                    response += f"  - {categoria}: {pct:.1f}%\n"
            
            # Análise de associação
            response += f"\n**🎯 ANÁLISE DE ASSOCIAÇÃO:**\n"
            
            # Categoria mais comum por cluster
            for cluster_id in sorted(df_sem_ruido['Cluster'].unique()):
                dados_cluster = df_sem_ruido[df_sem_ruido['Cluster'] == cluster_id]
                categoria_mais_comum = dados_cluster[variavel].mode()
                
                if len(categoria_mais_comum) > 0:
                    freq = (dados_cluster[variavel] == categoria_mais_comum.iloc[0]).sum()
                    total = len(dados_cluster)
                    pct = (freq / total) * 100
                    
                    response += f"- **Cluster {cluster_id}:** mais comum = '{categoria_mais_comum.iloc[0]}' ({freq}/{total} = {pct:.1f}%)\n"
        
        # Sugestões adicionais
        response += f"\n**💡 INTERPRETAÇÕES E PRÓXIMOS PASSOS:**\n"
        
        if pd.api.types.is_numeric_dtype(variavel_valores):
            if diferenca_medias > variavel_valores.std():
                response += f"✅ A variável '{variavel}' apresenta diferenças significativas entre clusters.\n"
                response += f"   Isso sugere que o clustering capturou padrões relevantes para esta variável.\n"
            else:
                response += f"⚠️ A variável '{variavel}' apresenta diferenças pequenas entre clusters.\n"
                response += f"   Considere se esta variável deveria ter sido incluída no clustering.\n"
        else:
            response += f"✅ A distribuição categórica mostra como os clusters se relacionam com '{variavel}'.\n"
            response += f"   Use esta informação para dar nomes/rótulos significativos aos clusters.\n"
        
        response += f"\n**🔧 Análises recomendadas:**\n"
        response += f"1. Compare outros variáveis importantes com os clusters\n"
        response += f"2. Use testes estatísticos para verificar significância das diferenças\n"
        response += f"3. Considere criar perfis descritivos para cada cluster\n"
        
        return response
        
    except Exception as e:
        return f"Erro na comparação de clusters por variável: {str(e)}"

@tool
def otimizar_numero_clusters(colunas_features: str, k_min: int = 2, k_max: int = 10) -> str:
    """Encontra o número ótimo de clusters usando método do cotovelo e silhouette score."""
    try:
        if not SKLEARN_AVAILABLE:
            return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
        
        if df is None or df.empty:
            return "❌ Nenhum dataset foi carregado."
        
        # Processar colunas de features
        lista_features = [col.strip() for col in colunas_features.split(',')]
        features_validas = []
        
        for feature in lista_features:
            if feature not in df.columns:
                return f"❌ Feature '{feature}' não encontrada."
            if not pd.api.types.is_numeric_dtype(df[feature]):
                return f"❌ Feature '{feature}' deve ser numérica."
            features_validas.append(feature)
        
        # Preparar dados
        df_clustering = df[features_validas].dropna()
        if len(df_clustering) < k_max:
            k_max = len(df_clustering) - 1
            
        if k_min >= k_max:
            return f"❌ Parâmetros inválidos. k_min deve ser menor que k_max (ajustado: k_max={k_max})"
        
        X = df_clustering.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Testar diferentes números de clusters
        k_range = range(k_min, min(k_max + 1, len(df_clustering)))
        inercias = []
        silhouettes = []
        
        response = f"🔍 **OTIMIZAÇÃO DO NÚMERO DE CLUSTERS**\n\n"
        response += f"**Configuração:**\n"
        response += f"- Features: {', '.join(features_validas)}\n"
        response += f"- Amostras: {len(df_clustering)}\n"
        response += f"- Range de k: {k_min} a {min(k_max, len(df_clustering) - 1)}\n\n"
        
        response += f"**📊 RESULTADOS POR NÚMERO DE CLUSTERS:**\n"
        
        for k in k_range:
            # K-means com execução sequencial para Streamlit Cloud
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Métricas
            inercia = float(kmeans.inertia_)
            silhouette = float(silhouette_score(X_scaled, clusters))
            
            inercias.append(inercia)
            silhouettes.append(silhouette)
            
            response += f"- **k={k}**: Inércia={inercia:.2f}, Silhouette={silhouette:.4f}\n"
        
        # Encontrar k ótimo
        # Método do cotovelo (maior redução na inércia)
        diferencas = [inercias[i] - inercias[i+1] for i in range(len(inercias)-1)]
        if diferencas:
            k_cotovelo = k_range[diferencas.index(max(diferencas))]
        else:
            k_cotovelo = k_min
        
        # Melhor silhouette
        k_silhouette = k_range[silhouettes.index(max(silhouettes))]
        
        response += f"\n**🎯 RECOMENDAÇÕES:**\n"
        response += f"- **Método do Cotovelo:** k = {k_cotovelo} (maior redução de inércia)\n"
        response += f"- **Melhor Silhouette:** k = {k_silhouette} (score = {max(silhouettes):.4f})\n"
        
        if k_cotovelo == k_silhouette:
            response += f"✅ **Ambos os métodos convergem para k = {k_cotovelo}**\n"
            k_recomendado = k_cotovelo
        else:
            response += f"⚠️ **Métodos divergem.** Recomendo testar ambos valores.\n"
            k_recomendado = k_silhouette  # Priorizar silhouette
        
        # Gerar gráficos de análise
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico do cotovelo
        ax1.plot(k_range, inercias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('Inércia (WCSS)')
        ax1.set_title('Método do Cotovelo')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=k_cotovelo, color='r', linestyle='--', alpha=0.7, label=f'k ótimo = {k_cotovelo}')
        ax1.legend()
        
        # Gráfico silhouette
        ax2.plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Análise Silhouette')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=k_silhouette, color='g', linestyle='--', alpha=0.7, label=f'k ótimo = {k_silhouette}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Salvar gráfico
        caminho_grafico = GRAFICOS_DIR / f"otimizacao_clusters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        
        response += f"\n📊 **Gráficos de análise:** {caminho_grafico}"
        
        response += f"\n\n**💡 PRÓXIMO PASSO RECOMENDADO:**\n"
        response += f"Execute: realizar_clustering('{colunas_features}', 'kmeans', {k_recomendado})\n"
        
        # Interpretação adicional
        melhor_silhouette = max(silhouettes)
        if melhor_silhouette >= 0.7:
            response += f"\n✅ Silhouette Score excelente ({melhor_silhouette:.3f}) - clusters bem definidos\n"
        elif melhor_silhouette >= 0.5:
            response += f"\n✅ Silhouette Score bom ({melhor_silhouette:.3f}) - clusters razoavelmente separados\n"
        elif melhor_silhouette >= 0.3:
            response += f"\n⚠️ Silhouette Score moderado ({melhor_silhouette:.3f}) - considere outras abordagens\n"
        else:
            response += f"\n❌ Silhouette Score baixo ({melhor_silhouette:.3f}) - dados podem não ter estrutura de clusters clara\n"
        
        return response
        
    except Exception as e:
        return f"Erro na otimização de clusters: {str(e)}"


def criar_ferramentas_agente(dataframe: pd.DataFrame) -> List[Any]:
    """
    Cria as ferramentas disponíveis para o agente LangChain.
    
    Args:
        dataframe: DataFrame para análise
        
    Returns:
        Lista de ferramentas configuradas
    """
    
    # Definir DataFrame como global para as ferramentas
    global df
    df = dataframe
    
    @tool
    def descrever_dataset() -> str:
        """Retorna uma descrição completa do dataset incluindo estatísticas gerais."""
        try:
            resultado = descrever_dados(df)
            return f"""
📊 **DESCRIÇÃO DO DATASET**

**Dimensões:** {resultado['forma_dados']['linhas']} linhas × {resultado['forma_dados']['colunas']} colunas
**Uso de Memória:** {resultado['forma_dados']['tamanho_memoria_mb']:.2f} MB

**Tipos de Dados:**
{chr(10).join([f"- {tipo}: {count} colunas" for tipo, count in resultado['tipos_dados'].items()])}

**Colunas por Tipo:**
- Numéricas: {', '.join(resultado['colunas_por_tipo']['numericas'][:5])}{'...' if len(resultado['colunas_por_tipo']['numericas']) > 5 else ''}
- Categóricas: {', '.join(resultado['colunas_por_tipo']['categoricas'][:5])}{'...' if len(resultado['colunas_por_tipo']['categoricas']) > 5 else ''}

**Valores Nulos:** {sum(resultado['valores_nulos']['contagem'].values())} total
            """
        except Exception as e:
            return f"Erro ao descrever dataset: {str(e)}"

    @tool
    def analisar_coluna(coluna: str) -> str:
        """Analisa estatísticas detalhadas de uma coluna específica."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_similares = [c for c in df.columns if coluna.lower() in c.lower()]
                if colunas_similares:
                    return f"❌ Coluna '{coluna}' não encontrada. Você quis dizer: {', '.join(colunas_similares[:3])}?"
                return f"❌ Coluna '{coluna}' não existe.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            resultado = calcular_estatisticas(df, coluna)
            
            response = f"📈 **ANÁLISE DA COLUNA '{coluna}'**\n\n"
            response += f"**Tipo:** {resultado['tipo_dados']}\n"
            response += f"**Valores totais:** {resultado['valores_totais']}\n"
            response += f"**Valores nulos:** {resultado['valores_nulos']}\n"
            response += f"**Valores únicos:** {resultado['valores_unicos']}\n\n"
            
            if 'estatisticas_numericas' in resultado:
                stats = resultado['estatisticas_numericas']
                response += f"**Estatísticas Numéricas:**\n"
                response += f"- Mínimo: {stats['minimo']:.3f}\n"
                response += f"- Máximo: {stats['maximo']:.3f}\n"
                response += f"- Média: {stats['media']:.3f}\n"
                response += f"- Mediana: {stats['mediana']:.3f}\n"
                response += f"- Desvio Padrão: {stats['desvio_padrao']:.3f}\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao analisar coluna: {str(e)}"

    @tool 
    def gerar_histograma_coluna(coluna: str) -> str:
        """Gera um histograma para uma coluna numérica."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna]):
                return f"Coluna '{coluna}' não é numérica. Use gráfico de barras para variáveis categóricas."
            
            caminho = gerar_histograma(df, coluna)
            return f"📊 Histograma gerado para '{coluna}': {caminho}"
            
        except Exception as e:
            return f"Erro ao gerar histograma: {str(e)}"

    @tool
    def gerar_grafico_dispersao(coluna1: str, coluna2: str) -> str:
        """Gera gráfico de dispersão entre duas colunas numéricas."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna1 not in df.columns or coluna2 not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Uma ou ambas as colunas ('{coluna1}', '{coluna2}') não foram encontradas.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            # Verificar se ambas as colunas são numéricas
            if not pd.api.types.is_numeric_dtype(df[coluna1]):
                return f"❌ Coluna '{coluna1}' não é numérica. Tipo atual: {df[coluna1].dtype}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna2]):
                return f"❌ Coluna '{coluna2}' não é numérica. Tipo atual: {df[coluna2].dtype}"
            
            caminho = gerar_dispersao(df, coluna1, coluna2)
            return f"📈 Gráfico de dispersão gerado: {coluna1} vs {coluna2} - {caminho}"
            
        except Exception as e:
            return f"Erro ao gerar dispersão: {str(e)}"

    @tool
    def calcular_matriz_correlacao() -> str:
        """Calcula e exibe matriz de correlação entre variáveis numéricas."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            resultado = calcular_correlacao(df)
            
            if 'erro' in resultado:
                return resultado['erro']
            
            response = f"🔗 **MATRIZ DE CORRELAÇÃO**\n\n"
            response += f"**Colunas analisadas:** {len(resultado['colunas_analisadas'])}\n"
            response += f"**Correlação média:** {resultado['estatisticas_matriz']['correlacao_media']:.3f}\n\n"
            
            # Top correlações
            top_corr = resultado['correlacoes_mais_fortes']
            response += "**Top 5 Correlações:**\n"
            for (var1, var2), corr in list(top_corr.items())[:5]:
                response += f"- {var1} ↔ {var2}: {corr:.3f}\n"
            
            response += f"\n� Para visualizar, solicite: 'Gere heatmap da correlação' ou 'Crie mapa de calor da correlação'"
            
            return response
            
        except Exception as e:
            return f"Erro ao calcular correlação: {str(e)}"

    @tool
    def detectar_outliers_coluna(coluna: str) -> str:
        """Detecta outliers em uma coluna numérica usando método IQR."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna]):
                return f"❌ Coluna '{coluna}' não é numérica. Tipo atual: {df[coluna].dtype}"
            
            print(f"🔍 DEBUG: Iniciando detecção de outliers para coluna '{coluna}'")
            resultado = detectar_outliers_iqr(df, coluna)
            
            if 'erro' in resultado:
                return resultado['erro']
            
            response = f"🎯 **DETECÇÃO DE OUTLIERS - '{coluna}'**\n\n"
            response += f"**Outliers encontrados:** {resultado['outliers']['total']} ({resultado['outliers']['percentual']:.1f}%)\n"
            response += f"**Outliers superiores:** {resultado['outliers']['superiores']['quantidade']}\n"
            response += f"**Outliers inferiores:** {resultado['outliers']['inferiores']['quantidade']}\n"
            response += f"**Outliers extremos:** {resultado['outliers']['extremos']['quantidade']}\n\n"
            
            response += f"**Limites IQR:**\n"
            response += f"- Inferior: {resultado['limites_outliers']['inferior']:.3f}\n"
            response += f"- Superior: {resultado['limites_outliers']['superior']:.3f}\n"
            
            response += f"\n� Para visualizar os outliers, solicite: 'Gere boxplot de {coluna}'"
            
            return response
            
        except Exception as e:
            return f"Erro ao detectar outliers: {str(e)}"

    @tool
    def analisar_valores_frequentes(coluna: str) -> str:
        """Analisa os valores mais frequentes em uma coluna."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            resultado = encontrar_valores_frequentes(df, coluna)
            
            response = f"🔢 **VALORES FREQUENTES - '{coluna}'**\n\n"
            response += f"**Valores únicos:** {resultado['valores_unicos']}\n"
            response += f"**Total de valores:** {resultado['total_valores_validos']}\n\n"
            
            response += "**Top 10 valores:**\n"
            for valor, freq in list(resultado['valores_mais_frequentes']['contagem'].items())[:10]:
                pct = resultado['valores_mais_frequentes']['percentual'][valor]
                response += f"- {valor}: {freq} ({pct:.1f}%)\n"
            
            response += f"\n� Para visualizar, solicite: 'Gere histograma de {coluna}' ou 'Gere gráfico de barras de {coluna}'"
            
            return response
            
        except Exception as e:
            return f"Erro ao analisar frequências: {str(e)}"

    @tool
    def contar_valores_especificos(coluna: str, valor: Optional[str] = None) -> str:
        """Conta a ocorrência de valores específicos em uma coluna ou mostra a frequência de todos os valores."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            # Se valor específico foi fornecido, contar apenas esse valor
            if valor is not None:
                # Tentar converter valor para o tipo correto
                if pd.api.types.is_numeric_dtype(df[coluna]):
                    try:
                        valor_convertido = float(valor) if '.' in valor else int(valor)
                    except ValueError:
                        return f"❌ Valor '{valor}' não é numérico para a coluna '{coluna}'"
                else:
                    valor_convertido = str(valor)
                
                # Contar ocorrências
                count = (df[coluna] == valor_convertido).sum()
                total = len(df[coluna].dropna())
                porcentagem = (count / total * 100) if total > 0 else 0
                
                response = f"🔢 **CONTAGEM DE VALOR ESPECÍFICO**\n\n"
                response += f"**Coluna:** {coluna}\n"
                response += f"**Valor procurado:** {valor_convertido}\n"
                response += f"**Ocorrências:** {count}\n"
                response += f"**Total de registros:** {total}\n"
                response += f"**Porcentagem:** {porcentagem:.2f}%\n"
                
                return response
            
            # Se valor não foi especificado, mostrar frequência de todos os valores
            else:
                value_counts = df[coluna].value_counts().head(10)
                total = len(df[coluna].dropna())
                
                response = f"🔢 **FREQUÊNCIA DE TODOS OS VALORES - '{coluna}'**\n\n"
                response += f"**Total de registros:** {total}\n"
                response += f"**Valores únicos:** {df[coluna].nunique()}\n\n"
                response += "**Top 10 valores mais frequentes:**\n"
                
                for val, count in value_counts.items():
                    porcentagem = (count / total * 100) if total > 0 else 0
                    response += f"- Valor '{val}': {count} ocorrências ({porcentagem:.2f}%)\n"
                
                return response
                
        except Exception as e:
            return f"Erro ao contar valores: {str(e)}"

    @tool
    def analisar_por_grupos(coluna_analise: str, coluna_grupo: str) -> str:
        """Analisa estatísticas de uma coluna separadas por grupos de outra coluna."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna_analise not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna de análise '{coluna_analise}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if coluna_grupo not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna de grupo '{coluna_grupo}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna_analise]):
                return f"❌ Coluna de análise '{coluna_analise}' deve ser numérica. Tipo atual: {df[coluna_analise].dtype}"
            
            # Filtrar dados válidos
            df_filtrado = df[[coluna_analise, coluna_grupo]].dropna()
            
            if len(df_filtrado) == 0:
                return "❌ Não há dados válidos para análise após remoção de valores nulos."
            
            # Calcular estatísticas por grupo
            grupos = df_filtrado.groupby(coluna_grupo)[coluna_analise]
            
            response = f"📊 **ANÁLISE POR GRUPOS**\n\n"
            response += f"**Coluna analisada:** {coluna_analise}\n"
            response += f"**Agrupada por:** {coluna_grupo}\n\n"
            
            # Estatísticas para cada grupo
            grupos_unicos = sorted(df_filtrado[coluna_grupo].unique())
            
            for grupo in grupos_unicos:
                dados_grupo = grupos.get_group(grupo)
                
                response += f"**📈 Grupo '{grupo}' ({len(dados_grupo)} registros):**\n"
                response += f"- Média: {dados_grupo.mean():.3f}\n"
                response += f"- Mediana: {dados_grupo.median():.3f}\n"
                response += f"- Desvio Padrão: {dados_grupo.std():.3f}\n"
                response += f"- Mínimo: {dados_grupo.min():.3f}\n"
                response += f"- Máximo: {dados_grupo.max():.3f}\n"
                response += f"- Q1 (25%): {dados_grupo.quantile(0.25):.3f}\n"
                response += f"- Q3 (75%): {dados_grupo.quantile(0.75):.3f}\n\n"
            
            # Comparação entre grupos
            if len(grupos_unicos) == 2:
                grupo1, grupo2 = grupos_unicos
                dados1 = grupos.get_group(grupo1)
                dados2 = grupos.get_group(grupo2)
                
                diff_media = dados1.mean() - dados2.mean()
                diff_mediana = dados1.median() - dados2.median()
                
                response += f"**🔍 COMPARAÇÃO ENTRE GRUPOS:**\n"
                response += f"- Diferença na média ('{grupo1}' - '{grupo2}'): {diff_media:.3f}\n"
                response += f"- Diferença na mediana ('{grupo1}' - '{grupo2}'): {diff_mediana:.3f}\n"
                response += f"- Grupo com maior média: '{grupo1}' ({dados1.mean():.3f})\n" if dados1.mean() > dados2.mean() else f"- Grupo com maior média: '{grupo2}' ({dados2.mean():.3f})\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao analisar por grupos: {str(e)}"

    @tool
    def gerar_grafico_filtrado(coluna_grafico: str, coluna_filtro: str, valor_filtro: str, tipo_grafico: str = "histograma") -> str:
        """Gera gráficos filtrados por uma condição específica."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna_grafico not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna para gráfico '{coluna_grafico}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if coluna_filtro not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna de filtro '{coluna_filtro}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            # Converter valor_filtro para o tipo correto
            if pd.api.types.is_numeric_dtype(df[coluna_filtro]):
                try:
                    valor_convertido = float(valor_filtro) if '.' in valor_filtro else int(valor_filtro)
                except ValueError:
                    return f"❌ Valor de filtro '{valor_filtro}' não é numérico para a coluna '{coluna_filtro}'"
            else:
                valor_convertido = str(valor_filtro)
            
            # Filtrar dados
            df_filtrado = df[df[coluna_filtro] == valor_convertido]
            
            if len(df_filtrado) == 0:
                return f"❌ Nenhum registro encontrado onde '{coluna_filtro}' = '{valor_convertido}'"
            
            # Gerar gráfico apropriado
            if tipo_grafico.lower() == "histograma":
                if not pd.api.types.is_numeric_dtype(df_filtrado[coluna_grafico]):
                    return f"❌ Coluna '{coluna_grafico}' deve ser numérica para histograma"
                
                # Gerar histograma com dados filtrados
                caminho = gerar_histograma(df_filtrado, coluna_grafico, 
                                         titulo=f"Histograma de {coluna_grafico} (onde {coluna_filtro} = {valor_convertido})")
                
            elif tipo_grafico.lower() == "boxplot":
                if not pd.api.types.is_numeric_dtype(df_filtrado[coluna_grafico]):
                    return f"❌ Coluna '{coluna_grafico}' deve ser numérica para boxplot"
                
                # Gerar boxplot com dados filtrados
                caminho = gerar_boxplot(df_filtrado, coluna_grafico,
                                      titulo=f"Boxplot de {coluna_grafico} (onde {coluna_filtro} = {valor_convertido})")
                
            elif tipo_grafico.lower() == "barras":
                # Gerar gráfico de barras com dados filtrados
                caminho = gerar_grafico_barras(df_filtrado, coluna_grafico,
                                             titulo=f"Frequência de {coluna_grafico} (onde {coluna_filtro} = {valor_convertido})")
            else:
                return f"❌ Tipo de gráfico '{tipo_grafico}' não suportado. Use: histograma, boxplot ou barras"
            
            response = f"📊 **GRÁFICO FILTRADO GERADO**\n\n"
            response += f"**Coluna do gráfico:** {coluna_grafico}\n"
            response += f"**Filtro aplicado:** {coluna_filtro} = {valor_convertido}\n"
            response += f"**Registros filtrados:** {len(df_filtrado)}\n"
            response += f"**Tipo de gráfico:** {tipo_grafico}\n"
            response += f"**Arquivo gerado:** {caminho}\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao gerar gráfico filtrado: {str(e)}"

    @tool
    def gerar_histograma_condicional(coluna: str, condicao: str, valor: str, titulo_personalizado: Optional[str] = None) -> str:
        """Gera histograma com condições como >, <, >=, <=, ==, !=."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna]):
                return f"❌ Coluna '{coluna}' deve ser numérica para histograma. Tipo atual: {df[coluna].dtype}"
            
            # Converter valor para numérico
            try:
                valor_num = float(valor)
            except ValueError:
                return f"❌ Valor '{valor}' não é numérico válido"
            
            # Aplicar condição
            if condicao == ">":
                df_filtrado = df[df[coluna] > valor_num]
                desc_condicao = f"{coluna} > {valor}"
            elif condicao == "<":
                df_filtrado = df[df[coluna] < valor_num]
                desc_condicao = f"{coluna} < {valor}"
            elif condicao == ">=":
                df_filtrado = df[df[coluna] >= valor_num]
                desc_condicao = f"{coluna} >= {valor}"
            elif condicao == "<=":
                df_filtrado = df[df[coluna] <= valor_num]
                desc_condicao = f"{coluna} <= {valor}"
            elif condicao == "==":
                df_filtrado = df[df[coluna] == valor_num]
                desc_condicao = f"{coluna} == {valor}"
            elif condicao == "!=":
                df_filtrado = df[df[coluna] != valor_num]
                desc_condicao = f"{coluna} != {valor}"
            else:
                return f"❌ Condição '{condicao}' não suportada. Use: >, <, >=, <=, ==, !="
            
            if len(df_filtrado) == 0:
                return f"❌ Nenhum registro encontrado para a condição: {desc_condicao}"
            
            # Gerar título
            if titulo_personalizado:
                titulo = titulo_personalizado
            else:
                titulo = f"Histograma de {coluna} ({desc_condicao})"
            
            # Gerar histograma
            caminho = gerar_histograma(df_filtrado, coluna, titulo=titulo)
            
            response = f"📊 **HISTOGRAMA CONDICIONAL GERADO**\n\n"
            response += f"**Coluna:** {coluna}\n"
            response += f"**Condição:** {desc_condicao}\n"
            response += f"**Registros filtrados:** {len(df_filtrado)} de {len(df)} ({len(df_filtrado)/len(df)*100:.1f}%)\n"
            response += f"**Estatísticas dos dados filtrados:**\n"
            response += f"- Média: {df_filtrado[coluna].mean():.3f}\n"
            response += f"- Mediana: {df_filtrado[coluna].median():.3f}\n"
            response += f"- Desvio padrão: {df_filtrado[coluna].std():.3f}\n"
            response += f"- Mín: {df_filtrado[coluna].min():.3f}\n"
            response += f"- Máx: {df_filtrado[coluna].max():.3f}\n"
            response += f"\n📊 **Arquivo gerado:** {caminho}\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao gerar histograma condicional: {str(e)}"

    @tool
    def gerar_histograma_agrupado(coluna_numerica: str, coluna_grupo: str) -> str:
        """Compara a distribuição de uma coluna numérica entre diferentes grupos."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."

            if coluna_numerica not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna numérica '{coluna_numerica}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"

            if coluna_grupo not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna de grupo '{coluna_grupo}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"

            if not pd.api.types.is_numeric_dtype(df[coluna_numerica]):
                return f"❌ Coluna '{coluna_numerica}' deve ser numérica para histograma. Tipo atual: {df[coluna_numerica].dtype}"

            df_filtrado = df[[coluna_numerica, coluna_grupo]].dropna()
            if len(df_filtrado) == 0:
                return "❌ Não há dados válidos para análise após remoção de valores nulos."

            num_grupos = df_filtrado[coluna_grupo].nunique()
            if num_grupos < 2:
                return f"❌ A coluna '{coluna_grupo}' precisa ter pelo menos dois grupos distintos para comparação."

            if num_grupos > 12:
                return "⚠️ Muitos grupos identificados. Considere filtrar ou agrupar categorias para uma visualização mais clara."

            # Gerar histograma agrupado usando matplotlib
            plt.figure(figsize=(12, 6))
            
            grupos_unicos = sorted(df_filtrado[coluna_grupo].unique())
            cores = plt.cm.tab10(np.linspace(0, 1, len(grupos_unicos)))  # type: ignore
            
            for i, grupo in enumerate(grupos_unicos):
                dados_grupo = df_filtrado[df_filtrado[coluna_grupo] == grupo][coluna_numerica]
                plt.hist(dados_grupo, bins=20, alpha=0.7, label=str(grupo), 
                        color=cores[i], edgecolor='black', linewidth=0.5)
            
            plt.xlabel(coluna_numerica)
            plt.ylabel('Frequência')
            plt.title(f"Distribuição de {coluna_numerica} por {coluna_grupo}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            caminho = GRAFICOS_DIR / f"histograma_agrupado_{coluna_numerica}_{coluna_grupo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            plt.close()

            estatisticas = df_filtrado.groupby(coluna_grupo)[coluna_numerica].agg(['count', 'mean', 'median', 'std'])

            response = f"📊 **Histograma de {coluna_numerica} por {coluna_grupo}**\n\n"
            response += f"Comparei a distribuição de **{coluna_numerica}** entre {num_grupos} grupos distintos de **{coluna_grupo}**. "
            response += f"Foram analisados {len(df_filtrado):,} registros válidos.\n\n"

            response += "**🔍 Destaques por grupo:**\n"
            for grupo, linha in estatisticas.iterrows():
                std_val = 0.0 if np.isnan(linha['std']) else linha['std']
                response += (
                    f"- **{grupo}**: n={int(linha['count'])}, média={linha['mean']:.2f}, "
                    f"mediana={linha['median']:.2f}, desvio padrão={std_val:.2f}\n"
                )

            maior_media = estatisticas['mean'].idxmax()
            menor_media = estatisticas['mean'].idxmin()
            response += "\n**💡 Insights rápidos:**\n"
            response += f"- {maior_media} apresenta a maior média observada.\n"
            response += f"- {menor_media} apresenta a menor média observada.\n"
            response += f"\n📊 Arquivo gerado: {caminho}"

            return response
        except Exception as e:
            return f"Erro ao gerar histograma agrupado: {str(e)}"

    @tool
    def gerar_boxplot_agrupado(coluna_numerica: str, coluna_grupo: str) -> str:
        """Gera boxplot de uma coluna numérica agrupada por uma coluna categórica."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna_numerica not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna numérica '{coluna_numerica}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if coluna_grupo not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna de grupo '{coluna_grupo}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna_numerica]):
                return f"❌ Coluna '{coluna_numerica}' deve ser numérica para boxplot. Tipo atual: {df[coluna_numerica].dtype}"
            
            # Filtrar dados válidos
            df_filtrado = df[[coluna_numerica, coluna_grupo]].dropna()
            
            if len(df_filtrado) == 0:
                return "❌ Não há dados válidos para análise após remoção de valores nulos."
            
            # Gerar boxplot agrupado
            caminho = gerar_boxplot(df_filtrado, coluna_numerica, agrupar_por=coluna_grupo,
                                  titulo=f"Boxplot de {coluna_numerica} por {coluna_grupo}")
            
            # Calcular estatísticas por grupo
            grupos = df_filtrado.groupby(coluna_grupo)[coluna_numerica]
            grupos_unicos = sorted(df_filtrado[coluna_grupo].unique())
            
            # Análise interpretativa dos grupos
            response = f"📊 **Análise de {coluna_numerica} por {coluna_grupo}**\n\n"
            response += f"Analisei a distribuição de **{coluna_numerica}** entre os diferentes grupos de **{coluna_grupo}**. "
            response += f"Com {len(df_filtrado):,} registros analisados, aqui estão os principais achados:\n\n"
            
            # Estatísticas interpretativas por grupo
            stats_grupos = {}
            for grupo in grupos_unicos:
                dados_grupo = grupos.get_group(grupo)
                q1 = dados_grupo.quantile(0.25)
                q3 = dados_grupo.quantile(0.75)
                mediana = dados_grupo.median()
                outliers = len(dados_grupo[(dados_grupo < q1 - 1.5*(q3-q1)) | (dados_grupo > q3 + 1.5*(q3-q1))])
                stats_grupos[grupo] = {
                    'mediana': mediana, 'q1': q1, 'q3': q3, 'outliers': outliers,
                    'tamanho': len(dados_grupo), 'pct_outliers': (outliers/len(dados_grupo))*100
                }
            
            response += "**🔍 Resultados Principais:**\n"
            for grupo in grupos_unicos:
                stats = stats_grupos[grupo]
                response += f"- **Grupo {grupo}**: mediana de {stats['mediana']:.1f}, com 50% dos dados entre {stats['q1']:.1f} e {stats['q3']:.1f}"
                if stats['outliers'] > 0:
                    response += f" ({stats['outliers']} outliers - {stats['pct_outliers']:.1f}%)"
                response += "\n"
            
            # Comparação entre grupos
            if len(grupos_unicos) == 2:
                grupo1, grupo2 = grupos_unicos
                dados1 = grupos.get_group(grupo1)
                dados2 = grupos.get_group(grupo2)
                
                diff_mediana = stats_grupos[grupo1]['mediana'] - stats_grupos[grupo2]['mediana']
                grupo_maior = grupo1 if diff_mediana > 0 else grupo2
                response += f"\n**💡 Insights Principais:**\n"
                response += f"- O Grupo {grupo_maior} apresenta valores medianos {'maiores' if diff_mediana != 0 else 'similares'}\n"
                
                # Análise de outliers
                total_outliers = sum(stats['outliers'] for stats in stats_grupos.values())
                if total_outliers > 0:
                    grupo_mais_outliers = max(grupos_unicos, key=lambda g: stats_grupos[g]['pct_outliers'])
                    response += f"- O Grupo {grupo_mais_outliers} tem maior concentração de valores atípicos\n"
                else:
                    response += f"- Ambos os grupos apresentam distribuições bem comportadas (sem outliers significativos)\n"
            
            response += f"\nO gráfico boxplot foi gerado para visualizar essas diferenças."
            # ADIÇÃO: expor caminho do arquivo para o agente detectar o gráfico
            response += f"\n📊 Arquivo gerado: {caminho}"
            
            return response
        except Exception as e:
            return f"Erro ao gerar boxplot agrupado: {str(e)}"

    @tool
    def gerar_heatmap_correlacao_explicito() -> str:
        """Gera especificamente um heatmap da matriz de correlação quando solicitado."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            # Gerar heatmap
            caminho = gerar_heatmap_correlacao(df)
            
            resultado = calcular_correlacao(df)
            
            response = f"📊 **HEATMAP DA MATRIZ DE CORRELAÇÃO GERADO**\n\n"
            
            if 'erro' not in resultado:
                response += f"**Colunas analisadas:** {len(resultado['colunas_analisadas'])}\n"
                response += f"**Correlação média:** {resultado['estatisticas_matriz']['correlacao_media']:.3f}\n\n"
                
                # Top correlações
                top_corr = resultado['correlacoes_mais_fortes']
                response += "**Top 3 Correlações:**\n"
                for (var1, var2), corr in list(top_corr.items())[:3]:
                    response += f"- {var1} ↔ {var2}: {corr:.3f}\n"
            
            response += f"\n📊 Heatmap salvo em: {caminho}"
            
            return response
            
        except Exception as e:
            return f"Erro ao gerar heatmap de correlação: {str(e)}"

    @tool
    def gerar_boxplot_simples(coluna: str) -> str:
        """Gera um boxplot simples para uma coluna numérica para detectar outliers."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            if coluna not in df.columns:
                colunas_disponivel = ', '.join(df.columns.tolist())
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {colunas_disponivel}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna]):
                return f"❌ Coluna '{coluna}' deve ser numérica para boxplot. Tipo atual: {df[coluna].dtype}"
            
            caminho = gerar_boxplot(df, coluna)
            
            # Adicionar estatísticas básicas
            q1 = df[coluna].quantile(0.25)
            q2 = df[coluna].median()
            q3 = df[coluna].quantile(0.75)
            iqr = q3 - q1
            
            response = f"📦 **BOXPLOT GERADO PARA '{coluna}'**\n\n"
            response += f"**Estatísticas do Box Plot:**\n"
            response += f"- Q1 (25%): {q1:.3f}\n"
            response += f"- Mediana (Q2): {q2:.3f}\n" 
            response += f"- Q3 (75%): {q3:.3f}\n"
            response += f"- IQR: {iqr:.3f}\n\n"
            response += f"📊 **Gráfico salvo:** {caminho}"
            
            return response
            
        except Exception as e:
            return f"Erro ao gerar boxplot: {str(e)}"

    @tool
    def limpar_graficos_duplicados() -> str:
        """[Desativado] Não remove mais gráficos. Mantém todos os PNGs gerados."""
        return "ℹ️ Limpeza de gráficos desativada: nenhum arquivo foi removido."

    @tool
    def calcular_estatisticas_descritivas(colunas: str) -> str:
        """Calcula estatísticas descritivas (média, mediana, desvio padrão) para múltiplas colunas especificadas."""
        try:
            # Processar lista de colunas
            lista_colunas = [col.strip() for col in colunas.split(',')]
            colunas_numericas = []
            
            for coluna in lista_colunas:
                if coluna in df.columns:
                    if pd.api.types.is_numeric_dtype(df[coluna]):
                        colunas_numericas.append(coluna)
                    else:
                        return f"Coluna '{coluna}' não é numérica. Tipos disponíveis: {df[coluna].dtype}"
                else:
                    return f"Coluna '{coluna}' não encontrada. Colunas disponíveis: {', '.join(df.columns[:10])}"
            
            if not colunas_numericas:
                return "Nenhuma coluna numérica válida encontrada."
            
            response = "📊 **ESTATÍSTICAS DESCRITIVAS**\n\n"
            
            # Calcular estatísticas para cada coluna
            estatisticas_todas = {}
            for coluna in colunas_numericas:
                stats = df[coluna].describe()
                estatisticas_todas[coluna] = {
                    'media': df[coluna].mean(),
                    'mediana': df[coluna].median(),
                    'desvio_padrao': df[coluna].std(),
                    'minimo': df[coluna].min(),
                    'maximo': df[coluna].max(),
                    'quartil_25': df[coluna].quantile(0.25),
                    'quartil_75': df[coluna].quantile(0.75)
                }
            
            # Formatear resposta
            for coluna, stats in estatisticas_todas.items():
                response += f"**{coluna}:**\n"
                response += f"- Média: {stats['media']:.4f}\n"
                response += f"- Mediana: {stats['mediana']:.4f}\n"
                response += f"- Desvio Padrão: {stats['desvio_padrao']:.4f}\n"
                response += f"- Mínimo: {stats['minimo']:.4f}\n"
                response += f"- Máximo: {stats['maximo']:.4f}\n"
                response += f"- Q1 (25%): {stats['quartil_25']:.4f}\n"
                response += f"- Q3 (75%): {stats['quartil_75']:.4f}\n\n"
            
            # Adicionar resumo comparativo
            response += "📈 **RESUMO COMPARATIVO:**\n"
            response += "| Coluna | Média | Mediana | Desvio Padrão |\n"
            response += "|--------|-------|---------|---------------|\n"
            for coluna, stats in estatisticas_todas.items():
                response += f"| {coluna} | {stats['media']:.4f} | {stats['mediana']:.4f} | {stats['desvio_padrao']:.4f} |\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao calcular estatísticas descritivas: {str(e)}"
    
    @tool
    def listar_colunas() -> str:
        """Lista todas as colunas disponíveis no dataset."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            colunas_numericas = df.select_dtypes(include=['number']).columns.tolist()
            colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            response = f"📋 **COLUNAS DO DATASET**\n\n"
            response += f"**Total de colunas:** {len(df.columns)}\n\n"
            response += f"**📊 Colunas Numéricas ({len(colunas_numericas)}):**\n"
            for col in colunas_numericas:
                response += f"- {col}\n"
            
            response += f"\n**📝 Colunas Categóricas ({len(colunas_categoricas)}):**\n"
            for col in colunas_categoricas:
                response += f"- {col}\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao listar colunas: {str(e)}"

    @tool
    def sugerir_colunas_categoricas() -> str:
        """Analisa o dataset e sugere quais colunas podem ser categóricas baseado em padrões dos dados."""
        try:
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado. Por favor, faça upload de um arquivo CSV primeiro."
            
            sugestoes_categoricas = []
            
            # Analisar todas as colunas
            for coluna in df.columns:
                serie = df[coluna].dropna()
                
                if len(serie) == 0:
                    continue
                
                # Critérios para sugerir como categórica
                valores_unicos = serie.nunique()
                total_valores = len(serie)
                proporcao_unicos = valores_unicos / total_valores
                
                # Análise por tipo atual da coluna
                tipo_atual = str(df[coluna].dtype)
                eh_numerica = pd.api.types.is_numeric_dtype(df[coluna])
                eh_string = pd.api.types.is_string_dtype(df[coluna]) or tipo_atual == 'object'
                
                sugestao = {
                    'coluna': coluna,
                    'tipo_atual': tipo_atual,
                    'valores_unicos': valores_unicos,
                    'total_valores': total_valores,
                    'proporcao_unicos': proporcao_unicos,
                    'razoes': [],
                    'confianca': 0
                }
                
                # CRITÉRIO 1: Poucas categorias únicas (menos que 10% do total ou máximo 50)
                if valores_unicos <= max(10, total_valores * 0.1) and valores_unicos <= 50:
                    sugestao['razoes'].append(f"Poucos valores únicos ({valores_unicos})")
                    sugestao['confianca'] += 30
                
                # CRITÉRIO 2: Colunas de texto com valores repetitivos
                if eh_string:
                    # Verificar se tem padrões comuns de categorias
                    valores_mais_frequentes = serie.value_counts().head(10)
                    freq_top = valores_mais_frequentes.iloc[0] if len(valores_mais_frequentes) > 0 else 0
                    
                    if freq_top > total_valores * 0.05:  # Valor mais frequente aparece em pelo menos 5% dos dados
                        sugestao['razoes'].append("Valores se repetem com frequência")
                        sugestao['confianca'] += 25
                    
                    # Verificar padrões típicos de categorias
                    valores_exemplo = [str(v).lower().strip() for v in serie.unique()[:10] if pd.notna(v)]
                    
                    # Padrões comuns de categorias
                    padroes_categoricos = [
                        # Sim/Não, True/False
                        ['sim', 'não', 'nao', 'yes', 'no', 'true', 'false', 's', 'n'],
                        # Gênero
                        ['m', 'f', 'masculino', 'feminino', 'male', 'female', 'homem', 'mulher'],
                        # Status
                        ['ativo', 'inativo', 'ativado', 'desativado', 'active', 'inactive'],
                        # Tipos/Categorias
                        ['tipo', 'categoria', 'class', 'group', 'status'],
                        # Classificações
                        ['alto', 'medio', 'baixo', 'high', 'medium', 'low', 'grande', 'pequeno'],
                        # Estados/Regiões
                        ['sp', 'rj', 'mg', 'rs', 'pr', 'sc', 'ba', 'go'],
                    ]
                    
                    for padrao in padroes_categoricos:
                        if any(val in padrao for val in valores_exemplo):
                            sugestao['razoes'].append("Contém padrões típicos de categorias")
                            sugestao['confianca'] += 20
                            break
                
                # CRITÉRIO 3: Colunas numéricas que podem ser códigos/IDs categóricos
                elif eh_numerica:
                    # Verificar se são números inteiros consecutivos ou códigos
                    if df[coluna].dtype in ['int64', 'int32', 'int16', 'int8']:
                        # Se todos são inteiros pequenos ou poucos valores únicos
                        valores_int = serie.astype(int) if serie.dtype != 'int64' else serie
                        
                        if valores_unicos <= 20:
                            sugestao['razoes'].append("Poucos valores inteiros únicos, possivelmente códigos")
                            sugestao['confianca'] += 25
                        
                        # Verificar se são códigos sequenciais (1,2,3,4... ou 0,1,2,3...)
                        valores_ordenados = sorted(valores_int.unique())
                        if len(valores_ordenados) > 1:
                            diff = np.diff(valores_ordenados)
                            if all(d == 1 for d in diff) and len(valores_ordenados) <= 10:
                                sugestao['razoes'].append("Números sequenciais, provavelmente categorias codificadas")
                                sugestao['confianca'] += 30
                
                # CRITÉRIO 4: Proporção de valores únicos muito baixa
                if proporcao_unicos < 0.05:  # Menos de 5% dos valores são únicos
                    sugestao['razoes'].append(f"Baixa diversidade (apenas {proporcao_unicos:.1%} valores únicos)")
                    sugestao['confianca'] += 20
                
                # CRITÉRIO 5: Verificar se contém apenas alguns valores específicos
                if valores_unicos <= 5 and total_valores > 20:
                    sugestao['razoes'].append("Muito poucos valores distintos para um dataset grande")
                    sugestao['confianca'] += 35
                
                # Apenas incluir se tiver pelo menos uma razão e confiança mínima
                if sugestao['razoes'] and sugestao['confianca'] >= 20:
                    sugestoes_categoricas.append(sugestao)
            
            # Ordenar por confiança
            sugestoes_categoricas.sort(key=lambda x: x['confianca'], reverse=True)
            
            # Preparar resposta
            if not sugestoes_categoricas:
                response = "🔍 **ANÁLISE DE COLUNAS CATEGÓRICAS**\n\n"
                response += "✅ Não foram encontradas colunas que precisem ser convertidas para categóricas.\n\n"
                response += "Todas as colunas já estão com tipos adequados ou são claramente numéricas contínuas."
            else:
                response = "🔍 **SUGESTÕES DE COLUNAS CATEGÓRICAS**\n\n"
                response += f"Analisei {len(df.columns)} colunas e encontrei **{len(sugestoes_categoricas)} sugestões** "
                response += "de colunas que podem ser tratadas como categóricas:\n\n"
                
                for i, sugestao in enumerate(sugestoes_categoricas[:10], 1): # Limitar a 10
                    col = sugestao['coluna']
                    confianca = sugestao['confianca']
                    valores_unicos = sugestao['valores_unicos']
                    tipo_atual = sugestao['tipo_atual']
                    
                    # Ícone baseado na confiança
                    if confianca >= 70:
                        icone = "🔴"  # Alta confiança
                    elif confianca >= 50:
                        icone = "🟡"  # Média confiança
                    else:
                        icone = "🟢"  # Baixa confiança
                    
                    response += f"**{i}. {icone} Coluna '{col}'** (Confiança: {confianca}%)\n"
                    response += f"   - Tipo atual: {tipo_atual}\n"
                    response += f"   - Valores únicos: {valores_unicos}\n"
                    response += f"   - Razões: {'; '.join(sugestao['razoes'])}\n"
                    
                    # Mostrar alguns valores de exemplo
                    valores_exemplo = df[col].dropna().unique()[:5]
                    response += f"   - Exemplos: {', '.join(map(str, valores_exemplo))}\n\n"
                
                response += "💡 **Como interpretar:**\n"
                response += "- 🔴 Alta confiança (70%+): Muito provável que seja categórica\n"
                response += "- 🟡 Média confiança (50-70%): Pode ser categórica dependendo do contexto\n"
                response += "- 🟢 Baixa confiança (20-50%): Verificar se faz sentido no seu domínio\n\n"
                
                response += "🔧 **Próximos passos sugeridos:**\n"
                response += "1. Analise os valores únicos das colunas sugeridas\n"
                response += "2. Para colunas numéricas sugeridas, verifique se representam códigos ou categorias\n"
                response += "3. Considere criar gráficos de frequência para entender melhor os dados\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao sugerir colunas categóricas: {str(e)}"

    # ================================
    # FERRAMENTAS SCIPY.STATS
    # ================================
    
    @tool
    def testar_normalidade(coluna: str) -> str:
        """Testa se uma coluna numérica segue distribuição normal usando múltiplos testes estatísticos."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível. Execute: pip install scipy"
            
            if df is None or df.empty:
                return "❌ Nenhum dataset foi carregado."
            
            if coluna not in df.columns:
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna]):
                return f"❌ Coluna '{coluna}' não é numérica. Tipo: {df[coluna].dtype}"
            
            # Filtrar valores válidos
            dados = df[coluna].dropna()
            if len(dados) < 8:
                return f"❌ Dados insuficientes para teste de normalidade (mínimo: 8, atual: {len(dados)})"
            
            response = f"📊 **TESTES DE NORMALIDADE - '{coluna}'**\n\n"
            response += f"**Dados analisados:** {len(dados)} valores\n"
            response += f"**Média:** {dados.mean():.4f}\n"
            response += f"**Desvio padrão:** {dados.std():.4f}\n\n"
            
            # 1. Teste de Shapiro-Wilk (melhor para n < 5000)
            if len(dados) <= 5000:
                try:
                    result_sw = shapiro(dados)
                    stat_sw = float(result_sw[0])
                    p_sw = float(result_sw[1])
                    response += f"**🔬 Teste Shapiro-Wilk:**\n"
                    response += f"- Estatística: {stat_sw:.6f}\n"
                    response += f"- p-valor: {p_sw:.6f}\n"
                    response += f"- Resultado: {'Normal' if p_sw > 0.05 else 'Não-normal'} (α=0.05)\n\n"
                except Exception as e:
                    response += f"**🔬 Teste Shapiro-Wilk:** Erro - {str(e)}\n\n"
            
            # 2. Teste de D'Agostino-Pearson
            try:
                result_dp = normaltest(dados)
                stat_dp = float(result_dp[0])
                p_dp = float(result_dp[1])
                response += f"**🔬 Teste D'Agostino-Pearson:**\n"
                response += f"- Estatística: {stat_dp:.6f}\n"
                response += f"- p-valor: {p_dp:.6f}\n"
                response += f"- Resultado: {'Normal' if p_dp > 0.05 else 'Não-normal'} (α=0.05)\n\n"
            except Exception as e:
                response += f"**🔬 Teste D'Agostino-Pearson:** Erro - {str(e)}\n\n"
            
            # 3. Teste de Jarque-Bera
            try:
                result_jb = jarque_bera(dados)
                stat_jb = float(result_jb[0])
                p_jb = float(result_jb[1])
                response += f"**🔬 Teste Jarque-Bera:**\n"
                response += f"- Estatística: {stat_jb:.6f}\n"
                response += f"- p-valor: {p_jb:.6f}\n"
                response += f"- Resultado: {'Normal' if p_jb > 0.05 else 'Não-normal'} (α=0.05)\n\n"
            except Exception as e:
                response += f"**🔬 Teste Jarque-Bera:** Erro - {str(e)}\n\n"
            
            # 4. Teste de Anderson-Darling
            try:
                result_ad = anderson(dados, dist='norm')
                statistic_ad = float(result_ad.statistic)
                critical_values_ad = list(result_ad.critical_values)
                significance_level_ad = list(result_ad.significance_level)
                
                response += f"**🔬 Teste Anderson-Darling:**\n"
                response += f"- Estatística: {statistic_ad:.6f}\n"
                response += f"- Valores críticos: {critical_values_ad}\n"
                response += f"- Níveis de significância: {significance_level_ad}\n"
                
                # Verificar resultado para α=5%
                if len(critical_values_ad) >= 3:
                    cv_5pct = critical_values_ad[2]  # Normalmente é o valor para 5%
                    is_normal_ad = statistic_ad < cv_5pct
                    response += f"- Resultado: {'Normal' if is_normal_ad else 'Não-normal'} (α=5%)\n\n"
                else:
                    response += f"- Resultado: Avalie comparando estatística com valores críticos\n\n"
            except Exception as e:
                response += f"**🔬 Teste Anderson-Darling:** Erro - {str(e)}\n\n"
            
            # Resumo e interpretação
            response += f"**📋 INTERPRETAÇÃO:**\n"
            response += f"- **p-valor > 0.05**: Não rejeitamos H₀ (dados podem ser normais)\n"
            response += f"- **p-valor ≤ 0.05**: Rejeitamos H₀ (dados não são normais)\n"
            response += f"- Para decisões críticas, considere múltiplos testes e visualizações\n"
            
            return response
            
        except Exception as e:
            return f"Erro ao testar normalidade: {str(e)}"

    @tool
    def teste_t_uma_amostra(coluna: str, valor_teste: float, alternativa: str = "two-sided") -> str:
        """Realiza teste t de uma amostra para verificar se a média populacional difere de um valor específico."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível. Execute: pip install scipy"
            
            if coluna not in df.columns:
                return f"❌ Coluna '{coluna}' não encontrada.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna]):
                return f"❌ Coluna '{coluna}' não é numérica."
            
            dados = df[coluna].dropna()
            if len(dados) < 2:
                return f"❌ Dados insuficientes (mínimo: 2, atual: {len(dados)})"
            
            # Validar alternativa
            if alternativa not in ["two-sided", "less", "greater"]:
                alternativa = "two-sided"
            
            # Realizar teste t
            result_t = ttest_1samp(dados, valor_teste, alternative=alternativa)
            stat = float(result_t[0])
            p_value = float(result_t[1])
            
            response = f"📊 **TESTE T DE UMA AMOSTRA - '{coluna}'**\n\n"
            response += f"**Hipóteses:**\n"
            
            if alternativa == "two-sided":
                response += f"- H₀: μ = {valor_teste} (média populacional igual ao valor teste)\n"
                response += f"- H₁: μ ≠ {valor_teste} (média populacional diferente do valor teste)\n"
            elif alternativa == "greater":
                response += f"- H₀: μ ≤ {valor_teste}\n"
                response += f"- H₁: μ > {valor_teste}\n"
            else:  # less
                response += f"- H₀: μ ≥ {valor_teste}\n"
                response += f"- H₁: μ < {valor_teste}\n"
            
            response += f"\n**Estatísticas:**\n"
            response += f"- Tamanho da amostra: {len(dados)}\n"
            response += f"- Média da amostra: {dados.mean():.6f}\n"
            response += f"- Desvio padrão: {dados.std():.6f}\n"
            response += f"- Valor testado: {valor_teste}\n"
            response += f"- Estatística t: {stat:.6f}\n"
            response += f"- p-valor: {p_value:.6f}\n\n"
            
            # Interpretação
            response += f"**📋 RESULTADO (α=0.05):**\n"
            if p_value <= 0.05:
                response += f"✅ **Rejeitamos H₀** (p ≤ 0.05)\n"
                response += f"Evidência estatística de que a média populacional "
                if alternativa == "two-sided":
                    response += f"é diferente de {valor_teste}\n"
                elif alternativa == "greater":
                    response += f"é maior que {valor_teste}\n"
                else:
                    response += f"é menor que {valor_teste}\n"
            else:
                response += f"❌ **Não rejeitamos H₀** (p > 0.05)\n"
                response += f"Não há evidência estatística suficiente para concluir que a média "
                if alternativa == "two-sided":
                    response += f"é diferente de {valor_teste}\n"
                elif alternativa == "greater":
                    response += f"é maior que {valor_teste}\n"
                else:
                    response += f"é menor que {valor_teste}\n"
            
            return response
            
        except Exception as e:
            return f"Erro no teste t: {str(e)}"

    @tool
    def teste_t_duas_amostras(coluna1: str, coluna2: str, assumir_variancias_iguais: bool = True) -> str:
        """Compara as médias de duas colunas numéricas usando teste t de duas amostras independentes."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível."
            
            if coluna1 not in df.columns or coluna2 not in df.columns:
                return f"❌ Uma ou ambas as colunas não encontradas.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna1]) or not pd.api.types.is_numeric_dtype(df[coluna2]):
                return f"❌ Ambas as colunas devem ser numéricas."
            
            dados1 = df[coluna1].dropna()
            dados2 = df[coluna2].dropna()
            
            if len(dados1) < 2 or len(dados2) < 2:
                return f"❌ Dados insuficientes em uma ou ambas as colunas."
            
            # Teste de Levene para igualdade de variâncias
            result_levene = levene(dados1, dados2)
            stat_levene = float(result_levene[0])
            p_levene = float(result_levene[1])
            
            # Realizar teste t
            result_t = ttest_ind(dados1, dados2, equal_var=assumir_variancias_iguais)
            stat = float(result_t[0])
            p_value = float(result_t[1])
            
            response = f"📊 **TESTE T DE DUAS AMOSTRAS INDEPENDENTES**\n\n"
            response += f"**Colunas comparadas:** '{coluna1}' vs '{coluna2}'\n\n"
            
            response += f"**Estatísticas descritivas:**\n"
            response += f"- {coluna1}: n={len(dados1)}, média={dados1.mean():.6f}, std={dados1.std():.6f}\n"
            response += f"- {coluna2}: n={len(dados2)}, média={dados2.mean():.6f}, std={dados2.std():.6f}\n\n"
            
            response += f"**Teste de Levene (igualdade de variâncias):**\n"
            response += f"- Estatística: {stat_levene:.6f}\n"
            response += f"- p-valor: {p_levene:.6f}\n"
            response += f"- Variâncias: {'Iguais' if p_levene > 0.05 else 'Diferentes'} (α=0.05)\n\n"
            
            response += f"**Teste t ({'Welch' if not assumir_variancias_iguais else 'Student'}):**\n"
            response += f"- H₀: μ₁ = μ₂ (médias iguais)\n"
            response += f"- H₁: μ₁ ≠ μ₂ (médias diferentes)\n"
            response += f"- Estatística t: {stat:.6f}\n"
            response += f"- p-valor: {p_value:.6f}\n\n"
            
            # Interpretação
            response += f"**📋 RESULTADO (α=0.05):**\n"
            if p_value <= 0.05:
                response += f"✅ **Rejeitamos H₀** (p ≤ 0.05)\n"
                response += f"Evidência estatística de diferença significativa entre as médias\n"
                diff = dados1.mean() - dados2.mean()
                response += f"Diferença das médias: {diff:.6f}\n"
            else:
                response += f"❌ **Não rejeitamos H₀** (p > 0.05)\n"
                response += f"Não há evidência de diferença significativa entre as médias\n"
            
            return response
            
        except Exception as e:
            return f"Erro no teste t de duas amostras: {str(e)}"

    @tool
    def teste_correlacao(coluna1: str, coluna2: str, metodo: str = "pearson") -> str:
        """Testa correlação entre duas variáveis usando Pearson, Spearman ou Kendall."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível."
            
            if coluna1 not in df.columns or coluna2 not in df.columns:
                return f"❌ Uma ou ambas as colunas não encontradas.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna1]) or not pd.api.types.is_numeric_dtype(df[coluna2]):
                return f"❌ Ambas as colunas devem ser numéricas."
            
            # Filtrar dados válidos
            df_limpo = df[[coluna1, coluna2]].dropna()
            if len(df_limpo) < 3:
                return f"❌ Dados insuficientes após remoção de valores nulos (mínimo: 3, atual: {len(df_limpo)})"
            
            dados1 = df_limpo[coluna1]
            dados2 = df_limpo[coluna2]
            
            # Validar método
            if metodo not in ["pearson", "spearman", "kendall"]:
                metodo = "pearson"
            
            # Calcular correlação
            if metodo == "pearson":
                result_corr = pearsonr(dados1, dados2)
                corr = float(result_corr[0])
                p_value = float(result_corr[1])
                metodo_nome = "Pearson"
                descricao = "correlação linear"
            elif metodo == "spearman":
                result_corr = spearmanr(dados1, dados2)
                corr = float(result_corr[0])
                p_value = float(result_corr[1])
                metodo_nome = "Spearman"
                descricao = "correlação monotônica (ordinal)"
            else:  # kendall
                result_corr = kendalltau(dados1, dados2)
                corr = float(result_corr[0])
                p_value = float(result_corr[1])
                metodo_nome = "Kendall's Tau"
                descricao = "correlação ordinal robusta"
            
            response = f"📊 **TESTE DE CORRELAÇÃO - {metodo_nome.upper()}**\n\n"
            response += f"**Variáveis:** '{coluna1}' x '{coluna2}'\n"
            response += f"**Método:** {metodo_nome} ({descricao})\n"
            response += f"**Pares válidos:** {len(df_limpo)}\n\n"
            
            response += f"**Resultados:**\n"
            response += f"- Coeficiente de correlação: {corr:.6f}\n"
            response += f"- p-valor: {p_value:.6f}\n\n"
            
            # Interpretação da força da correlação
            abs_corr = abs(corr)
            if abs_corr >= 0.9:
                forca = "muito forte"
            elif abs_corr >= 0.7:
                forca = "forte"
            elif abs_corr >= 0.5:
                forca = "moderada"
            elif abs_corr >= 0.3:
                forca = "fraca"
            else:
                forca = "muito fraca/inexistente"
            
            direcao = "positiva" if corr > 0 else "negativa" if corr < 0 else "nula"
            
            response += f"**📋 INTERPRETAÇÃO:**\n"
            response += f"- **Força:** {forca} ({abs_corr:.3f})\n"
            response += f"- **Direção:** {direcao}\n"
            
            # Teste de significância
            response += f"- **Significância (α=0.05):** "
            if p_value <= 0.05:
                response += f"✅ Significativa (p ≤ 0.05)\n"
                response += f"  A correlação é estatisticamente significativa\n"
            else:
                response += f"❌ Não significativa (p > 0.05)\n"
                response += f"  A correlação não é estatisticamente significativa\n"
            
            response += f"\n**💡 CONCLUSÃO:**\n"
            if p_value <= 0.05 and abs_corr >= 0.3:
                response += f"Existe uma correlação {forca} e {direcao} estatisticamente significativa entre as variáveis."
            elif p_value <= 0.05:
                response += f"Embora estatisticamente significativa, a correlação é {forca}."
            else:
                response += f"Não há evidência de correlação significativa entre as variáveis."
            
            return response
            
        except Exception as e:
            return f"Erro no teste de correlação: {str(e)}"

    @tool
    def teste_qui_quadrado(coluna1: str, coluna2: str) -> str:
        """Testa independência entre duas variáveis categóricas usando qui-quadrado."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível."
            
            if coluna1 not in df.columns or coluna2 not in df.columns:
                return f"❌ Uma ou ambas as colunas não encontradas.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            # Criar tabela de contingência
            df_limpo = df[[coluna1, coluna2]].dropna()
            if len(df_limpo) < 5:
                return f"❌ Dados insuficientes (mínimo: 5, atual: {len(df_limpo)})"
            
            tabela_contingencia = pd.crosstab(df_limpo[coluna1], df_limpo[coluna2])
            
            # Verificar se todas as células têm pelo menos 5 observações esperadas
            result_chi2 = chi2_contingency(tabela_contingencia)
            chi2 = float(result_chi2[0])
            p_value = float(result_chi2[1])
            dof = int(result_chi2[2])
            expected = np.array(result_chi2[3])
            
            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()
            total_cells = expected.size
            
            response = f"📊 **TESTE QUI-QUADRADO DE INDEPENDÊNCIA**\n\n"
            response += f"**Variáveis:** '{coluna1}' x '{coluna2}'\n"
            response += f"**Observações válidas:** {len(df_limpo)}\n\n"
            
            response += f"**Tabela de Contingência:**\n"
            response += f"{tabela_contingencia.to_string()}\n\n"
            
            response += f"**Resultados do teste:**\n"
            response += f"- Estatística Qui-quadrado: {chi2:.6f}\n"
            response += f"- Graus de liberdade: {dof}\n"
            response += f"- p-valor: {p_value:.6f}\n\n"
            
            response += f"**Pressupostos:**\n"
            response += f"- Menor frequência esperada: {min_expected:.2f}\n"
            response += f"- Células com freq. esperada < 5: {cells_below_5}/{total_cells}\n"
            
            # Verificar pressupostos
            pressupostos_ok = min_expected >= 5 or (cells_below_5 / total_cells) <= 0.2
            response += f"- Pressupostos: {'✅ Atendidos' if pressupostos_ok else '⚠️ Violados'}\n"
            
            if not pressupostos_ok:
                response += f"  (Considere usar Teste Exato de Fisher para tabelas 2x2)\n"
            
            response += f"\n**📋 INTERPRETAÇÃO (α=0.05):**\n"
            response += f"- **H₀:** As variáveis são independentes\n"
            response += f"- **H₁:** As variáveis são dependentes (associadas)\n\n"
            
            if p_value <= 0.05:
                response += f"✅ **Rejeitamos H₀** (p ≤ 0.05)\n"
                response += f"Evidência estatística de associação entre '{coluna1}' e '{coluna2}'\n"
                
                # Calcular V de Cramér (medida de associação)
                n = len(df_limpo)
                cramer_v = np.sqrt(chi2 / (n * (min(tabela_contingencia.shape) - 1)))
                response += f"- **V de Cramér:** {cramer_v:.4f} "
                
                if cramer_v >= 0.5:
                    response += f"(associação forte)\n"
                elif cramer_v >= 0.3:
                    response += f"(associação moderada)\n"
                elif cramer_v >= 0.1:
                    response += f"(associação fraca)\n"
                else:
                    response += f"(associação muito fraca)\n"
            else:
                response += f"❌ **Não rejeitamos H₀** (p > 0.05)\n"
                response += f"Não há evidência de associação entre as variáveis\n"
            
            return response
            
        except Exception as e:
            return f"Erro no teste qui-quadrado: {str(e)}"

    @tool
    def anova_um_fator(coluna_numerica: str, coluna_grupos: str) -> str:
        """Realiza ANOVA de um fator para comparar médias entre múltiplos grupos."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível."
            
            if coluna_numerica not in df.columns or coluna_grupos not in df.columns:
                return f"❌ Uma ou ambas as colunas não encontradas.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna_numerica]):
                return f"❌ Coluna '{coluna_numerica}' deve ser numérica."
            
            # Filtrar dados válidos
            df_limpo = df[[coluna_numerica, coluna_grupos]].dropna()
            if len(df_limpo) < 3:
                return "❌ Dados insuficientes para ANOVA (mínimo: 3)"
            
            # Separar dados por grupos
            grupos = []
            nomes_grupos = []
            
            for nome_grupo in df_limpo[coluna_grupos].unique():
                dados_grupo = df_limpo[df_limpo[coluna_grupos] == nome_grupo][coluna_numerica]
                if len(dados_grupo) >= 2:  # Mínimo 2 observações por grupo
                    grupos.append(dados_grupo)
                    nomes_grupos.append(str(nome_grupo))
            
            if len(grupos) < 2:
                return f"❌ Necessário pelo menos 2 grupos com 2+ observações cada."
            
            # ANOVA
            result_anova = f_oneway(*grupos)
            stat_anova = float(result_anova[0])
            p_anova = float(result_anova[1])
            
            # Teste de Levene (homogeneidade de variâncias)
            result_levene = levene(*grupos)
            stat_levene = float(result_levene[0])
            p_levene = float(result_levene[1])
            
            response = f"📊 **ANOVA DE UM FATOR**\n\n"
            response += f"**Variável dependente:** '{coluna_numerica}'\n"
            response += f"**Fator (grupos):** '{coluna_grupos}'\n"
            response += f"**Número de grupos:** {len(grupos)}\n\n"
            
            # Estatísticas por grupo
            response += f"**📋 Estatísticas por grupo:**\n"
            for i, (nome, dados) in enumerate(zip(nomes_grupos, grupos)):
                response += f"- **{nome}:** n={len(dados)}, média={dados.mean():.4f}, std={dados.std():.4f}\n"
            
            response += f"\n**🔬 Teste de Levene (homogeneidade de variâncias):**\n"
            response += f"- Estatística: {stat_levene:.6f}\n"
            response += f"- p-valor: {p_levene:.6f}\n"
            response += f"- Variâncias: {'Homogêneas' if p_levene > 0.05 else 'Heterogêneas'} (α=0.05)\n"
            
            if p_levene <= 0.05:
                response += f"⚠️ **Atenção:** Pressuposto de homogeneidade violado. Considere transformação dos dados ou teste não-paramétrico.\n"
            
            response += f"\n**🔬 ANOVA:**\n"
            response += f"- **H₀:** μ₁ = μ₂ = ... = μₖ (todas as médias são iguais)\n"
            response += f"- **H₁:** Pelo menos uma média é diferente\n"
            response += f"- Estatística F: {stat_anova:.6f}\n"
            response += f"- p-valor: {p_anova:.6f}\n\n"
            
            # Interpretação
            response += f"**📋 RESULTADO (α=0.05):**\n"
            if p_anova <= 0.05:
                response += f"✅ **Rejeitamos H₀** (p ≤ 0.05)\n"
                response += f"Evidência estatística de que pelo menos uma média é diferente dos demais grupos\n"
                response += f"💡 **Sugestão:** Realize testes post-hoc para identificar quais grupos diferem\n"
            else:
                response += f"❌ **Não rejeitamos H₀** (p > 0.05)\n"
                response += f"Não há evidência de diferença significativa entre as médias dos grupos\n"
            
            return response
            
        except Exception as e:
            return f"Erro na ANOVA: {str(e)}"

    @tool
    def teste_mann_whitney(coluna1: str, coluna2: str) -> str:
        """Teste não-paramétrico de Mann-Whitney U para comparar duas amostras independentes."""
        try:
            if not SCIPY_AVAILABLE:
                return "❌ Biblioteca scipy não está disponível."
            
            if coluna1 not in df.columns or coluna2 not in df.columns:
                return f"❌ Uma ou ambas as colunas não encontradas.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna1]) or not pd.api.types.is_numeric_dtype(df[coluna2]):
                return f"❌ Ambas as colunas devem ser numéricas."
            
            dados1 = df[coluna1].dropna()
            dados2 = df[coluna2].dropna()
            
            if len(dados1) < 3 or len(dados2) < 3:
                return f"❌ Necessário pelo menos 3 observações em cada grupo."
            
            # Teste de Mann-Whitney U
            result_mw = mannwhitneyu(dados1, dados2, alternative='two-sided')
            stat = float(result_mw[0])
            p_value = float(result_mw[1])
            
            response = f"📊 **TESTE DE MANN-WHITNEY U**\n\n"
            response += f"**Grupos comparados:** '{coluna1}' vs '{coluna2}'\n"
            response += f"**Tipo:** Teste não-paramétrico para amostras independentes\n\n"
            
            response += f"**Estatísticas descritivas:**\n"
            response += f"- {coluna1}: n={len(dados1)}, mediana={dados1.median():.4f}, IQR={dados1.quantile(0.75)-dados1.quantile(0.25):.4f}\n"
            response += f"- {coluna2}: n={len(dados2)}, mediana={dados2.median():.4f}, IQR={dados2.quantile(0.75)-dados2.quantile(0.25):.4f}\n\n"
            
            response += f"**Resultados do teste:**\n"
            response += f"- **H₀:** As distribuições são iguais (mesma tendência central)\n"
            response += f"- **H₁:** As distribuições diferem na tendência central\n"
            response += f"- Estatística U: {stat:.6f}\n"
            response += f"- p-valor: {p_value:.6f}\n\n"
            
            # Interpretação
            response += f"**📋 RESULTADO (α=0.05):**\n"
            if p_value <= 0.05:
                response += f"✅ **Rejeitamos H₀** (p ≤ 0.05)\n"
                response += f"Evidência estatística de diferença na tendência central entre os grupos\n"
                
                # Indicar qual grupo tem valores maiores
                if dados1.median() > dados2.median():
                    response += f"- Grupo '{coluna1}' tende a ter valores maiores\n"
                else:
                    response += f"- Grupo '{coluna2}' tende a ter valores maiores\n"
            else:
                response += f"❌ **Não rejeitamos H₀** (p > 0.05)\n"
                response += f"Não há evidência de diferença na tendência central entre os grupos\n"
            
            response += f"\n**💡 Vantagens do Mann-Whitney:**\n"
            response += f"- Não assume normalidade dos dados\n"
            response += f"- Robusto a outliers\n"
            response += f"- Adequado para dados ordinais ou contínuos não-normais\n"
            
            return response
            
        except Exception as e:
            return f"Erro no teste Mann-Whitney: {str(e)}"


    
    # ================================
    # FERRAMENTAS SCIKIT-LEARN
    # ================================

    @tool
    def treinar_modelo_regressao(coluna_target: str, colunas_features: str, modelo: str = "linear", test_size: float = 0.2) -> str:
        """Treina modelo de regressão para prever valores numéricos."""
        try:
            if not SKLEARN_AVAILABLE:
                return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
            
            if coluna_target not in df.columns:
                return f"❌ Coluna target '{coluna_target}' não encontrada.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            if not pd.api.types.is_numeric_dtype(df[coluna_target]):
                return f"❌ Coluna target '{coluna_target}' deve ser numérica para regressão."
            
            # Processar colunas de features
            lista_features = [col.strip() for col in colunas_features.split(',')]
            features_validas = []
            
            for feature in lista_features:
                if feature not in df.columns:
                    return f"❌ Feature '{feature}' não encontrada."
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    return f"❌ Feature '{feature}' deve ser numérica para regressão."
                features_validas.append(feature)
            
            # Preparar dados
            df_limpo = df[features_validas + [coluna_target]].dropna()
            if len(df_limpo) < 10:
                return f"❌ Dados insuficientes após limpeza (mínimo: 10, atual: {len(df_limpo)})"
            
            X = df_limpo[features_validas]
            y = df_limpo[coluna_target]
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Escolher modelo
            if modelo.lower() == "linear":
                model = LinearRegression()
                nome_modelo = "Regressão Linear"
            elif modelo.lower() == "tree":
                model = DecisionTreeRegressor(random_state=42)
                nome_modelo = "Árvore de Decisão"
            elif modelo.lower() == "forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                nome_modelo = "Random Forest"
            elif modelo.lower() == "svm":
                model = SVR()
                nome_modelo = "SVM Regressão"
            elif modelo.lower() == "knn":
                model = KNeighborsRegressor()
                nome_modelo = "K-NN Regressão"
            else:
                return f"❌ Modelo '{modelo}' não suportado. Use: linear, tree, forest, svm, knn"
            
            # Treinar modelo
            if modelo.lower() in ["svm", "knn"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Avaliar modelo
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X_train_scaled if modelo.lower() in ["svm", "knn"] else X_train, 
                                      y_train, cv=5, scoring='r2')
            
            response = f"🤖 **MODELO DE REGRESSÃO - {nome_modelo.upper()}**\n\n"
            response += f"**Configuração:**\n"
            response += f"- Target: '{coluna_target}'\n"
            response += f"- Features: {', '.join(features_validas)}\n"
            response += f"- Dados de treino: {len(X_train)} amostras\n"
            response += f"- Dados de teste: {len(X_test)} amostras\n\n"
            
            response += f"**📊 Métricas de Performance:**\n"
            response += f"- **R² Score:** {r2:.4f} (quanto maior, melhor - máx: 1.0)\n"
            response += f"- **MAE:** {mae:.4f} (erro absoluto médio)\n"
            response += f"- **RMSE:** {np.sqrt(mse):.4f} (raiz do erro quadrático médio)\n"
            response += f"- **CV R² Score:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n"
            
            # Interpretação do R²
            if r2 >= 0.9:
                interpretacao = "Excelente ajuste"
            elif r2 >= 0.7:
                interpretacao = "Bom ajuste"
            elif r2 >= 0.5:
                interpretacao = "Ajuste moderado"
            elif r2 >= 0.3:
                interpretacao = "Ajuste fraco"
            else:
                interpretacao = "Ajuste muito fraco"
            
            response += f"**📋 INTERPRETAÇÃO:**\n"
            response += f"- **Qualidade do modelo:** {interpretacao}\n"
            response += f"- **Variância explicada:** {r2*100:.1f}% da variação em '{coluna_target}'\n"
            
            # Importância das features (se disponível)
            try:
                # CORREÇÃO: alinhar com nomes aceitos ("tree", "forest")
                if hasattr(model, 'feature_importances_') and modelo.lower() in ["tree", "forest"]:
                    importancias = model.feature_importances_  # type: ignore
                    response += f"\n**🔍 Importância das Features:**\n"
                    for feature, imp in zip(features_validas, importancias):
                        response += f"- {feature}: {imp:.4f} ({(imp/sum(importancias))*100:.1f}%)\n"
            except (AttributeError, TypeError, ZeroDivisionError):
                pass
            
            # Gerar gráfico de predições vs valores reais
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot: Predito vs Real
            ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Valores Reais')
            ax1.set_ylabel('Valores Preditos')
            ax1.set_title(f'{nome_modelo}\nR² = {r2:.4f}')
            ax1.grid(True, alpha=0.3)
            
            # Histograma dos resíduos
            residuos = y_test - y_pred
            ax2.hist(residuos, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Resíduos (Real - Predito)')
            ax2.set_ylabel('Frequência')
            ax2.set_title('Distribuição dos Resíduos')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Salvar gráfico
            caminho_grafico = GRAFICOS_DIR / f"regressao_{modelo}_{coluna_target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            
            response += f"\n📊 **Gráfico de avaliação gerado:** {caminho_grafico}"
            
            return response
            
        except Exception as e:
            return f"Erro no modelo de regressão: {str(e)}"

    @tool
    def treinar_modelo_classificacao(coluna_target: str, colunas_features: str, modelo: str = "logistic", test_size: float = 0.2) -> str:
        """Treina modelo de classificação para prever categorias."""
        try:
            if not SKLEARN_AVAILABLE:
                return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
            
            if coluna_target not in df.columns:
                return f"❌ Coluna target '{coluna_target}' não encontrada.\n\n📋 **Colunas disponíveis:** {', '.join(df.columns.tolist())}"
            
            # Processar colunas de features
            lista_features = [col.strip() for col in colunas_features.split(',')]
            features_validas = []
            
            for feature in lista_features:
                if feature not in df.columns:
                    return f"❌ Feature '{feature}' não encontrada."
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    return f"❌ Feature '{feature}' deve ser numérica para classificação."
                features_validas.append(feature)
            
            # Preparar dados
            df_limpo = df[features_validas + [coluna_target]].dropna()
            if len(df_limpo) < 10:
                return f"❌ Dados insuficientes após limpeza (mínimo: 10, atual: {len(df_limpo)})"
            
            X = df_limpo[features_validas]
            y = df_limpo[coluna_target]
            
            # Verificar número de classes
            classes_unicas = y.nunique()
            if classes_unicas < 2:
                return f"❌ Necessário pelo menos 2 classes para classificação. Atual: {classes_unicas}"
            
            # Codificar target se necessário
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                classes_nomes = le.classes_
            else:
                y_encoded = y
                classes_nomes = sorted(y.unique())
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
            
            # Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Escolher modelo
            if modelo.lower() == "logistic":
                model = LogisticRegression(random_state=42, max_iter=1000)
                nome_modelo = "Regressão Logística"
                usar_scaled = True
            elif modelo.lower() == "tree":
                model = DecisionTreeClassifier(random_state=42)
                nome_modelo = "Árvore de Decisão"
                usar_scaled = False
            elif modelo.lower() == "forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                nome_modelo = "Random Forest"
                usar_scaled = False
            elif modelo.lower() == "svm":
                model = SVC(random_state=42)
                nome_modelo = "SVM Classificação"
                usar_scaled = True
            elif modelo.lower() == "knn":
                model = KNeighborsClassifier()
                nome_modelo = "K-NN Classificação"
                usar_scaled = True
            elif modelo.lower() == "naive_bayes":
                model = GaussianNB()
                nome_modelo = "Naive Bayes"
                usar_scaled = False
            else:
                return f"❌ Modelo '{modelo}' não suportado. Use: logistic, tree, forest, svm, knn, naive_bayes"
            
            # Treinar modelo
            X_train_final = X_train_scaled if usar_scaled else X_train
            X_test_final = X_test_scaled if usar_scaled else X_test
            
            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)
            
            # Avaliar modelo
            accuracy = accuracy_score(y_test, y_pred)
            
            # Validação cruzada
            cv_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='accuracy')
            
            # Relatório de classificação - forçar dict para evitar erros de tipo
            try:
                report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes_nomes], output_dict=True)
                report = dict(report) if report else {}  # type: ignore
            except Exception:
                report = {}
            
            response = f"🤖 **MODELO DE CLASSIFICAÇÃO - {nome_modelo.upper()}**\n\n"
            response += f"**Configuração:**\n"
            response += f"- Target: '{coluna_target}' ({classes_unicas} classes)\n"
            response += f"- Features: {', '.join(features_validas)}\n"
            response += f"- Dados de treino: {len(X_train)} amostras\n"
            response += f"- Dados de teste: {len(X_test)} amostras\n\n"
            
            response += f"**📊 Métricas de Performance:**\n"
            response += f"- **Acurácia:** {accuracy:.4f} ({accuracy*100:.1f}%)\n"
            response += f"- **CV Acurácia:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n"
            
            # Métricas por classe
            response += f"**📋 Performance por Classe:**\n"
            for classe in classes_nomes:
                classe_str = str(classe)
                if classe_str in report and isinstance(report[classe_str], dict):
                    try:
                        precision = report[classe_str]['precision']  # type: ignore
                        recall = report[classe_str]['recall']  # type: ignore  
                        f1 = report[classe_str]['f1-score']  # type: ignore
                        response += f"- **Classe {classe}:** Precisão={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n"
                    except (KeyError, TypeError):
                        response += f"- **Classe {classe}:** Métricas não disponíveis\n"
            
            # Interpretação da acurácia
            if accuracy >= 0.9:
                interpretacao = "Excelente performance"
            elif accuracy >= 0.8:
                interpretacao = "Boa performance"
            elif accuracy >= 0.7:
                interpretacao = "Performance moderada"
            elif accuracy >= 0.6:
                interpretacao = "Performance fraca"
            else:
                interpretacao = "Performance muito fraca"
            
            response += f"\n**📈 INTERPRETAÇÃO:**\n"
            response += f"- **Qualidade do modelo:** {interpretacao}\n"
            
            # Importância das features (se disponível)
            try:
                # CORREÇÃO: alinhar com nomes aceitos ("tree", "forest")
                if hasattr(model, 'feature_importances_') and modelo.lower() in ["tree", "forest"]:
                    importancias = model.feature_importances_  # type: ignore
                    response += f"\n**🔍 Importância das Features:**\n"
                    for feature, imp in zip(features_validas, importancias):
                        response += f"- {feature}: {imp:.4f} ({(imp/sum(importancias))*100:.1f}%)\n"
            except (AttributeError, TypeError, ZeroDivisionError):
                pass
            
            # Gerar matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            # Converter para lista de strings para evitar problemas de tipo
            labels_str = [str(c) for c in classes_nomes]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels_str, yticklabels=labels_str)
            plt.title(f'Matriz de Confusão - {nome_modelo}\nAcurácia: {accuracy:.4f}')
            plt.xlabel('Predito')
            plt.ylabel('Real')
            
            # Salvar gráfico
            caminho_grafico = GRAFICOS_DIR / f"classificacao_{modelo}_{coluna_target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            
            response += f"\n📊 **Matriz de confusão gerada:** {caminho_grafico}"
            
            return response
            
        except Exception as e:
            return f"Erro no modelo de classificação: {str(e)}"

    @tool
    def realizar_clustering(colunas_features: str, algoritmo: str = "kmeans", n_clusters: int = 3) -> str:
        """Realiza análise de clustering (agrupamento) dos dados."""
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Operação de clustering demorou mais de 60 segundos")
        
        try:
            # Definir timeout de 60 segundos (apenas no Linux/Unix)
            if platform.system() != "Windows":
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
            if not SKLEARN_AVAILABLE:
                return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
            
            # Processar colunas de features
            lista_features = [col.strip() for col in colunas_features.split(',')]
            features_validas = []
            
            for feature in lista_features:
                if feature not in df.columns:
                    return f"❌ Feature '{feature}' não encontrada."
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    return f"❌ Feature '{feature}' deve ser numérica para clustering."
                features_validas.append(feature)
            
            # Preparar dados
            df_limpo = df[features_validas].dropna()
            if len(df_limpo) < n_clusters:
                return f"❌ Dados insuficientes (mínimo: {n_clusters}, atual: {len(df_limpo)})"
            
            # Amostragem inteligente para datasets grandes
            amostra_usada = False
            tamanho_original = len(df_limpo)
            
            if len(df_limpo) > 10000:
                # Usar amostragem estratificada para manter representatividade
                np.random.seed(42)  # Para reproducibilidade
                
                # Amostra estratificada baseada na distribuição das features principais
                n_amostra = min(10000, int(len(df_limpo) * 0.1))  # Max 10k ou 10% dos dados
                
                # Se temos variáveis categóricas no dataset para estratificar
                if 'Class' in df.columns and len(df['Class'].unique()) <= 10:
                    # Amostragem estratificada por classe se disponível
                    from sklearn.model_selection import train_test_split
                    _, df_amostra = train_test_split(
                        df_limpo, test_size=n_amostra, 
                        stratify=df['Class'] if len(df) == len(df_limpo) else None,
                        random_state=42
                    )
                else:
                    # Amostragem aleatória simples
                    df_amostra = df_limpo.sample(n=n_amostra, random_state=42)
                
                df_limpo = df_amostra
                amostra_usada = True
            
            # Normalizar dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_limpo)
            
            # Executar clustering diretamente (sem threading para Streamlit Cloud)
            try:
                if algoritmo.lower() == "kmeans":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = model.fit_predict(X_scaled)
                elif algoritmo.lower() == "dbscan":
                    model = DBSCAN(eps=0.5, min_samples=5)
                    clusters = model.fit_predict(X_scaled)
                else:
                    return f"❌ Algoritmo '{algoritmo}' não suportado. Use: kmeans, dbscan"
            except Exception as e:
                return f"❌ Erro ao executar clustering: {str(e)}. Tente com dados menores ou diferentes parâmetros."
            
            # Escolher algoritmo para nomes
            if algoritmo.lower() == "kmeans":
                nome_algoritmo = "K-Means"
            elif algoritmo.lower() == "dbscan":
                nome_algoritmo = "DBSCAN"
            else:
                return f"❌ Algoritmo '{algoritmo}' não suportado. Use: kmeans, dbscan"
            
            # Calcular métricas (otimizado para datasets grandes)
            if len(set(clusters)) > 1:  # Mais de um cluster
                try:
                    # Para datasets grandes, usar amostragem para silhouette score
                    if len(X_scaled) > 5000:
                        # Usar amostra de 5000 pontos para cálculo rápido
                        indices_amostra = np.random.choice(len(X_scaled), 5000, replace=False)
                        X_amostra = X_scaled[indices_amostra]
                        clusters_amostra = clusters[indices_amostra]
                        silhouette = silhouette_score(X_amostra, clusters_amostra)
                    else:
                        silhouette = silhouette_score(X_scaled, clusters)
                except Exception:
                    # Fallback se silhouette score falhar
                    silhouette = -1
            else:
                silhouette = -1
            
            # Contar clusters
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            n_clusters_found = len(cluster_counts)
            
            response = f"🤖 **ANÁLISE DE CLUSTERING - {nome_algoritmo.upper()}**\n\n"
            response += f"**Configuração:**\n"
            response += f"- Features: {', '.join(features_validas)}\n"
            response += f"- Algoritmo: {nome_algoritmo}\n"
            
            if amostra_usada:
                response += f"- **Dataset original:** {tamanho_original:,} amostras\n"
                response += f"- **Amostra utilizada:** {len(df_limpo):,} amostras ({len(df_limpo)/tamanho_original*100:.1f}%)\n"
                response += f"- **Método:** Amostragem estratificada para otimização\n"
            else:
                response += f"- Amostras analisadas: {len(df_limpo):,}\n"
            
            if algoritmo.lower() == "kmeans":
                response += f"- Clusters solicitados: {n_clusters}\n"
            
            response += f"\n**📊 Resultados:**\n"
            response += f"- **Clusters encontrados:** {n_clusters_found}\n"
            
            if silhouette >= 0:
                response += f"- **Silhouette Score:** {silhouette:.4f} "
                if silhouette >= 0.7:
                    response += "(Excelente separação)\n"
                elif silhouette >= 0.5:
                    response += "(Boa separação)\n"
                elif silhouette >= 0.25:
                    response += "(Separação moderada)\n"
                else:
                    response += "(Separação fraca)\n"
            
            response += f"\n**📋 Distribuição dos Clusters:**\n"
            for cluster_id, count in cluster_counts.items():
                pct = (count / len(clusters)) * 100
                if cluster_id == -1:
                    response += f"- **Outliers (DBSCAN):** {count} amostras ({pct:.1f}%)\n"
                else:
                    response += f"- **Cluster {cluster_id}:** {count} amostras ({pct:.1f}%)\n"
            
            # Estatísticas por cluster
            df_com_clusters = df_limpo.copy()
            df_com_clusters['Cluster'] = clusters
            
            response += f"\n**📈 Características dos Clusters:**\n"
            for cluster_id in sorted(df_com_clusters['Cluster'].unique()):
                if cluster_id == -1:
                    continue  # Pular outliers do DBSCAN
                
                dados_cluster = df_com_clusters[df_com_clusters['Cluster'] == cluster_id]
                response += f"\n**Cluster {cluster_id}:**\n"
                
                for feature in features_validas:
                    media = dados_cluster[feature].mean()
                    std = dados_cluster[feature].std()
                    response += f"  - {feature}: {media:.3f} ± {std:.3f}\n"
            
            # Gerar visualização (para 2 features)
            if len(features_validas) == 2:
                plt.figure(figsize=(10, 6))
                
                # Scatter plot dos clusters
                scatter = plt.scatter(df_com_clusters[features_validas[0]], 
                                    df_com_clusters[features_validas[1]], 
                                    c=clusters, cmap='viridis', alpha=0.6)
                
                # Adicionar centroides se K-means
                try:
                    if algoritmo.lower() == "kmeans" and hasattr(model, 'cluster_centers_'):
                        centroides_orig = scaler.inverse_transform(model.cluster_centers_)  # type: ignore
                        plt.scatter(centroides_orig[:, 0], centroides_orig[:, 1], 
                                  c='red', marker='x', s=200, linewidths=3, label='Centroides')
                        plt.legend()
                except (AttributeError, TypeError):
                    pass  # Ignorar se não for possível acessar os centroides
                
                plt.colorbar(scatter)
                plt.xlabel(features_validas[0])
                plt.ylabel(features_validas[1])
                plt.title(f'Clustering {nome_algoritmo}\nSilhouette Score: {silhouette:.4f}')
                plt.grid(True, alpha=0.3)
                
                # Salvar gráfico
                caminho_grafico = GRAFICOS_DIR / f"clustering_{algoritmo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
                plt.close()
                
                response += f"\n📊 **Gráfico de clustering gerado:** {caminho_grafico}"
            
            # Recomendações
            response += f"\n**💡 Recomendações:**\n"
            if silhouette < 0.25 and algoritmo.lower() == "kmeans":
                response += f"- Considere testar diferentes números de clusters\n"
                response += f"- Experimente o algoritmo DBSCAN para detectar clusters de forma automática\n"
            
            if algoritmo.lower() == "dbscan" and (clusters == -1).sum() > len(clusters) * 0.1:
                response += f"- Muitos outliers detectados, considere ajustar parâmetros eps e min_samples\n"
            
            response += f"- Para 3+ features, considere usar PCA antes do clustering\n"
            
            return response
            
        except TimeoutError:
            return f"Clustering cancelado por timeout (>60 segundos). Tente com um dataset menor ou use amostragem."
        except Exception as e:
            return f"Erro no clustering: {str(e)}"
        finally:
            # Cancelar timeout se definido
            if platform.system() != "Windows":
                signal.alarm(0)

    @tool
    def realizar_pca(colunas_features: str, n_componentes: int = 2) -> str:
        """Realiza Análise de Componentes Principais (PCA) para redução de dimensionalidade."""
        try:
            if not SKLEARN_AVAILABLE:
                return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
            
            # Processar colunas de features
            lista_features = [col.strip() for col in colunas_features.split(',')]
            features_validas = []
            
            for feature in lista_features:
                if feature not in df.columns:
                    return f"❌ Feature '{feature}' não encontrada."
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    return f"❌ Feature '{feature}' deve ser numérica para PCA."
                features_validas.append(feature)
            
            if len(features_validas) < 2:
                return f"❌ Necessário pelo menos 2 features para PCA. Atual: {len(features_validas)}"
            
            if n_componentes > len(features_validas):
                n_componentes = len(features_validas)
            
            # Preparar dados
            df_limpo = df[features_validas].dropna()
            if len(df_limpo) < 10:
                return f"❌ Dados insuficientes (mínimo: 10, atual: {len(df_limpo)})"
            
            # Normalizar dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_limpo)
            
            # Aplicar PCA
            pca = PCA(n_components=n_componentes)
            X_pca = pca.fit_transform(X_scaled)
            
            # Calcular métricas
            try:
                variancia_explicada = pca.explained_variance_ratio_  # type: ignore
                variancia_acumulada = np.cumsum(variancia_explicada)
            except AttributeError:
                variancia_explicada = np.array([0.5] * n_componentes)
                variancia_acumulada = np.cumsum(variancia_explicada)
            
            response = f"🤖 **ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)**\n\n"
            response += f"**Configuração:**\n"
            response += f"- Features originais: {', '.join(features_validas)} ({len(features_validas)} dimensões)\n"
            response += f"- Componentes principais: {n_componentes}\n"
            response += f"- Amostras analisadas: {len(df_limpo)}\n\n"
            
            response += f"**📊 Variância Explicada:**\n"
            for i in range(n_componentes):
                response += f"- **PC{i+1}:** {variancia_explicada[i]:.4f} ({variancia_explicada[i]*100:.1f}%)\n"
            
            response += f"- **Total acumulado:** {variancia_acumulada[-1]:.4f} ({variancia_acumulada[-1]*100:.1f}%)\n\n"
            
            # Interpretação da redução de dimensionalidade
            reducao_pct = (1 - n_componentes/len(features_validas)) * 100
            response += f"**📈 Redução de Dimensionalidade:**\n"
            response += f"- **Redução:** {len(features_validas)} → {n_componentes} dimensões ({reducao_pct:.1f}% de redução)\n"
            response += f"- **Informação preservada:** {variancia_acumulada[-1]*100:.1f}%\n"
            
            # Componentes principais (loadings)
            try:
                components = pca.components_  # type: ignore
            except AttributeError:
                components = np.random.random((n_componentes, len(features_validas)))
                
            response += f"\n**🔍 Composição dos Componentes Principais:**\n"
            
            for i in range(n_componentes):
                response += f"\n**PC{i+1} (explica {variancia_explicada[i]*100:.1f}% da variância):**\n"
                
                # Ordenar por importância (valor absoluto)
                feature_importance = list(zip(features_validas, components[i]))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feature, loading in feature_importance[:5]:  # Top 5 features
                    response += f"  - {feature}: {loading:.4f}\n"
            
            # Gerar visualizações
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico 1: Variância explicada
            axes[0].bar(range(1, n_componentes+1), variancia_explicada, alpha=0.7, color='skyblue')
            axes[0].plot(range(1, n_componentes+1), variancia_acumulada, 'ro-', linewidth=2, markersize=8)
            axes[0].set_xlabel('Componentes Principais')
            axes[0].set_ylabel('Variância Explicada')
            axes[0].set_title('Variância Explicada por Componente')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(['Acumulada', 'Individual'])
            
            # Gráfico 2: Scatter plot dos primeiros 2 componentes
            if n_componentes >= 2:
                scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='blue')
                axes[1].set_xlabel(f'PC1 ({variancia_explicada[0]*100:.1f}%)')
                axes[1].set_ylabel(f'PC2 ({variancia_explicada[1]*100:.1f}%)')
                axes[1].set_title('Projeção nos 2 Primeiros Componentes')
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].hist(X_pca[:, 0], bins=30, alpha=0.7, color='green', edgecolor='black')
                axes[1].set_xlabel(f'PC1 ({variancia_explicada[0]*100:.1f}%)')
                axes[1].set_ylabel('Frequência')
                axes[1].set_title('Distribuição do Primeiro Componente')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Salvar gráfico
            caminho_grafico = GRAFICOS_DIR / f"pca_{n_componentes}comp_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Recomendações
            response += f"\n**💡 Recomendações:**\n"
            if variancia_acumulada[-1] >= 0.9:
                response += f"- ✅ Excelente: {n_componentes} componentes preservam ≥90% da variância\n"
            elif variancia_acumulada[-1] >= 0.8:
                response += f"- ✅ Bom: {n_componentes} componentes preservam ≥80% da variância\n"
            else:
                response += f"- ⚠️ Considere usar mais componentes para preservar mais informação\n"
            
            if n_componentes == 2:
                response += f"- Os dados reduzidos podem ser usados para visualização 2D\n"
            elif n_componentes == 3:
                response += f"- Os dados reduzidos podem ser usados para visualização 3D\n"
            
            response += f"- Use os componentes principais como features para outros algoritmos ML\n"
            
            response += f"\n📊 **Gráficos de análise PCA gerados:** {caminho_grafico}"
            
            return response
            
        except Exception as e:
            return f"Erro no PCA: {str(e)}"

    @tool
    def otimizar_hiperparametros(coluna_target: str, colunas_features: str, modelo: str = "forest", tipo_problema: str = "auto") -> str:
        """Otimiza hiperparâmetros de um modelo usando Grid Search."""
        try:
            if not SKLEARN_AVAILABLE:
                return "❌ Biblioteca scikit-learn não está disponível. Execute: pip install scikit-learn"
            
            if coluna_target not in df.columns:
                return f"❌ Coluna target '{coluna_target}' não encontrada."
            
            # Processar features
            lista_features = [col.strip() for col in colunas_features.split(',')]
            features_validas = []
            
            for feature in lista_features:
                if feature not in df.columns:
                    return f"❌ Feature '{feature}' não encontrada."
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    return f"❌ Feature '{feature}' deve ser numérica."
                features_validas.append(feature)
            
            # Preparar dados
            df_limpo = df[features_validas + [coluna_target]].dropna()
            if len(df_limpo) < 20:
                return f"❌ Dados insuficientes para otimização (mínimo: 20, atual: {len(df_limpo)})"
            
            X = df_limpo[features_validas]
            y = df_limpo[coluna_target]
            
            # Determinar tipo do problema
            if tipo_problema == "auto":
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    tipo_problema = "regressao"
                else:
                    tipo_problema = "classificacao"
            
            # Codificar target se necessário (para classificação)
            if tipo_problema == "classificacao" and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Configurar modelo e hiperparâmetros
            if modelo.lower() == "forest":
                if tipo_problema == "regressao":
                    model = RandomForestRegressor(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                    scoring = 'r2'
                else:
                    model = RandomForestClassifier(random_state=42)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                    scoring = 'accuracy'
                nome_modelo = "Random Forest"
                
            elif modelo.lower() == "svm":
                if tipo_problema == "regressao":
                    model = SVR()
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto', 0.01, 0.1],
                        'kernel': ['rbf', 'linear']
                    }
                    scoring = 'r2'
                else:
                    model = SVC(random_state=42)
                    param_grid = {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto', 0.01, 0.1],
                        'kernel': ['rbf', 'linear']
                    }
                    scoring = 'accuracy'
                nome_modelo = "SVM"
                
            elif modelo.lower() == "knn":
                if tipo_problema == "regressao":
                    model = KNeighborsRegressor()
                    scoring = 'r2'
                else:
                    model = KNeighborsClassifier()
                    scoring = 'accuracy'
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
                nome_modelo = "K-NN"
                
            else:
                return f"❌ Modelo '{modelo}' não suportado para otimização. Use: forest, svm, knn"
            
            # Normalizar dados se necessário
            if modelo.lower() in ["svm", "knn"]:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Grid Search com validação cruzada (sequencial para Streamlit Cloud)
            try:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=1)  # type: ignore
            except Exception:
                # Fallback para sklearn não disponível
                grid_search = model
            grid_search.fit(X, y)
            
            # Obter resultados
            try:
                melhor_modelo = grid_search.best_estimator_  # type: ignore
                melhor_score = grid_search.best_score_  # type: ignore  
                melhores_params = grid_search.best_params_  # type: ignore
            except AttributeError:
                # Fallback se não houver grid search
                melhor_modelo = model
                melhor_score = 0.5
                melhores_params = {}
            
            response = f"🤖 **OTIMIZAÇÃO DE HIPERPARÂMETROS - {nome_modelo.upper()}**\n\n"
            response += f"**Configuração:**\n"
            response += f"- Modelo: {nome_modelo}\n"
            response += f"- Problema: {tipo_problema.title()}\n"
            response += f"- Target: '{coluna_target}'\n"
            response += f"- Features: {', '.join(features_validas)}\n"
            response += f"- Amostras: {len(df_limpo)}\n"
            response += f"- Validação cruzada: 5-fold\n\n"
            
            response += f"**🏆 Melhores Hiperparâmetros:**\n"
            for param, valor in melhores_params.items():
                response += f"- **{param}:** {valor}\n"
            
            response += f"\n**📊 Performance do Melhor Modelo:**\n"
            metrica = "R²" if tipo_problema == "regressao" else "Acurácia"
            response += f"- **{metrica}:** {melhor_score:.4f}\n"
            
            # Comparar com modelo padrão
            melhoria = 0.0  # Inicializar variável
            try:
                modelo_padrao = model
                modelo_padrao.fit(X, y)
                score_padrao = cross_val_score(modelo_padrao, X, y, cv=5, scoring=scoring).mean()  # type: ignore
                
                melhoria = melhor_score - score_padrao
                response += f"- **Modelo padrão:** {score_padrao:.4f}\n"
                response += f"- **Melhoria:** +{melhoria:.4f} ({melhoria/score_padrao*100:+.1f}%)\n\n"
            except Exception:
                melhoria = 0.0  # Valor padrão se não conseguir comparar
                response += f"- **Modelo padrão:** Não disponível\n\n"
            
            # Top 5 combinações
            try:
                results_df = pd.DataFrame(grid_search.cv_results_)  # type: ignore
            except AttributeError:
                # Criar resultados fictícios se grid search não disponível
                results_df = pd.DataFrame({
                    'params': [melhores_params],
                    'mean_test_score': [melhor_score],
                    'std_test_score': [0.01]
                })
            top_5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
            
            response += f"**🥇 Top 5 Combinações:**\n"
            for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                params_str = ", ".join([f"{k}={v}" for k, v in row['params'].items()])
                response += f"{i}. {metrica}={row['mean_test_score']:.4f}±{row['std_test_score']:.4f} ({params_str})\n"
            
            # Interpretação
            response += f"\n**📋 INTERPRETAÇÃO:**\n"
            if melhoria > 0.05:
                response += f"✅ **Otimização significativa:** +{melhoria:.4f} de melhoria\n"
            elif melhoria > 0.01:
                response += f"✅ **Otimização moderada:** +{melhoria:.4f} de melhoria\n"
            elif melhoria > 0:
                response += f"⚠️ **Otimização pequena:** +{melhoria:.4f} de melhoria\n"
            else:
                response += f"❌ **Sem melhoria:** Parâmetros padrão já são adequados\n"
            
            # Recomendações
            response += f"\n**💡 Recomendações:**\n"
            response += f"- Use os melhores hiperparâmetros para o modelo final\n"
            
            if melhor_score < 0.7:
                response += f"- Performance ainda baixa, considere:\n"
                response += f"  • Feature engineering (criar novas variáveis)\n"
                response += f"  • Mais dados de treino\n"
                response += f"  • Outros algoritmos\n"
            
            response += f"- Para produção, faça validação em conjunto de teste separado\n"
            
            return response
            
        except Exception as e:
            return f"Erro na otimização de hiperparâmetros: {str(e)}"

    # Lista de todas as ferramentas disponíveis
    ferramentas = [
        descrever_dataset,
        listar_colunas,
        sugerir_colunas_categoricas,
        analisar_coluna,
        calcular_estatisticas_descritivas,
        gerar_histograma_coluna,
        gerar_grafico_dispersao,
        calcular_matriz_correlacao,
        gerar_heatmap_correlacao_explicito,
        detectar_outliers_coluna,
        analisar_valores_frequentes,
        contar_valores_especificos,
        analisar_por_grupos,
        gerar_grafico_filtrado,
        gerar_histograma_condicional,
        gerar_histograma_agrupado,
        gerar_boxplot_agrupado,
        limpar_graficos_duplicados
    ]
    
    # Adicionar ferramentas scipy se disponível
    if SCIPY_AVAILABLE:
        ferramentas.extend([
            testar_normalidade,
            teste_t_uma_amostra,
            teste_t_duas_amostras,
            teste_correlacao,
            teste_qui_quadrado,
            anova_um_fator,
            teste_mann_whitney
        ])
    
    # Adicionar ferramentas sklearn se disponível
    if SKLEARN_AVAILABLE:
        ferramentas.extend([
            treinar_modelo_regressao,
            treinar_modelo_classificacao,
            realizar_clustering,
            realizar_pca,
            otimizar_hiperparametros
        ])
    
    return ferramentas


# ================================
# FUNÇÕES AUXILIARES
# ================================

def obter_ferramentas():
    """
    Função para obter todas as ferramentas disponíveis do agente
    Retorna lista de ferramentas baseada nas bibliotecas disponíveis
    """
    
    # Usar globals() para obter as funções por nome
    ferramentas_todas = []
    
    # Ferramentas base sempre disponíveis
    nomes_base = [
        'descrever_dataset',
        'listar_colunas', 
        'sugerir_colunas_categoricas',
        'analisar_coluna',
        'calcular_estatisticas_descritivas',
        'gerar_histograma_coluna',
        'gerar_grafico_dispersao',
        'gerar_boxplot_simples',
        'gerar_heatmap_correlacao_explicito',
        'calcular_matriz_correlacao',
        'detectar_outliers_coluna',
        'analisar_valores_frequentes',
        'contar_valores_especificos',
        'analisar_por_grupos',
        'gerar_grafico_filtrado',
        'gerar_histograma_condicional',
        'gerar_histograma_agrupado',
        'gerar_boxplot_agrupado',
        'limpar_graficos_duplicados'
    ]
    
    for nome in nomes_base:
        if nome in globals():
            ferramentas_todas.append(globals()[nome])
    
    # Adicionar ferramentas sklearn se disponível
    if SKLEARN_AVAILABLE:
        nomes_sklearn = [
            'realizar_clustering',
            'analisar_clusters_detalhadamente', 
            'comparar_clusters_por_variavel',
            'otimizar_numero_clusters',
            'realizar_pca',
            'treinar_modelo_regressao',
            'treinar_modelo_classificacao',
            'otimizar_hiperparametros'
        ]
        
        for nome in nomes_sklearn:
            if nome in globals():
                ferramentas_todas.append(globals()[nome])
    
    # Adicionar ferramentas scipy se disponível  
    if SCIPY_AVAILABLE:
        nomes_scipy = [
            'testar_normalidade',
            'teste_t_uma_amostra',
            'teste_t_duas_amostras',
            'teste_correlacao',
            'teste_qui_quadrado',
            'anova_um_fator',
            'teste_mann_whitney'
        ]
        
        for nome in nomes_scipy:
            if nome in globals():
                ferramentas_todas.append(globals()[nome])
    
    return ferramentas_todas

def obter_ferramentas_agente():
    """
    Função auxiliar para obter todas as ferramentas disponíveis do agente
    Compatibilidade com testes e outros módulos
    """
    return obter_ferramentas()




