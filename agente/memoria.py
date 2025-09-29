"""
Módulo de Memória do Agente
Gerencia o histórico de conversas e contexto das análises.
"""

from typing import List, Tuple, Dict, Any, Optional
import datetime
import json
from pathlib import Path

# Imports condicionais para LangChain
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class MemoriaConversa:
    """
    Gerencia a memória de curto prazo do agente para manter contexto das conversas.
    """
    
    def __init__(self, limite_interacoes: int = 50):
        """
        Inicializa a memória da conversa.
        
        Args:
            limite_interacoes: Número máximo de interações a manter em memória
        """
        self.limite_interacoes = limite_interacoes
        self.historico: List[Dict[str, Any]] = []
        self.contexto_atual: Dict[str, Any] = {}
        
        # Inicializar memória LangChain se disponível
        if LANGCHAIN_AVAILABLE:
            self.langchain_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            self.langchain_memory = None
    
    def adicionar_interacao(self, pergunta: str, resposta: Optional[str], 
                          metadados: Optional[Dict[str, Any]] = None):
        """
        Adiciona uma nova interação ao histórico.
        
        Args:
            pergunta: Pergunta do usuário
            resposta: Resposta do agente
            metadados: Informações adicionais (gráficos gerados, análises, etc.)
        """
        interacao = {
            'timestamp': datetime.datetime.now().isoformat(),
            'pergunta': pergunta,
            'resposta': resposta,
            'metadados': metadados or {}
        }
        
        self.historico.append(interacao)
        
        # Manter apenas as últimas N interações
        if len(self.historico) > self.limite_interacoes:
            self.historico = self.historico[-self.limite_interacoes:]
        
        # Atualizar memória LangChain se disponível
        if self.langchain_memory and resposta:
            self.langchain_memory.chat_memory.add_user_message(pergunta)
            self.langchain_memory.chat_memory.add_ai_message(resposta)
    
    def get_historico_recente(self, n_interacoes: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna as últimas N interações.
        
        Args:
            n_interacoes: Número de interações a retornar
            
        Returns:
            Lista com as interações mais recentes
        """
        return self.historico[-n_interacoes:] if self.historico else []
    
    def get_historico_formatado(self, incluir_metadados: bool = False) -> str:
        """
        Retorna o histórico formatado como texto.
        
        Args:
            incluir_metadados: Se deve incluir metadados na formatação
            
        Returns:
            Histórico formatado como string
        """
        if not self.historico:
            return "Nenhuma conversa anterior."
        
        texto_formatado = []
        
        for i, interacao in enumerate(self.historico[-10:], 1):  # Últimas 10 interações
            timestamp = datetime.datetime.fromisoformat(interacao['timestamp'])
            tempo_formatado = timestamp.strftime("%H:%M")
            
            texto_formatado.append(f"[{tempo_formatado}] Usuário: {interacao['pergunta']}")
            
            if interacao['resposta']:
                # Limitar tamanho da resposta para não sobrecarregar o contexto
                resposta = interacao['resposta']
                if len(resposta) > 200:
                    resposta = resposta[:200] + "..."
                
                texto_formatado.append(f"[{tempo_formatado}] Agente: {resposta}")
            
            if incluir_metadados and interacao['metadados']:
                metadados_str = json.dumps(interacao['metadados'], indent=2)[:100] + "..."
                texto_formatado.append(f"[{tempo_formatado}] Metadados: {metadados_str}")
            
            texto_formatado.append("")  # Linha em branco entre interações
        
        return "\n".join(texto_formatado)
    
    def get_langchain_memory(self):
        """Retorna a instância de memória do LangChain."""
        return self.langchain_memory
    
    def extrair_contexto_relevante(self, pergunta_atual: str) -> Dict[str, Any]:
        """
        Extrai contexto relevante baseado na pergunta atual.
        
        Args:
            pergunta_atual: Pergunta atual do usuário
            
        Returns:
            Dicionário com contexto relevante
        """
        contexto = {
            'colunas_mencionadas': [],
            'analises_anteriores': [],
            'graficos_gerados': [],
            'temas_recorrentes': []
        }
        
        pergunta_lower = pergunta_atual.lower()
        
        # Analisar histórico para extrair contexto
        for interacao in self.historico[-5:]:  # Últimas 5 interações
            pergunta_hist = interacao['pergunta'].lower()
            
            # Procurar por colunas mencionadas
            # (Esta é uma implementação simples, pode ser melhorada com NLP)
            palavras_pergunta = pergunta_lower.split()
            palavras_historico = pergunta_hist.split()
            
            # Encontrar palavras em comum que podem ser nomes de colunas
            palavras_comuns = set(palavras_pergunta) & set(palavras_historico)
            
            for palavra in palavras_comuns:
                if len(palavra) > 3 and palavra not in ['dados', 'gráfico', 'análise']:
                    contexto['colunas_mencionadas'].append(palavra)
            
            # Extrair análises anteriores
            if interacao['resposta']:
                if 'correlação' in pergunta_hist or 'correlacao' in pergunta_hist:
                    contexto['analises_anteriores'].append('correlacao')
                if 'outlier' in pergunta_hist:
                    contexto['analises_anteriores'].append('outliers')
                if 'histograma' in pergunta_hist:
                    contexto['analises_anteriores'].append('histograma')
            
            # Extrair gráficos gerados
            if interacao['metadados'].get('graficos'):
                contexto['graficos_gerados'].extend(interacao['metadados']['graficos'])
        
        # Remover duplicatas
        contexto['colunas_mencionadas'] = list(set(contexto['colunas_mencionadas']))
        contexto['analises_anteriores'] = list(set(contexto['analises_anteriores']))
        
        return contexto
    
    def atualizar_contexto_sessao(self, chave: str, valor: Any):
        """
        Atualiza o contexto atual da sessão.
        
        Args:
            chave: Chave do contexto
            valor: Valor a ser armazenado
        """
        self.contexto_atual[chave] = valor
    
    def get_contexto_sessao(self, chave: str, default: Any = None) -> Any:
        """
        Obtém um valor do contexto da sessão.
        
        Args:
            chave: Chave do contexto
            default: Valor padrão se a chave não existir
            
        Returns:
            Valor armazenado ou valor padrão
        """
        return self.contexto_atual.get(chave, default)
    
    def limpar_historico(self):
        """Limpa todo o histórico de conversas."""
        self.historico.clear()
        self.contexto_atual.clear()
        
        if self.langchain_memory:
            self.langchain_memory.clear()
    
    def salvar_historico(self, caminho_arquivo: str):
        """
        Salva o histórico em um arquivo JSON.
        
        Args:
            caminho_arquivo: Caminho para salvar o histórico
        """
        try:
            dados_salvamento = {
                'historico': self.historico,
                'contexto_atual': self.contexto_atual,
                'timestamp_salvamento': datetime.datetime.now().isoformat()
            }
            
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                json.dump(dados_salvamento, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Histórico salvo em: {caminho_arquivo}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar histórico: {e}")
    
    def carregar_historico(self, caminho_arquivo: str):
        """
        Carrega o histórico de um arquivo JSON.
        
        Args:
            caminho_arquivo: Caminho do arquivo a carregar
        """
        try:
            if not Path(caminho_arquivo).exists():
                print(f"⚠️ Arquivo não encontrado: {caminho_arquivo}")
                return
            
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                dados_carregados = json.load(f)
            
            self.historico = dados_carregados.get('historico', [])
            self.contexto_atual = dados_carregados.get('contexto_atual', {})
            
            # Recriar memória LangChain se disponível
            if self.langchain_memory:
                self.langchain_memory.clear()
                for interacao in self.historico:
                    if interacao['resposta']:
                        self.langchain_memory.chat_memory.add_user_message(interacao['pergunta'])
                        self.langchain_memory.chat_memory.add_ai_message(interacao['resposta'])
            
            print(f"✅ Histórico carregado: {len(self.historico)} interações")
            
        except Exception as e:
            print(f"❌ Erro ao carregar histórico: {e}")
    
    def get_estatisticas_conversa(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre a conversa atual.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.historico:
            return {'total_interacoes': 0}
        
        # Calcular estatísticas
        total_interacoes = len(self.historico)
        perguntas_com_resposta = sum(1 for h in self.historico if h['resposta'])
        
        # Duração da conversa
        primeiro_timestamp = datetime.datetime.fromisoformat(self.historico[0]['timestamp'])
        ultimo_timestamp = datetime.datetime.fromisoformat(self.historico[-1]['timestamp'])
        duracao_minutos = (ultimo_timestamp - primeiro_timestamp).total_seconds() / 60
        
        # Análise de temas
        temas = []
        for interacao in self.historico:
            pergunta = interacao['pergunta'].lower()
            if 'histograma' in pergunta:
                temas.append('visualizacao')
            elif any(palavra in pergunta for palavra in ['correlação', 'correlacao']):
                temas.append('correlacao')
            elif 'outlier' in pergunta:
                temas.append('outliers')
            elif any(palavra in pergunta for palavra in ['descrev', 'resumo']):
                temas.append('descricao')
        
        from collections import Counter
        temas_counter = Counter(temas)
        
        return {
            'total_interacoes': total_interacoes,
            'perguntas_com_resposta': perguntas_com_resposta,
            'duracao_conversa_minutos': round(duracao_minutos, 1),
            'temas_mais_comuns': dict(temas_counter.most_common(5)),
            'primeira_interacao': primeiro_timestamp.strftime("%H:%M:%S"),
            'ultima_interacao': ultimo_timestamp.strftime("%H:%M:%S")
        }


class MemoriaLongoPrazo:
    """
    Memória de longo prazo para armazenar insights e aprendizados
    sobre diferentes datasets analisados.
    """
    
    def __init__(self, diretorio_memoria: str = "memoria_insights"):
        """
        Inicializa a memória de longo prazo.
        
        Args:
            diretorio_memoria: Diretório para armazenar insights
        """
        self.diretorio = Path(diretorio_memoria)
        self.diretorio.mkdir(exist_ok=True)
        self.insights_atuais: List[Dict[str, Any]] = []
    
    def adicionar_insight(self, categoria: str, descricao: str, 
                         contexto: Dict[str, Any], importancia: int = 1):
        """
        Adiciona um insight descoberto durante a análise.
        
        Args:
            categoria: Categoria do insight (ex: 'correlacao', 'outliers', 'distribuicao')
            descricao: Descrição do insight
            contexto: Contexto da descoberta (colunas, valores, etc.)
            importancia: Nível de importância (1-5)
        """
        insight = {
            'timestamp': datetime.datetime.now().isoformat(),
            'categoria': categoria,
            'descricao': descricao,
            'contexto': contexto,
            'importancia': importancia
        }
        
        self.insights_atuais.append(insight)
    
    def salvar_insights_dataset(self, nome_dataset: str):
        """
        Salva os insights atuais para um dataset específico.
        
        Args:
            nome_dataset: Nome identificador do dataset
        """
        if not self.insights_atuais:
            return
        
        arquivo_insights = self.diretorio / f"insights_{nome_dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        dados_salvamento = {
            'dataset': nome_dataset,
            'timestamp': datetime.datetime.now().isoformat(),
            'insights': self.insights_atuais
        }
        
        try:
            with open(arquivo_insights, 'w', encoding='utf-8') as f:
                json.dump(dados_salvamento, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Insights salvos: {len(self.insights_atuais)} descobertas em {arquivo_insights}")
            self.insights_atuais.clear()
            
        except Exception as e:
            print(f"❌ Erro ao salvar insights: {e}")
    
    def buscar_insights_similares(self, contexto_atual: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Busca insights similares de análises anteriores.
        
        Args:
            contexto_atual: Contexto da análise atual
            
        Returns:
            Lista de insights similares encontrados
        """
        insights_similares = []
        
        # Implementação simples - pode ser melhorada com embedding/vetorização
        for arquivo in self.diretorio.glob("insights_*.json"):
            try:
                with open(arquivo, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                
                for insight in dados.get('insights', []):
                    # Verificar similaridade baseada em categorias e contexto
                    if self._verificar_similaridade(insight['contexto'], contexto_atual):
                        insights_similares.append({
                            'insight': insight,
                            'fonte_dataset': dados['dataset'],
                            'timestamp': dados['timestamp']
                        })
                        
            except Exception as e:
                print(f"⚠️ Erro ao ler insights de {arquivo}: {e}")
        
        return sorted(insights_similares, key=lambda x: x['insight']['importancia'], reverse=True)
    
    def _verificar_similaridade(self, contexto1: Dict[str, Any], contexto2: Dict[str, Any]) -> bool:
        """
        Verifica se dois contextos são similares.
        
        Args:
            contexto1: Primeiro contexto
            contexto2: Segundo contexto
            
        Returns:
            True se os contextos são similares
        """
        # Implementação simples baseada em chaves comuns
        chaves1 = set(contexto1.keys())
        chaves2 = set(contexto2.keys())
        
        intersecao = chaves1 & chaves2
        uniao = chaves1 | chaves2
        
        if not uniao:
            return False
        
        similaridade = len(intersecao) / len(uniao)
        return similaridade > 0.3  # Threshold de similaridade