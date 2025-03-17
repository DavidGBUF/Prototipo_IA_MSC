
# Gymnasium - Framework para criação de ambientes de simulação
import gymnasium as gym  # Criação de ambientes de aprendizado por reforço.
from gymnasium import spaces  # Espaços de ação e observação em ambientes Gymnasium.
import networkx as nx 
import numpy as np

from Topology.Paloalto import find_path_with_bandwidth
from SFC_utils.Sfc_util import *
import random

#Para executar
#cd "/home/lacis/Downloads/Proj laboratorio/SFC_Reinf_Learning" python -m Env_utils.Env_util



def init_link_usage(G):
    """
    Inicializa o uso dos links na rede.

    Args:
        G (networkx.Graph): Grafo representando a rede.

    Retorna:
        dict: Dicionário onde as chaves são arestas (ordenadas) e os valores são inicializados como 0.
    """
    return {tuple(sorted((u, v))): 0 for u, v in G.edges()}


def calculate_cost(
    chosen_node,
    last_chosen_node,
    server_resources,
    service,
    link_usage,
    G,
    pesos={"cpu": 1, "cache": 1, "bandwidth": 1},
):
    """
    Calcula o custo total de alocação de um serviço em um nó, considerando recursos do servidor, reutilização de serviços
    e uso de links da rede. Penalidades e incentivos são aplicados dependendo das condições.

    :param chosen_node: O nó onde o serviço será alocado.
    :param last_chosen_node: O último nó usado na alocação (para cálculo de custo de links).
    :param server_resources: Dicionário de recursos disponíveis para cada nó (CPU, cache, reutilização).
    :param service: Tupla contendo a chave do serviço e seus requisitos de recursos.
    :param link_usage: Dicionário rastreando o uso de largura de banda nos links da rede.
    :param G: Grafo representando a topologia da rede (NetworkX).
    :param pesos: Dicionário de pesos para CPU, cache e largura de banda nos cálculos de custo.
    :return: Tupla com o custo total, recursos atualizados do servidor e uso atualizado dos links.
    """
    total_cost = 0  # Inicializa o custo total.
    service_key = service[0]  # Identificador da função de serviço (SF).
    service_costs = service[1]  # Requisitos do serviço.

    # Pesos para os recursos nos cálculos de custo.
    peso_cpu = pesos['cpu']
    peso_cache = pesos['cache']
    peso_bandwidth = pesos['bandwidth']

    # Verifica se o serviço ainda não está em reuso no nó escolhido.
    if service_key not in server_resources[chosen_node]["reuse"]:
        # Verifica se o nó possui recursos suficientes para atender ao serviço.
        if (
            server_resources[chosen_node]['cpu'] <= service_costs['cpu'] or
            server_resources[chosen_node]['cache'] <= service_costs['cache'] or
            server_resources[chosen_node]['cpu'] - service_costs['cpu'] < 5 or
            server_resources[chosen_node]['cache'] - service_costs['cache'] < 5
        ):
            # Retorna custo infinito caso os recursos sejam insuficientes.
            return float('inf'), server_resources, link_usage

        # Custo baseado na proporção de recursos consumidos.
        total_cost += (
            (service_costs['cpu'] / server_resources[chosen_node]['cpu'] + 1) ** peso_cpu +
            (service_costs['cache'] / server_resources[chosen_node]['cache'] + 1) ** peso_cache
        )
        # Atualiza os recursos disponíveis no nó após a alocação do serviço.
        server_resources[chosen_node]['cpu'] -= service_costs['cpu']
        server_resources[chosen_node]['cache'] -= service_costs['cache']

        # Se o serviço for compartilhável, adiciona à lista de reutilização do nó.
        if service_costs["shareable"]:
            server_resources[chosen_node]["reuse"].append(service_key)
    else:
        # Incentivo para reutilizar serviços já alocados no nó.
        total_cost -= 5

    # Penalidade para evitar o uso de um nó especial (pseudo cloud) representado por '0'.
    if chosen_node == 0:
        total_cost += 50

    # Cálculo do custo de uso de links (entre o nó atual e o último nó).
    if chosen_node != last_chosen_node and last_chosen_node is not None:
        try:
            bandwidth_required = service_costs['bandwidth_output']
            # Busca o caminho com largura de banda suficiente entre os nós.
            path = find_path_with_bandwidth(G, last_chosen_node, chosen_node, bandwidth_required)
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                edge = tuple(sorted((u, v)))  # Ordena os nós para evitar duplicação.
                # Atualiza o uso de largura de banda nos links do caminho.
                link_usage[edge] = link_usage.get(edge, 0) + bandwidth_required
                total_cost += peso_bandwidth * 1  # Incrementa o custo com base no peso da largura de banda.
        except nx.NetworkXNoPath:
            # Retorna custo infinito se não houver caminho disponível com largura de banda suficiente.
            print("Estourou largura de banda")
            return float('inf'), server_resources, link_usage

    return total_cost, server_resources, link_usage



class NetworkEnv(gym.Env):
    """
    Ambiente de rede baseado no Gym para simular alocação de serviços em uma topologia de rede.
    """

    def __init__(self, num_nodes, graph, service_requirements, pesos=None, num_scenarios=1):
        """
        Inicializa o ambiente de rede.

        Args:
            num_nodes (int): Número de nós na rede.
            graph (networkx.Graph): Grafo representando a topologia da rede.
            service_requirements (list): Lista de SFCs (cadeias de funções de serviço) e seus requisitos.
            pesos (dict, opcional): Pesos para CPU, cache e largura de banda. Padrão: {"cpu": 1, "cache": 1, "bandwidth": 1}.
            num_scenarios (int): Número de cenários predefinidos.
        """
        super().__init__()
        self.num_nodes = num_nodes  # Número total de nós na rede.
        self.service_requirements = service_requirements  # Lista de SFCs e requisitos de serviços.
        self.current_sfc_index = 0  # Índice da SFC atualmente sendo processada.
        self.current_service_index = 0  # Índice da função de serviço atual na SFC.
        self.last_chosen_node = None  # Último nó selecionado para alocação.
        self.pesos = pesos or {"cpu": 1, "cache": 1, "bandwidth": 1}  # Pesos dos custos.
        self.G = graph  # Grafo representando a topologia da rede.
        self.graphBackup = graph.copy()  # Grafo representando a topologia da rede.
        self.link_usage = init_link_usage(self.G)  # Inicializa o uso de links da rede.
        self.allproposta = []  # Histórico de todas as alocações realizadas.
        self.proposta = []  # Alocação atual.
        self.show_resources = True  # Indica se os recursos devem ser exibidos durante a execução.
        self.totalreward = 0  # Recompensa acumulada.

         # Inicializa os recursos dos servidores.
        self.server_resources = self.initialize_server_resources(num_nodes)

        # Define o espaço de observação (estado do ambiente).
        num_services = max(len(sfc["SFs"]) for sfc in self.service_requirements)  # Determina o maior número de SFs em uma SFC.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_nodes * 2 + num_nodes + 2 * len(self.service_requirements[0]["SFs"]),),  # Dimensão do estado.
            dtype=np.float32
        )

        # Define o espaço de ação (nós disponíveis para escolha).
        self.action_space = spaces.Discrete(num_nodes)

        # Predefine cenários de serviço.
        self.predefined_scenarios = [service_requirements]
        for _ in range(1, num_scenarios):
            self.predefined_scenarios.append(generate_list_service_requirements(len(service_requirements), self.G))

        self.current_scenario_index = 0  # Índice do cenário atual.


    def initialize_server_resources(self, num_nodes):
        """
        Inicializa os recursos de todos os nós da rede.

        Args:
            num_nodes (int): Número de nós na rede.

        Returns:
            dict: Dicionário contendo recursos iniciais (CPU, cache e SFs reutilizáveis) de cada nó.
        """
        return {
            i: {'cpu': 100, 'cache': 100, 'reuse': []}
            for i in range(0, num_nodes)
        }

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o estado inicial, reinicializando todos os recursos.

        Args:
            seed (int, opcional): Semente para geração de números aleatórios.
            options (dict, opcional): Parâmetros adicionais.

        Returns:
            tuple: Estado inicial normalizado e informações adicionais.
        """
        # Reinicia recursos dos servidores e uso de links.
        self.server_resources = self.initialize_server_resources(self.num_nodes)
        self.link_usage = init_link_usage(self.G)
        self.current_sfc_index = 0
        self.current_service_index = 0
        self.last_chosen_node = None
        self.proposta = []
        self.allproposta = []

        # Escolhe aleatoriamente um dos cenários predefinidos.
        self.current_scenario_index = random.randint(0, len(self.predefined_scenarios) - 1)
        self.service_requirements = self.predefined_scenarios[self.current_scenario_index]
        self.totalreward = 0

        return self.get_normalized_state(), {}


    def step(self, action):
        """
        Executa uma ação no ambiente, atualizando o estado e calculando recompensas.

        Args:
            action (int): Índice do nó escolhido para alocação.

        Returns:
            tuple: Novo estado, recompensa, indicador de finalização, e informações adicionais.
        """
        reward = 0
        if not 0 <= action < self.num_nodes:
            raise ValueError(f"Ação inválida: {action}. Deve estar entre 0 e {self.num_nodes - 1}.")

        chosen_node = int(action)  # Nó escolhido para alocação.
        current_sfc = self.service_requirements[self.current_sfc_index]  # SFC atual.
        current_service_key = f"s{self.current_service_index}"  # Chave do serviço atual.
        service = (current_service_key, current_sfc["SFs"][current_service_key])  # Requisitos do serviço.

        # Penalidade/recompensa para alocação do último serviço em um nó específico.
        last_service_key, _ = list(current_sfc["SFs"].items())[-1]
        if current_service_key == last_service_key and chosen_node != current_sfc["last_service_server"]:
            chosen_node = current_sfc["last_service_server"]
            reward = -15
        if current_service_key == last_service_key and chosen_node == current_sfc["last_service_server"]:
            reward = 15

        self.proposta.append(chosen_node)
        done = False

        if self.current_sfc_index >= len(self.service_requirements):
            done = True
            return self.get_normalized_state(), 0, done, False, {}

        # Calcula o custo da alocação e atualiza os recursos.
        cost, self.server_resources, self.link_usage = calculate_cost(
            chosen_node,
            self.last_chosen_node,
            self.server_resources,
            service,
            self.link_usage,
            self.G,
            self.pesos
        )

        if cost == float('inf'):
            # Penalidade por falha na alocação.
            reward = -50 * len(self.service_requirements)
            done = True
        else:
            reward = -cost if not reward else (reward - cost)
            self.last_chosen_node = chosen_node

            # Avança para o próximo serviço da SFC.
            self.current_service_index += 1
            if self.current_service_index >= len(current_sfc["SFs"]):
                self.allproposta.append(self.proposta)
                self.proposta = []
                self.current_sfc_index += 1
                self.current_service_index = 0

        # Finaliza o ambiente se todas as SFCs forem processadas.
        done = done or self.current_sfc_index >= len(self.service_requirements)

        state = self.get_normalized_state()
        self.totalreward = reward
        return state, reward, done, False, {}

    def get_normalized_state(self):
        """
        Normaliza os recursos dos servidores e adiciona informações relevantes ao estado.

        Returns:
            np.ndarray: Vetor de estado normalizado.
        """
        state = []

        # Normaliza os recursos dos servidores (CPU e Cache)
        for node_id, resources in self.server_resources.items():
            cpu = resources['cpu']
            cache = resources['cache']
            state.append(cpu / 100 if cpu > 0 else 0)  # CPU disponível
            state.append(cache / 100 if cache > 0 else 0)  # Cache disponível

        # Adiciona informações do último servidor e reutilização de SFs.
        # ... (mantém a lógica existente).

        # Adiciona a codificação one-hot para o último servidor usado (last_service_server)
        if self.current_sfc_index < len(self.service_requirements):
            last_service_server = self.service_requirements[self.current_sfc_index]["last_service_server"]
            one_hot_server = [1 if i == last_service_server else 0 for i in range(1, self.num_nodes + 1)]
            state.extend(one_hot_server)

            # Adiciona as SFs que podem ser reutilizadas
            current_sfc = self.service_requirements[self.current_sfc_index]
            reuse_vector = []
            for sf_key, sf in current_sfc["SFs"].items():
                reuse_vector.append(1 if sf["shareable"] and sf_key in self.server_resources[last_service_server]["reuse"] else 0)
            state.extend(reuse_vector)

            # Adiciona a SF atual como codificação one-hot
            num_sfs = len(current_sfc["SFs"])
            current_service_key = f"s{self.current_service_index}"
            sf_one_hot = [1 if i == self.current_service_index else 0 for i in range(num_sfs)]
            state.extend(sf_one_hot)


        else:
            # Caso inválido, preenche com zeros para manter o formato consistente
            num_sfs = len(self.service_requirements[self.current_sfc_index - 1]["SFs"]) if self.current_sfc_index > 0 else 0
            state.extend([0] * (self.num_nodes + num_sfs + num_sfs))

        return np.array(state, dtype=np.float32)




    def render(self):
        """
        Exibe os recursos dos nós e uso de links para depuração.
        """
        if self.show_resources:
          print("Recursos dos nós:")
          for node_id, resources in self.server_resources.items():
              print(f"Nó {node_id}: {resources}")
          print("Uso de links:")
          print(self.link_usage)

    def close(self):
        """
        Fecha o ambiente (não implementado).
        """
        pass

    def get_routes(self):
        routes = {}
        graph = self.graphBackup.copy()  # Cria uma cópia do grafo original
    
        for index_SFC, allocation_SFC in enumerate(self.allproposta):
            # Inicializa o dicionário para a SFC atual
            routes[f"SFC_{index_SFC}"] = {}
    
            for i, node in enumerate(allocation_SFC):
                # Validações
                if not allocation_SFC:
                    print(f"SFC_{index_SFC} está vazia.")
                    continue
                if node not in graph:
                    print(f"Nó {node} não existe no grafo para SFC_{index_SFC}, SF_{i}.")
                    continue
    
                # Determina a largura de banda necessária
                if "SFs" not in self.service_requirements[0] or f"s{i}" not in self.service_requirements[0]["SFs"]:
                    print(f"Requisitos de serviço ausentes para SFC_{index_SFC}, SF_{i}.")
                    continue
                
                
                source_node = 0 if i == 0 else allocation_SFC[i - 1]
                bandwidth_required = 0 if source_node == 0 else self.service_requirements[0]["SFs"][f"s{i}"]["bandwidth_output"]
                target_node = node
    
                # Encontra o caminho mais curto no grafo filtrado
                try:
                    path, graph = find_path_with_bandwidth(graph, source_node, target_node, bandwidth_required, routes=True)
                    if path is None:
                        print(f"Não foi possível encontrar caminho para SFC_{index_SFC}, SF_{i}.")
                        continue
                    routes[f"SFC_{index_SFC}"][f"SF_{i}"] = path
                except Exception as e:
                    print(f"Erro ao calcular a rota para SFC_{index_SFC}, SF_{i}: {e}")
                    continue
    
        return routes