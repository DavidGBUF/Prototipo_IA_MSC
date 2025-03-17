
#Para executar
#cd "/home/lacis/Downloads/Proj laboratorio/SFC_Reinf_Learning" python -m Env_utils.Env_util

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import copy
from Topology.Paloalto import find_path_with_bandwidth
from SFC_utils.Sfc_util import *
import random

def init_link_usage(G):
    """
    Inicializa o uso dos links na rede.
    """
    return {tuple(sorted((u, v))): 0 for u, v in G.edges()}

def calculate_cost(
    chosen_node, last_chosen_node, server_resources, service, link_usage, G,
    pesos={"cpu": 1, "cache": 1, "bandwidth": 1}
):
    """
    Calcula o custo de alocação de um serviço em um nó, considerando recursos do servidor e uso dos links.
    """
    total_cost = 0
    service_key = service[0]
    service_costs = service[1]

    # Pesos para os recursos
    peso_cpu = pesos['cpu']
    peso_cache = pesos['cache']
    peso_bandwidth = pesos['bandwidth']

    if service_key not in server_resources[chosen_node]["reuse"]:
        if (
            server_resources[chosen_node]['cpu'] <= service_costs['cpu'] or
            server_resources[chosen_node]['cache'] <= service_costs['cache'] or
            server_resources[chosen_node]['cpu'] - service_costs['cpu'] < 5 or
            server_resources[chosen_node]['cache'] - service_costs['cache'] < 5
        ):

            return float('inf'), server_resources, link_usage

        total_cost += (
            (service_costs['cpu'] / server_resources[chosen_node]['cpu'] + 1) ** peso_cpu +
            (service_costs['cache'] / server_resources[chosen_node]['cache'] + 1) ** peso_cache
        )
        server_resources[chosen_node]['cpu'] -= service_costs['cpu']
        server_resources[chosen_node]['cache'] -= service_costs['cache']

        if service_costs["shareable"]:
            server_resources[chosen_node]["reuse"].append(service_key)
    else:
        total_cost -= 5

    if chosen_node == 0:
        total_cost += 50

    if chosen_node != last_chosen_node and last_chosen_node is not None:
        try:
            bandwidth_required = service_costs['bandwidth_output']
            path = find_path_with_bandwidth(G, last_chosen_node, chosen_node, bandwidth_required)
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                edge = tuple(sorted((u, v)))
                link_usage[edge] = link_usage.get(edge, 0) + bandwidth_required
                total_cost += peso_bandwidth * 1
        except nx.NetworkXNoPath:
            return float('inf'), server_resources, link_usage

    return total_cost, server_resources, link_usage


import random

class NetworkEnv(gym.Env):
    """
    Ambiente de rede para alocação de serviços em uma topologia de rede.
    """
    def __init__(self, num_nodes, graph, List_sessions_sfc, pesos=None, num_scenarios=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.sessions_to_allocate = List_sessions_sfc
        self.sessions_to_allocate_backup = copy.deepcopy(List_sessions_sfc)
        self.Allocated_sessions = []
        self.current_sfc_index = 0
        self.current_SF_index = 0
        self.current_session_index = 0
        self.last_chosen_node = None
        self.pesos = pesos or {"cpu": 1, "cache": 1, "bandwidth": 1}
        self.G = graph
        self.graphBackup = graph.copy()
        self.link_usage = init_link_usage(self.G)
        self.proposta_all_sessions = []
        self.proposta_session = []
        self.proposta_allocation_sfc = []
        self.show_resources = True
        self.totalreward = 0
        self.server_resources = self.initialize_server_resources(num_nodes)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            # Dados de CPU + cache, depois o último nó de serviço, quais SF são reusáveis e atual SF
            shape=(num_nodes * 3 + num_nodes + 2 * len(self.sessions_to_allocate[0][0]["SFs"]),),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_nodes)
        self.predefined_scenarios = [self.sessions_to_allocate]
        
        # Gerando múltiplos cenários adicionais
        for _ in range(1, num_scenarios):
            aux = generate_list_sessions(self.G, len(List_sessions_sfc))
            self.predefined_scenarios.append(aux)

    def initialize_server_resources(self, num_nodes):
        return {i: {'cpu': 100, 'cache': 100, 'reuse': []} for i in range(0, num_nodes)}

    def reset(self, seed=None, options=None):
        self.Allocated_sessions = []
        self.server_resources = self.initialize_server_resources(self.num_nodes)
        self.link_usage = init_link_usage(self.G)
        self.current_sfc_index = 0
        self.current_SF_index = 0
        self.current_session_index = 0
        self.last_chosen_node = None
        self.proposta_all_sessions = []
        self.proposta_allocation_sfc = []
        self.allproposta = []

        # Escolher um cenário aleatoriamente
        self.current_scenario_index = random.randint(0, len(self.predefined_scenarios) - 1)
        

        self.sessions_to_allocate = copy.deepcopy(self.predefined_scenarios[self.current_scenario_index])

        self.totalreward = 0

        return self.get_normalized_state(), {}


    def step(self, action):
        reward = 0
        if not 0 <= action < self.num_nodes:
            raise ValueError(f"Ação inválida: {action}. Deve estar entre 0 e {self.num_nodes - 1}.")
        
        chosen_node = int(action)

        # Obtém a sessão e serviço atuais
        current_sfc = self.sessions_to_allocate[0][self.current_sfc_index]
        current_service_key = f"s{self.current_SF_index}"
        service = (current_service_key, current_sfc["SFs"][current_service_key])

        last_service_key, _ = list(current_sfc["SFs"].items())[-1]
        
        # Penaliza se a alocação final não for no nó correto
        if current_service_key == last_service_key and chosen_node != current_sfc["last_service_server"]:
            chosen_node = current_sfc["last_service_server"]
            reward = -15
        elif chosen_node == current_sfc["last_service_server"]:
            reward = 5

        self.proposta_allocation_sfc.append(chosen_node)
        done = False

        # Calcula custo da alocação
        cost, self.server_resources, self.link_usage = calculate_cost(
            chosen_node,
            self.last_chosen_node,
            self.server_resources,
            service,
            self.link_usage,
            self.G,
            self.pesos
        )

        # **Condição 1: Custo Infinito → Encerrar episódio**
        if cost == float('inf'):
            reward = -50 * len(self.sessions_to_allocate_backup) * len(self.sessions_to_allocate_backup[0])


            done = True  # Força o fim do episódio
            self.totalreward = reward


        else:
            # Reduz custo da recompensa
            reward -= cost
            self.totalreward += reward

            self.last_chosen_node = chosen_node
            self.current_SF_index += 1

            # Se todos os serviços da SFC foram alocados, avançar para a próxima
            if self.current_SF_index >= len(current_sfc["SFs"]):
                self.proposta_session.append(self.proposta_allocation_sfc)
                self.proposta_allocation_sfc = []
                self.current_SF_index = 0
                self.current_sfc_index += 1

                # Se todas as SFCs da sessão foram alocadas, avançar para a próxima sessão
                if self.current_sfc_index >= len(self.sessions_to_allocate_backup[0]):
                    self.proposta_all_sessions.append(self.proposta_session)
                    self.Allocated_sessions.append(self.sessions_to_allocate.pop(0))  # Remove sessão alocada
                    self.proposta_session = []
                    self.current_sfc_index = 0
                    self.current_session_index += 1
        # **Condição 2: Todas as sessões foram alocadas → Encerrar episódio**
        done = done or (len(self.sessions_to_allocate) == 0)

        # Atualiza estado e recompensa total
        state = self.get_normalized_state()

        return state, reward, done, False, {}


    def get_normalized_state(self):
        state = []
        current_service_key = f"s{self.current_SF_index}"
        current_sfc = self.sessions_to_allocate_backup[0][self.current_sfc_index]

        for node_id, resources in self.server_resources.items():
            cpu = resources['cpu']
            cache = resources['cache']
            state.append(cpu / 100 if cpu > 0 else 0)
            state.append(cache / 100 if cache > 0 else 0)
            state.append(1 if current_service_key in resources['reuse'] else 0)


        if  len(self.sessions_to_allocate)!=0 and self.current_sfc_index < len(self.sessions_to_allocate[0]):
            
            last_service_server = self.sessions_to_allocate[0][self.current_sfc_index]["last_service_server"]
            one_hot_server = [1 if i == last_service_server else 0 for i in range(1, self.num_nodes + 1)]
            state.extend(one_hot_server)

            
            reuse_vector = []
            one_hot_servers_with_reuse = []
            for sf_key, sf in current_sfc["SFs"].items():
                reuse_vector.append(1 if sf["shareable"] and sf_key in self.server_resources[last_service_server]["reuse"] else 0)
                
            
            state.extend(reuse_vector)

            
            


            num_sfs = len(current_sfc["SFs"])
            
            sf_one_hot = [1 if i == self.current_SF_index else 0 for i in range(num_sfs)]
            state.extend(sf_one_hot)

        else:
            num_sfs = len(self.sessions_to_allocate_backup[0][self.current_sfc_index - 1]["SFs"]) if self.current_sfc_index >= 0 else 0
            state.extend([0] * (self.num_nodes + num_sfs + num_sfs))

        return np.array(state, dtype=np.float32)

    def render(self):
        if self.show_resources:
            print("Recursos dos nós:")
            for node_id, resources in self.server_resources.items():
                print(f"Nó {node_id}: {resources}")
            print("Uso de links:")
            print(self.link_usage)

    def close(self):
        pass

    def get_routes(self):
        routes = {}
        graph = self.graphBackup.copy()

        for index_Session, allocation_SFCs in enumerate(self.proposta_all_sessions):
            routes[f"Session_ID{index_Session}"] = {}

            for index_SFC, list_proposta_SFC in enumerate(allocation_SFCs):
                routes[f"Session_ID{index_Session}"][f"SFC_ID_{index_SFC}"] = {}
                for index_node, node in enumerate(list_proposta_SFC):
                    if not list_proposta_SFC:
                        print(f"SFC_{index_SFC} está vazia.")
                        continue
                    if node not in graph:
                        print(f"Nó {node} não existe no grafo para SFC_{index_SFC}, SF_{index_node}.")
                        continue

                    source_node = 0 if index_node == 0 else list_proposta_SFC[index_node - 1]

                    bandwidth_required = 0 if source_node == 0 else self.Allocated_sessions[0][index_SFC]["SFs"][f"s{index_node}"]["bandwidth_output"]
                    target_node = node

                    try:
                        path, graph = find_path_with_bandwidth(graph, source_node, target_node, bandwidth_required, routes=True)
                        if path is None:
                            print(f"Não foi possível encontrar caminho para SFC_{index_SFC}, SF_{index_node}.")
                            continue
                        routes[f"Session_ID{index_Session}"][f"SFC_ID_{index_SFC}"][f"Sf_{index_node}"] = path
                    except Exception as e:
                        print(f"Erro ao calcular a rota para SFC_{index_SFC}, SF_{index_node}: {e}")
                        continue

        return routes
