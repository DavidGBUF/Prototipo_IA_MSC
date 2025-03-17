# Importação de Bibliotecas
import copy


# Bibliotecas padrão
import random  # Biblioteca para geração de números aleatórios e manipulação de listas aleatoriamente.
# Bibliotecas científicas
import numpy as np  # Biblioteca para manipulação eficiente de arrays multidimensionais e operações matemáticas.
# Bibliotecas para grafos e redes
import networkx as nx  # Biblioteca para criação e manipulação de grafos e redes complexas.
# Bibliotecas de visualização
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos e visualizações.

# Gymnasium - Framework para criação de ambientes de simulação
import gymnasium as gym  # Criação de ambientes de aprendizado por reforço.
from gymnasium import spaces  # Espaços de ação e observação em ambientes Gymnasium.
from stable_baselines3 import PPO  # Importação do algoritmo Proximal Policy Optimization (PPO), usado em aprendizado por reforço para treinar agentes.



# --- Configuração da Topologia da Rede ---
def create_network_topology():
    """
    Cria a topologia da rede com nós e conexões.
    Cada nó é representado como uma chave em um dicionário.
    As conexões (arestas) entre os nós incluem atributos, como 'bandwidth' (largura de banda).
    Retorna um dicionário representando a topologia da rede.
    """
    return {
        # Cada nó tem conexões com outros nós especificados no formato:
        # nó_destino: {'bandwidth': valor_da_largura_de_banda}.
        0: {3: {'bandwidth': 1000}},
        1: {2: {'bandwidth': 1000}},
        # O nó 2 está conectado a múltiplos nós (1, 3, 14, 15).
        2: {1: {'bandwidth': 1000}, 3: {'bandwidth': 1000}, 14: {'bandwidth': 1000}, 15: {'bandwidth': 1000}},
        3: {0: {'bandwidth': 1000}, 2: {'bandwidth': 1000}, 4: {'bandwidth': 1000}, 20: {'bandwidth': 1000}},
        # O mesmo padrão é repetido para definir toda a rede.
        # Cada aresta é bidirecional com um atributo de largura de banda (1000).
        # Nós podem ter diferentes números de conexões (graus).
        # Exemplos continuam para os nós 4 a 36...
        4: {3: {'bandwidth': 1000}, 5: {'bandwidth': 1000}},
        5: {4: {'bandwidth': 1000}, 6: {'bandwidth': 1000}},
        6: {5: {'bandwidth': 1000}, 7: {'bandwidth': 1000}},
        7: {6: {'bandwidth': 1000}, 8: {'bandwidth': 1000}, 27: {'bandwidth': 1000}, 29: {'bandwidth': 1000}},
        8: {7: {'bandwidth': 1000}, 9: {'bandwidth': 1000}, 28: {'bandwidth': 1000}},
        9: {8: {'bandwidth': 1000}, 10: {'bandwidth': 1000}},
        10: {9: {'bandwidth': 1000}, 11: {'bandwidth': 1000}},
        11: {10: {'bandwidth': 1000}, 12: {'bandwidth': 1000}, 34: {'bandwidth': 1000}},
        12: {11: {'bandwidth': 1000}, 13: {'bandwidth': 1000}, 31: {'bandwidth': 1000}},
        13: {12: {'bandwidth': 1000}, 14: {'bandwidth': 1000}},
        14: {13: {'bandwidth': 1000}, 2: {'bandwidth': 1000}, 33: {'bandwidth': 1000}, 36: {'bandwidth': 1000}},
        15: {2: {'bandwidth': 1000}, 16: {'bandwidth': 1000}},
        16: {15: {'bandwidth': 1000}, 17: {'bandwidth': 1000}, 25: {'bandwidth': 1000}, 31: {'bandwidth': 1000}},
        17: {16: {'bandwidth': 1000}, 18: {'bandwidth': 1000}},
        18: {17: {'bandwidth': 1000}, 19: {'bandwidth': 1000}, 21: {'bandwidth': 1000}, 24: {'bandwidth': 1000}},
        19: {18: {'bandwidth': 1000}, 20: {'bandwidth': 1000}},
        20: {3: {'bandwidth': 1000}, 19: {'bandwidth': 1000}, 24: {'bandwidth': 1000}},
        21: {18: {'bandwidth': 1000}, 22: {'bandwidth': 1000}},
        22: {21: {'bandwidth': 1000}, 23: {'bandwidth': 1000}, 26: {'bandwidth': 1000}},
        23: {22: {'bandwidth': 1000}, 24: {'bandwidth': 1000}},
        24: {23: {'bandwidth': 1000}, 18: {'bandwidth': 1000}, 20: {'bandwidth': 1000}, 25: {'bandwidth': 1000}},
        25: {24: {'bandwidth': 1000}, 16: {'bandwidth': 1000}, 26: {'bandwidth': 1000}},
        26: {25: {'bandwidth': 1000}, 22: {'bandwidth': 1000}, 32: {'bandwidth': 1000}},
        27: {7: {'bandwidth': 1000}, 28: {'bandwidth': 1000}, 30: {'bandwidth': 1000}},
        28: {27: {'bandwidth': 1000}, 8: {'bandwidth': 1000}},
        29: {7: {'bandwidth': 1000}},
        30: {27: {'bandwidth': 1000}},
        31: {12: {'bandwidth': 1000}, 16: {'bandwidth': 1000}, 32: {'bandwidth': 1000}},
        32: {31: {'bandwidth': 1000}, 26: {'bandwidth': 1000}, 34: {'bandwidth': 1000}, 33: {'bandwidth': 1000}},
        33: {32: {'bandwidth': 1000}, 14: {'bandwidth': 1000}, 35: {'bandwidth': 1000}},
        34: {11: {'bandwidth': 1000}, 32: {'bandwidth': 1000}},
        35: {33: {'bandwidth': 1000}, 36: {'bandwidth': 1000}},
        36: {35: {'bandwidth': 1000}, 14: {'bandwidth': 1000}},
    }


# --- Inicialização do Grafo ---
def initialize_graph(network_topology):
    """
    Inicializa um grafo (Graph) a partir da topologia fornecida.
    Para cada nó e suas conexões, adiciona nós e arestas ao grafo.
    As arestas incluem um atributo de largura de banda ('bandwidth').
    Parâmetros:
        - network_topology: Dicionário contendo a configuração da topologia da rede.
    Retorna:
        - Grafo da biblioteca NetworkX representando a rede.
    """
    G = nx.Graph()  # Cria um grafo vazio.
    for node, edges in network_topology.items():
        # Itera sobre as conexões de cada nó.
        for target, edge_attr in edges.items():
            # Adiciona uma aresta entre o nó atual e o nó de destino com o atributo 'bandwidth'.
            G.add_edge(node, target, bandwidth=edge_attr['bandwidth'])
    return G


# --- Visualização da Topologia ---
def plot_topology(G):
    """
    Gera um gráfico visual da topologia da rede usando Matplotlib e NetworkX.
    Mostra:
        - Nós: representados por círculos.
        - Arestas: representadas por linhas conectando os nós.
        - Atributos de largura de banda: mostrados como rótulos nas arestas.
    Parâmetros:
        - G: Grafo NetworkX representando a topologia da rede.
    """
    pos = nx.spring_layout(G)  # Calcula a posição dos nós para o layout gráfico.
    plt.figure(figsize=(12, 8))  # Define o tamanho da figura.
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color='lightblue',
        font_size=10, font_weight='bold'  # Estilo dos nós e rótulos.
    )
    # Obtém os rótulos das arestas (atributo 'bandwidth').
    labels = nx.get_edge_attributes(G, 'bandwidth')
    # Desenha os rótulos das arestas no gráfico.
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    plt.title("Topologia da Rede")  # Título do gráfico.
    plt.show()  # Exibe o gráfico.


# --- Recursos dos Servidores ---
def initialize_server_resources(num_nodes):
    """
    Inicializa os recursos de cada servidor na rede.
    Cada servidor possui:
        - 'cpu': Total de recursos de CPU (inicialmente 100 unidades).
        - 'cache': Total de memória cache (inicialmente 100 unidades).
        - 'reuse': Lista para rastrear recursos reutilizáveis, inicialmente vazia.
    Parâmetros:
        - num_nodes: Número total de servidores (nós) na rede.
    Retorna:
        - Dicionário onde cada chave é um identificador de servidor (número do nó),
          e o valor é outro dicionário com os recursos associados.
    """
    return {
        # Para cada nó (de 0 a num_nodes - 1), inicializa os recursos.
        i: {'cpu': 100, 'cache': 100, 'reuse': []}
        for i in range(0, num_nodes)
    }



def find_path_with_bandwidth(graph, source_node, target_node, bandwidth_required, routes=False):
    """
    Encontra o caminho mais curto entre dois nós em um grafo, considerando a largura de banda mínima exigida.

    :param graph: O grafo (NetworkX Graph) com as arestas e seus atributos, incluindo largura de banda.
    :param source_node: O nó inicial de onde começa a busca.
    :param target_node: O nó destino onde a busca termina.
    :param bandwidth_required: A largura de banda mínima necessária para considerar as arestas do caminho.
    :param routes: Se True, ajusta a largura de banda no grafo original após encontrar o caminho.
    :return: O caminho mais curto como uma lista de nós, ou None se não houver caminho disponível.
    """
    try:
        # Filtra as arestas do grafo que possuem largura de banda suficiente
        edges_to_keep = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get('bandwidth', 0) >= bandwidth_required
        ]
        # Cria um subgrafo com apenas as arestas que atendem ao requisito de largura de banda
        filtered_graph = graph.edge_subgraph(edges_to_keep).copy()
        
        # Encontra o caminho mais curto no subgrafo filtrado, com base no peso das arestas
        path = nx.shortest_path(filtered_graph, source=source_node, target=target_node, weight='weight')
        
        if routes:
            # Ajusta a largura de banda das arestas no caminho encontrado
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]  # Obtém dois nós consecutivos no caminho
                if graph.has_edge(u, v):  # Verifica se a aresta existe no grafo original
                    # Reduz a largura de banda na aresta (u, v)
                    graph[u][v]['bandwidth'] -= bandwidth_required
                    # Reduz a largura de banda na aresta (v, u), caso o grafo seja bidirecional
                    if graph.has_edge(v, u):
                        graph[v][u]['bandwidth'] -= bandwidth_required
            return path, graph  # Retorna o caminho encontrado e o grafo ajustado
        else:
            return path  # Retorna apenas o caminho encontrado
    except nx.NetworkXNoPath:
        # Lida com o caso em que não há um caminho entre os nós que atenda ao requisito de largura de banda
        print("Não há caminho disponível com a largura de banda necessária.")
        return None
    except nx.NodeNotFound:
        # Lida com o caso em que um ou ambos os nós não estão presentes no grafo
        print("Um ou ambos os nós não foram encontrados no grafo.")
        return None


if __name__ == "__main__":

    network_topology = create_network_topology()
    G = initialize_graph(network_topology)
    plot_topology(G)
