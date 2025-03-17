# --- Requisitos dos Serviços ---
# Define os recursos necessários para cada serviço, como CPU, cache e largura de banda.
import random

def generate_random_sfc():
    """
    Gera uma cadeia de funções de serviço (SFC - Service Function Chain) com requisitos de recursos aleatórios.
    Cada função de serviço (SF - Service Function) possui:
        - 'cpu': Quantidade de recursos de CPU exigidos (aleatório entre 10 e 30).
        - 'cache': Quantidade de memória cache exigida (aleatório entre 10 e 30).
        - 'bandwidth_output': Largura de banda de saída exigida (aleatório entre 10 e 30).
        - 'shareable': Indica se o serviço pode ser compartilhado por múltiplos clientes (True ou False).
    O serviço 's4' é uma exceção, pois não exige recursos.

    :return: Um dicionário representando o modelo SFC gerado.
    """
    sfc_model = {
        f's{i}': {'cpu': random.randint(10, 30), 'cache': random.randint(10, 30),
                  'bandwidth_output': random.randint(10, 30), 'shareable': random.choice([True, False])}
        for i in range(4)
    }
    sfc_model['s4'] = {'cpu': 0, 'cache': 0, 'bandwidth_output': 0, 'shareable': False}  # Serviço 4 sem requisitos de recursos.
    return sfc_model

def generate_session(network_topology, size_session=4):
    """
    Gera uma sessão de cadeias de funções de serviço (SFCs).
    Cada sessão contém um conjunto de SFCs com requisitos de recursos aleatórios.
    
    :param network_topology: Topologia da rede para alocação das SFCs.
    :param size_session: Número de SFCs na sessão.
    :return: Lista de SFCs na sessão.
    """
    sfc_model = generate_random_sfc()
    
    session = []
    for i in range(size_session):
        sfc = {
            "sfc_id": i,  # Identificador único da SFC.
            "SFs": sfc_model,  # Cadeia de funções de serviço.
            "last_service_server": random.randint(1, len(network_topology) - 1)  # Último servidor usado pela SFC.
        }
        session.append(sfc)
    return session

def generate_list_sessions(network_topology, quant_sessions=1, SFCs_per_session=4):
    """
    Gera uma lista de sessões contendo múltiplas cadeias de funções de serviço (SFCs).

    :param network_topology: Topologia da rede para alocação das SFCs.
    :param quant_sessions: Quantidade de sessões a serem geradas.
    :return: Lista de sessões, onde cada sessão é uma lista de SFCs.
    """
    aux = [generate_session(network_topology, SFCs_per_session) for _ in range(quant_sessions)]
    
    return aux

def visualize_SFCs(list_SFCs):
    """
    Visualiza os requisitos das SFCs geradas de forma organizada.

    :param list_SFCs: Lista de dicionários contendo os requisitos dos serviços.
    """
    for SFC in list_SFCs:
        print(f"SFC ID: {SFC['sfc_id']}")
        for sf, resources in SFC['SFs'].items():
            print(f"  {sf}: {resources}")
        print(f"  Último servidor: {SFC['last_service_server']}")
        print("-" * 40)

def visualize_sessions(list_sessions):
    """
    Visualiza as sessões e suas respectivas SFCs.

    :param list_sessions: Lista de sessões contendo as SFCs.
    """
    for session_index, session in enumerate(list_sessions):  # Corrigido o erro de desempacotamento incorreto
        print(f"Session ID: {session_index}")
        print("SFCs:")
        visualize_SFCs(session)
        print("-" * 40)

