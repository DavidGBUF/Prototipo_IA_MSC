
# Importação de Bibliotecas

from datetime import datetime
# Bibliotecas padrão

# Bibliotecas de visualização
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos e visualizações.

# Gymnasium - Framework para criação de ambientes de simulação
import gymnasium as gym  # Criação de ambientes de aprendizado por reforço.
from gymnasium import spaces  # Espaços de ação e observação em ambientes Gymnasium.
from stable_baselines3 import PPO  # Importação do algoritmo Proximal Policy Optimization (PPO), usado em aprendizado por reforço para treinar agentes.


from stable_baselines3.common.callbacks import BaseCallback

from Env_utils.Env_util import *
from Topology import Paloalto
from SFC_utils.Sfc_util import *

from stable_baselines3.common.env_checker import check_env

network_topology = Paloalto.create_network_topology()
G = Paloalto.initialize_graph(network_topology)
# Criação do ambiente
env = NetworkEnv(
    num_nodes=37,
    graph=G,
    List_sessions_sfc=generate_list_sessions(G), num_scenarios=10
)

# Verifica a conformidade do ambiente
check_env(env, warn=True)

from SFC_utils.Sfc_util import generate_list_sessions


# Geração dos requisitos das SFCs
num_sfcs = 8


list_sessions = generate_list_sessions(G,4,4)



# Número de nós na rede
num_nodes = len(network_topology)

# Inicialização do ambiente
env = NetworkEnv(num_nodes=num_nodes, graph=G, List_sessions_sfc=list_sessions, 
                 pesos={"cpu": 1.0, "cache": 1.0, "bandwidth": 1.0}, num_scenarios=25000)

# Criar callback para rastrear recompensas

episodios_log = 60*8


# Treinamento do modelo
model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, batch_size=64, n_steps=4096)
model.learn(total_timesteps=int(143000 *episodios_log))
model.save("ppo_sfc_allocation")








