import gym
import numpy as np
from DQN import DQNAgent

done = False
tamanho_amostra = 32



'''
• ENTRADAS AMBIENTE
	             	Min	    Max
0	posicao 	   -1.2	    0.6
1	velocidade	   -0.07	0.07

• ENTRADA AGENTE
	Action
0	push left
1	no push
2	push right

• RECOMPENSA
(-1) for each time step, until the goal position of 0.5 is reached.
'''

# Carregando ambiente do jogo
jogo = gym.make('MountainCar-v0')

# Separando universo de entradas
qtde_entradas_ambiente = jogo.observation_space.shape[0]
qtde_entrada_agente = jogo.action_space.n

# Iniciando o agente neural
agente_neural = DQNAgent(qtde_entradas_ambiente, qtde_entrada_agente)

for tentativa in range(100):
    recompensa_acumulada = 0
    entradas_ambiente = jogo.reset().reshape(1, qtde_entradas_ambiente)
    for frame in range(500):
        jogo.render()
        movimento = agente_neural.faz_algo(entradas_ambiente)
        proximas_entradas_ambiente, recompensa, jogo_acabou, info = jogo.step(movimento)
        proximas_entradas_ambiente = proximas_entradas_ambiente.reshape(1, qtde_entradas_ambiente)
        recompensa_acumulada = recompensa + recompensa_acumulada

        agente_neural.guardar_memoria(entradas_ambiente, movimento, recompensa_acumulada, proximas_entradas_ambiente, jogo_acabou)
        entradas_ambiente = proximas_entradas_ambiente

        print("tentativa: {}, frame: {}, score: {}, tx exploração: {:.2}".format(tentativa, frame, recompensa_acumulada, agente_neural.epsilon))

        # if len(agente_neural.memoria) > tamanho_amostra:
        #     agente_neural.replay(tamanho_amostra)
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")