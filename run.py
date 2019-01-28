import gym
import numpy as np
from deep_learning import DQNAgent

'''
• ENTRADAS AMBIENTE
	             	Min	    Max
0	posicao 	   -1.2	    0.6
1	velocidade	   -0.07	0.07

• ENTRADA AGENTE
	Ação
0	aperta esquerda
1	nao faz nada
2	aperta direita

• RECOMPENSA
(-1) para cada frame, até o objetivo da posição 0.5 ser alcançada.
'''

# Carregando ambiente do jogo
jogo = gym.make('MountainCar-v0')

# Separando universo de entradas
qtde_entradas_ambiente = jogo.observation_space.shape[0]
qtde_entradas_agente = jogo.action_space.n

# Iniciando o agente neural
agente_neural = DQNAgent(qtde_entradas_ambiente, qtde_entradas_agente)

for tentativa in range(100):
    recompensa_acumulada = 0
    entradas_ambiente = jogo.reset().reshape(1, 2)
    for frame in range(250):
        jogo.render()
        movimento = agente_neural.faz_algo(entradas_ambiente)
        proximas_entradas_ambiente, recompensa, jogo_acabou, info = jogo.step(movimento)
        proximas_entradas_ambiente = proximas_entradas_ambiente.reshape(1, 2)
        recompensa_acumulada = round(proximas_entradas_ambiente[0][0] * 10) + round(proximas_entradas_ambiente[0][1] * 1000)

        agente_neural.guardar_memoria(entradas_ambiente, movimento, recompensa_acumulada, proximas_entradas_ambiente, jogo_acabou)
        entradas_ambiente = proximas_entradas_ambiente

        print("tentativa: {}, frame: {}, score: {}, tx exploração: {:.2}".format(tentativa, frame, recompensa_acumulada, agente_neural.epsilon))

        if len(agente_neural.memoria) > 10:
            agente_neural.replay(10)
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")