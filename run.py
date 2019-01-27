import gym
import numpy as np
from DQN import DQNAgent

done = False
batch_size = 32


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
    entradas_ambiente = jogo.reset()
    entradas_ambiente = np.reshape(entradas_ambiente, [1, qtde_entradas_ambiente])
    for frame in range(500):
        jogo.render()
        movimento = agente_neural.faz_algo(entradas_ambiente)
        proximas_entradas_ambiente, recompensa, done, info = jogo.step(movimento)


        reward = reward + reward if not done else -10
        print(reward)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(tentativa, EPISODES, frame, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")