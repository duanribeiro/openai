from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np

class DQNAgent:
    def __init__(self, qtde_entradas_ambiente, qtde_entradas_agente):
        self.qtde_entradas_ambiente = qtde_entradas_ambiente
        self.qtde_entradas_agente = qtde_entradas_agente
        self.memoria = deque(maxlen=2000) # list-like container with fast appends and pops on either end
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # TAXA DE DESCONTO - determines the importance of future rewards.
        # A factor of 0 will make the agent "myopic" (or short-sighted) by only considering current rewards,
        # while a factor approaching 1 will make it strive for a long-term high reward.
        self.gamma = 0.95

        # TAXA DE EXPLORAÇÃO - Determines to what extent newly acquired information overrides old information.
        # A factor of 0 makes the agent learn nothing (exclusively exploiting prior knowledge),
        # while a factor of 1 makes the agent consider only the most recent information (ignoring prior knowledge to explore possibilities)
        self.epsilon = 1.0

        self.modelo_neural = self.gerador_modelo_neural()

    def gerador_modelo_neural(self):
        modelo_neural = Sequential()
        modelo_neural.add(Dense(24, input_dim=self.qtde_entradas_ambiente, activation='relu'))
        modelo_neural.add(Dense(24, activation='relu'))
        modelo_neural.add(Dense(self.qtde_entradas_agente, activation='linear'))

        funcao_otimizadora = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # https://keras.io/optimizers/
        modelo_neural.compile(loss='mse', optimizer=funcao_otimizadora)

        return modelo_neural

    def guardar_memoria(self, entradas_ambiente, movimento, recompensa_acumulada, proximas_entradas_ambiente, jogo_acabou):
        self.memoria.append((entradas_ambiente, movimento, recompensa_acumulada, proximas_entradas_ambiente, jogo_acabou))

    def faz_algo(self, entradas_ambiente):
        # Coloca uma aleatoridade em 'testar" novos movimentos
        if np.random.rand() <= self.epsilon:
            movimento_aleatorio = random.randrange(self.qtde_entradas_agente)
            return movimento_aleatorio

        # Escolhe o melhor movimento baseado na recompensa
        movimento_pensado = self.modelo_neural.predict(entradas_ambiente)[0]
        movimento_pensado = np.argmax(movimento_pensado)
        return movimento_pensado

    def replay(self, tamanho_amostra):
        amostra = random.sample(self.memoria, tamanho_amostra)
        for entradas_ambiente, movimento, recompensa_acumulada, proximas_entradas_ambiente, jogo_acabou in amostra:
            if jogo_acabou:
                objetivo = recompensa_acumulada
            else:
                objetivo = recompensa_acumulada + self.gamma * np.amax(self.modelo_neural.predict(proximas_entradas_ambiente)[0])

            objetivo_futuro = self.modelo_neural.predict(entradas_ambiente)
            objetivo_futuro[0][movimento] = objetivo

            self.modelo_neural.fit(entradas_ambiente, objetivo_futuro, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    #
    # def load(self, name):
    #     self.model.load_weights(name)
    #
    # def save(self, name):
    #     self.model.save_weights(name)