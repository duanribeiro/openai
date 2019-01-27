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
        self.memory = deque(maxlen=2000)
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

        optmizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # https://keras.io/optimizers/
        modelo_neural.compile(loss='mse', optimizer=optmizer)

        return modelo_neural

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def faz_algo(self, entradas_ambiente):
        # Coloca uma aleatoridade em 'testar" novos movimentos
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.qtde_entradas_agente)

        # Escolhe o melhor movimento baseado na recompensa
        act_values = self.modelo_neural.predict(entradas_ambiente)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)