import numpy as np


class MonteCarloFX:

    def __init__(self, S, K, T, rd, rf, sigma, simulations=100000):

        self.S = S
        self.K = K
        self.T = T
        self.rd = rd
        self.rf = rf
        self.sigma = sigma
        self.simulations = simulations

    def call_price(self):

        Z = np.random.normal(size=self.simulations)

        ST = self.S * np.exp(
            (self.rd - self.rf - 0.5*self.sigma**2)*self.T
            + self.sigma*np.sqrt(self.T)*Z
        )

        payoff = np.maximum(ST-self.K,0)

        price = np.exp(-self.rd*self.T)*np.mean(payoff)

        return price


    def put_price(self):

        Z = np.random.normal(size=self.simulations)

        ST = self.S * np.exp(
            (self.rd-self.rf-0.5*self.sigma**2)*self.T
            + self.sigma*np.sqrt(self.T)*Z
        )

        payoff = np.maximum(self.K-ST,0)

        price = np.exp(-self.rd*self.T)*np.mean(payoff)

        return price