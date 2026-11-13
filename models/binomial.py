import numpy as np


class BinomialFX:


    def __init__(
        self,
        S,
        K,
        T,
        rd,
        rf,
        sigma,
        steps=500
    ):

        self.S = S
        self.K = K
        self.T = T
        self.rd = rd
        self.rf = rf
        self.sigma = sigma
        self.steps = steps



    def price(self, option_type="Call"):


        dt = self.T / self.steps


        u = np.exp(
            self.sigma * np.sqrt(dt)
        )


        d = 1/u


        p = (
            np.exp(
                (self.rd-self.rf)*dt
            ) - d
        ) / (u-d)



        # Valeurs finales

        prices = np.zeros(self.steps+1)


        for i in range(self.steps+1):

            ST = (
                self.S *
                u**i *
                d**(self.steps-i)
            )


            if option_type == "Call":

                prices[i] = max(
                    ST-self.K,
                    0
                )

            else:

                prices[i] = max(
                    self.K-ST,
                    0
                )


        # Remonter dans l'arbre

        discount = np.exp(
            -self.rd*dt
        )


        for j in range(self.steps):

            for i in range(
                self.steps-j
            ):

                prices[i] = discount * (
                    p*prices[i+1]
                    +
                    (1-p)*prices[i]
                )


        return prices[0]



    def call_price(self):

        return self.price("Call")



    def put_price(self):

        return self.price("Put")