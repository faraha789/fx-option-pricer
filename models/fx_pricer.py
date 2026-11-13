import numpy as np
from scipy.stats import norm


class GarmanKohlhagen:

    def __init__(self, S, K, T, rd, rf, sigma):

        self.S = S
        self.K = K
        self.T = T
        self.rd = rd
        self.rf = rf
        self.sigma = sigma

    # =====================================================
    # d1 et d2
    # =====================================================

    def d1(self):

        return (
            np.log(self.S / self.K)
            + (self.rd - self.rf + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    def d2(self):

        return self.d1() - self.sigma * np.sqrt(self.T)

    # =====================================================
    # Prix
    # =====================================================

    def call_price(self):

        d1 = self.d1()
        d2 = self.d2()

        return (
            self.S * np.exp(-self.rf * self.T) * norm.cdf(d1)
            -
            self.K * np.exp(-self.rd * self.T) * norm.cdf(d2)
        )

    def put_price(self):

        d1 = self.d1()
        d2 = self.d2()

        return (
            self.K * np.exp(-self.rd * self.T) * norm.cdf(-d2)
            -
            self.S * np.exp(-self.rf * self.T) * norm.cdf(-d1)
        )

    # =====================================================
    # GREEKS
    # =====================================================

    def delta(self, option_type="Call"):

        d1 = self.d1()

        if option_type == "Call":
            return np.exp(-self.rf * self.T) * norm.cdf(d1)

        return np.exp(-self.rf * self.T) * (norm.cdf(d1) - 1)

    def gamma(self):

        d1 = self.d1()

        return (
            np.exp(-self.rf * self.T)
            * norm.pdf(d1)
            /
            (self.S * self.sigma * np.sqrt(self.T))
        )

    def vega(self):

        d1 = self.d1()

        return (
            self.S
            * np.exp(-self.rf * self.T)
            * norm.pdf(d1)
            * np.sqrt(self.T)
        ) / 100

    def theta(self, option_type="Call"):

        d1 = self.d1()
        d2 = self.d2()

        first = (
            -self.S
            * np.exp(-self.rf * self.T)
            * norm.pdf(d1)
            * self.sigma
            / (2 * np.sqrt(self.T))
        )

        if option_type == "Call":

            second = (
                self.rf
                * self.S
                * np.exp(-self.rf * self.T)
                * norm.cdf(d1)
            )

            third = (
                self.rd
                * self.K
                * np.exp(-self.rd * self.T)
                * norm.cdf(d2)
            )

            return (first + second - third) / 365

        second = (
            self.rf
            * self.S
            * np.exp(-self.rf * self.T)
            * norm.cdf(-d1)
        )

        third = (
            self.rd
            * self.K
            * np.exp(-self.rd * self.T)
            * norm.cdf(-d2)
        )

        return (first - second + third) / 365

    def rho(self, option_type="Call"):

        d2 = self.d2()

        if option_type == "Call":

            return (
                self.K
                * self.T
                * np.exp(-self.rd * self.T)
                * norm.cdf(d2)
            ) / 100

        return (
            -self.K
            * self.T
            * np.exp(-self.rd * self.T)
            * norm.cdf(-d2)
        ) / 100

    # =====================================================
    # AGRÉGATEUR - nécessaire pour l'interface Streamlit
    # qui fait : delta, gamma, vega, theta, rho = model.greeks(option_type)
    # =====================================================

    def greeks(self, option_type="Call"):

        return (
            self.delta(option_type),
            self.gamma(),
            self.vega(),
            self.theta(option_type),
            self.rho(option_type),
        )


if __name__ == "__main__":

    model = GarmanKohlhagen(
        S=1.10,
        K=1.12,
        T=1,
        rd=0.03,
        rf=0.02,
        sigma=0.15
    )

    print("Call :", model.call_price())
    print("Put  :", model.put_price())

    print("Delta :", model.delta("Call"))
    print("Gamma :", model.gamma())
    print("Vega  :", model.vega())
    print("Theta :", model.theta("Call"))
    print("Rho   :", model.rho("Call"))

    print("Greeks (tuple) :", model.greeks("Call"))