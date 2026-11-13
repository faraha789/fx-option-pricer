from marketdata import get_fx_spot
from marketdata import get_market_quotes

spot = get_fx_spot("EUR/USD")

print("EUR/USD Spot :", spot)

data = get_market_quotes()

print(data)