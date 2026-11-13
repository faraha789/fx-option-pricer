import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
def get_fx_spot(symbol="EUR/USD"):

    url = "https://api.twelvedata.com/price"

    params = {
        "symbol": symbol,
        "apikey": API_KEY
    }

    response = requests.get(url, params=params)

    data = response.json()

    if "price" in data:
        return float(data["price"])

    raise Exception(data)


def get_market_quotes():

    symbols = "EUR/USD,GBP/USD,USD/JPY,USD/CHF"

    url = "https://api.twelvedata.com/quote"

    params = {
        "symbol": symbols,
        "apikey": API_KEY
    }

    response = requests.get(url, params=params)

    return response.json()