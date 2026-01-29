import yfinance as yf
import pandas as pd


def load_data(tickers, start, end):
    data = {}

    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        df["Ticker"] = ticker
        data[ticker] = df

    return data
